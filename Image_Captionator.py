# ------------------------------ #
# Import Libraries          
# ------------------------------ #
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import datetime
import time
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import efficientnet.tfkeras as efn

efn.preprocess_input
tf.keras.applications.efficientnet.preprocess_input

###############################
# PARAMETERS
##############################

top_k = 10000  # Total Vocab

hyP = {'BATCH_SIZE': 200,
       'BUFFER_SIZE': 1000,
       'embedding_dim': 256,
       'units': 512,
       'vocab_size': top_k + 1,
       'EPOCHS': 20,
       'lr': 1e-4}


# optimizer = tf.keras.optimizers.SGD(lr=hyP['lr'])
optimizer = tf.keras.optimizers.Adam(lr=hyP['lr'])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# ------------------------------ #
# Prepering the dataset
# ------------------------------ #
# IMG_Path   = '/home/vpatel25/APM598/Dataset/Flickr8k_Dataset/'
# Annotation = pd.read_csv('/home/vpatel25/APM598/Dataset/Flickr8k_text/Flickr8k_lemma_token.csv')

IMG_Path = '/home/vpatel25/APM598/Dataset/flickr30k_images/flickr30k_images/'
Annotation = pd.read_csv('/home/vpatel25/APM598/Dataset/flickr30k_images/Results.csv', delimiter='|')

Annotation[' comment'] = '<start> ' + Annotation[' comment'] + ' <end>'
# Annotation['CAPTION']  = '<start> ' + Annotation['CAPTION'] + ' <end>'

# Creating list of image captions and their path
image_path_to_caption = collections.defaultdict(list)

# for z,val in enumerate(Annotation['CAPTION']):
for z, val in enumerate(Annotation[' comment']):
    caption = val
    #   image_path = IMG_Path + Annotation['IMG_NAME'][z]
    image_path = IMG_Path + Annotation['image_name'][z]
    image_path_to_caption[image_path].append(caption)

# Splitting dataset for test/train
image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)
train_image_paths = image_paths[:20000]
print('\n', len(train_image_paths), '\n')

train_captions = []
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    train_captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))


# ------------------------------ #
# Loading Pretrained Network
# ------------------------------ #
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    # img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return img, image_path


# image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
image_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')

new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

print("\n Features Extraction Model \n")

# Get individual images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image).batch(16)

for img, path in image_dataset:
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3]))
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())


# ------------------------------ #
# Tokenize
# ------------------------------ #
# Find the max length of caption
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# Capping max word
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Apply padding to max length (maximum capton length)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)

# ------------------------------ #
# Train/Test Split
# ------------------------------ #
img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
    img_to_cap_vector[img].append(cap)

# Create training and validation sets using an 80-20 split randomly.
img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=42)

print("\n Tokenizer Done \n")

print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

NumSteps = len(img_name_train) // hyP['BATCH_SIZE']


# Load numpy files for images
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')  # ,allow_pickle=True)
    return img_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# parallizing numpy files
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
    map_func, [item1, item2], [tf.float32, tf.int32]))

dataset = dataset.shuffle(hyP['BUFFER_SIZE']).batch(hyP['BATCH_SIZE'])
# dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))

# Use map to load the numpy files in parallel
val_dataset = val_dataset.map(lambda item1, item2: tf.numpy_function(
    map_func, [item1, item2], [tf.float32, tf.int32]))

# No Shuffle for Val and batch
val_dataset = val_dataset.batch(hyP['BATCH_SIZE']).prefetch(buffer_size=hyP['BATCH_SIZE'])


# ------------------------------ #
# MODEL
# ------------------------------ #
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))
        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)
        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)
        # self.fc1 = tf.keras.layers.Dense(embedding_dim // 2)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        # x = self.fc1(x)
        # x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)
        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


encoder = CNN_Encoder(hyP['embedding_dim'])
decoder = RNN_Decoder(hyP['embedding_dim'], hyP['units'], hyP['vocab_size'])


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


Save_Path = '/home/vpatel25/APM598/Results/' + time.strftime('%Y-%m-%d_%H-%M-%S') + '/'
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

PathLinker = tf.train.CheckpointManager(ckpt, Save_Path, max_to_keep=5)


@tf.function  # Non-teacher-forcing val_loss is too complicated at the moment
def val_step(img_tensor, target, teacher_forcing=True):
    loss = 0
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    #   print(dec_input.shape) # (BATCH_SIZE, 1)
    features = encoder(img_tensor)
    #   print(features.shape) # (BATCH_SIZE, IMG_FEAT_LEN, ENCODER_HID) = 64 100 256
    for i in range(1, target.shape[1]):
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        loss += loss_function(target[:, i], predictions)
        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)
    avg_loss = (loss / int(target.shape[1]))
    return loss, avg_loss


def cal_val_loss(val_dataset):
    # target.shape = (64,49) = (BATCH_SIZE, SEQ_LEN)
    val_num_steps = len(img_name_val) // hyP['BATCH_SIZE']
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(val_dataset):
        batch_loss, t_loss = val_step(img_tensor, target)
        total_loss += t_loss
    print('Valid Loss {:.6f}'.format(total_loss / val_num_steps))
    return total_loss / val_num_steps


# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []
val_loss_plot = []


@tf.function
def train_step(img_tensor, target):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)
    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss


print(' TRAINING INITIATION ')

df = pd.DataFrame(columns=('epoch', 'batch', 'loss'))
timer = []
best_val_loss = 100

for epoch in range(0, hyP['EPOCHS']):
    start = time.time()
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
        if batch % 100 == 0:
            average_batch_loss = batch_loss.numpy() / int(target.shape[1])
            print(f'Epoch {epoch} Batch {batch} Loss {average_batch_loss:.4f} ')
            df.loc[epoch] = [epoch, batch, average_batch_loss]
    val_loss = cal_val_loss(val_dataset)
    if val_loss < best_val_loss:
        print('update best val loss from %.4f to %.4f' % (best_val_loss, val_loss))
        best_val_loss = val_loss
        PathLinker.save()
    loss_plot.append(total_loss / NumSteps)
    val_loss_plot.append(val_loss)
    print(f'Epoch {epoch + 1} Loss {total_loss / NumSteps:.6f}')
    print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i,
                                                                                               nvidia_smi.nvmlDeviceGetName(
                                                                                                   handle),
                                                                                               100 * info.free / info.total,
                                                                                               info.total, info.free,
                                                                                               info.used))
    nvidia_smi.nvmlShutdown()

plt.grid()
plt.plot(loss_plot, color='green', marker='o', linestyle='-')
plt.plot(val_loss_plot, color='red', marker='o', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()
# plt.legend('Loss', 'Validation Loss')
plt.savefig(Save_Path + 'Loss_Plot.png')

print(time.time())

df.to_csv(Save_Path + 'Training_Data.csv')


hyP = {'attention_features_shape': bf.shape[0]}

def evaluate(image):
    attention_plot = np.zeros((max_length, hyP['attention_features_shape']))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    print(img_tensor_val.shape)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(20, 20))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(np.ceil(len_result / 2), 2)
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


Val_Path = '/home/vpatel25/APM598/Dataset/Validation_set/'
paths2 = sorted(os.listdir(Val_Path))

for k in range(0, 15):
    image_path2 = Val_Path + paths2[k]
    result = evaluate(image_path2)
    print(image_path2)
    print('Prediction Caption:', ' '.join(result[0]))
    print('\n')
