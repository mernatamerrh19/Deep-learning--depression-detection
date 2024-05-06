from pydub import AudioSegment
import os
from scipy import signal
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
from librosa import display
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow.keras
from tensorflow.keras import models, layers
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Embedding
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, LSTM, Bidirectional
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import itertools
import math
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds
import pathlib
from datasets import Dataset, Features, Audio, Image
from keras import layers, models
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib
matplotlib.use('TkAgg')

num_classes = 2
batch_size = 16
epochs = 10


# tf.random.set_seed(14)
# tf.keras.utils.set_random_seed(103)

tf.random.set_seed(101)
tf.keras.utils.set_random_seed(21)

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
# tf.config.experimental.enable_op_determinism()

# dataset_url = "G:\data\DP-preprocessed-train"
archive = os.path.abspath(r"G:\data\DP-preprocessed-train")
# archive = os.path.abspath(r"G:\data\DP-preprocessed-train - Copy")
# archive = tf.keras.utils.get_file(origin= 'file://'+dataset_url, fname=None)
data_dir = pathlib.Path(archive).with_suffix('')
image_count = len(list(data_dir.glob('*/*.png')))

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, labels='inferred',
     label_mode='binary', validation_split=None, color_mode='rgb', interpolation="bicubic")
class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names)
print("Number of classes:", num_classes)
# train_ds = train_ds.repeat()


archive_v = "G:\data\DP-preprocessed-dev"
# archive_v = "G:\data\DP-preprocessed-dev - Copy"
# archive_v = tf.keras.utils.get_file(origin=dataset_url_v)
data_dir_v = pathlib.Path(archive_v).with_suffix('')
image_count_v = len(list(data_dir_v.glob('*/*.png')))

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_v, labels='inferred',
    label_mode='binary', validation_split=None, color_mode='rgb', interpolation="bicubic")
# val_ds = val_ds.repeat()

archive_test = "G:\data\DP-preprocessed-test"
# archive_test = "G:\data\DP-preprocessed-test - Copy"
# archive_test = tf.keras.utils.get_file(origin=dataset_url_test)
data_dir_test = pathlib.Path(archive_test).with_suffix('')
image_count_test = len(list(data_dir_test.glob('*/*.png')))

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_test, labels='inferred', image_size=(256, 256), shuffle=False,
    label_mode='binary', validation_split=None, color_mode='rgb', interpolation="bicubic")

# Get the file paths and file names from the test dataset
file_paths = test_ds.file_paths
file_names = [tf.strings.split(path, '/')[-1] for path in file_paths]

# Extract the numerical part from the file names to sort them
numerical_file_names = [tf.strings.regex_replace(name, '[^0-9]', '') for name in file_names]

# Convert numerical file names to integers for sorting
numerical_file_names = [tf.strings.to_number(name, out_type=tf.int32) for name in numerical_file_names]

# Get the indices of numerical_file_names
indices = sorted(range(len(numerical_file_names)), key=lambda k: numerical_file_names[k])

# Sort file_paths and file_names based on indices
sorted_file_paths = [file_paths[i] for i in indices]
sorted_file_names = [file_names[i] for i in indices]

# np.array(sorted_file_names)

sorted_labels = []
for file_name_tensor in sorted_file_names:
    file_name = file_name_tensor.numpy().decode('utf-8')
    if 'Depressed' in file_name:
        sorted_labels.append(1)
    else:
        sorted_labels.append(0)


print(sorted_labels)

# Print the sorted file names
print(sorted_file_paths)
print(len(np.array(sorted_file_names)))



# Assuming sorted_file_paths and sorted_labels are already defined

# Function to load and preprocess an image from its path
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)  # Assuming PNG images with 3 channels
    img = tf.image.resize(img, (256, 256))  # Resize the image to (256, 256)
    # img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Build the sorted test dataset
test_ds = tf.data.Dataset.from_tensor_slices((sorted_file_paths, sorted_labels))
test_ds = test_ds.map(lambda x, y: (load_and_preprocess_image(x), y))
test_ds = test_ds.batch(1) #54=18*3

for images, labels in test_ds:
    print(images.shape, labels.shape)  # This will help you confirm the shapes are as expected



# Assuming val_ds contains image data
X_val = []
y_val = []

# Extract images and labels from val_ds

for images, labels in val_ds:
    X_val.extend(images.numpy())
    y_val.extend(labels.numpy())

# Convert lists to numpy arrays
X_val = np.array(X_val)
y_val = np.array(y_val)

# Reshape X_val if needed (assuming images are 4D with shape [batch_size, height, width, channels])
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)

# Define the oversampling and undersampling ratios
# oversample_ratio_dep = 60 / 41  # Oversample 'Depressed' class to have 60 images
# undersample_ratio_not = 60 / 93  # Undersample 'Not' class to have 60 images

class_names = val_ds.class_names
Depressed = class_names.index('Depressed')
Not = class_names.index('Not')

# Initialize the RandomOverSampler and RandomUnderSampler for validation data
#at 180 sec we chose them 40/40
#at 120 sec we chode them 80/80
oversampler = RandomOverSampler(sampling_strategy={Depressed: 40})
undersampler = RandomUnderSampler(sampling_strategy={Not: 40})

# Fit and transform the validation data using oversampling and undersampling
X_val_resampled, y_val_resampled = oversampler.fit_resample(X_val_reshaped, y_val)
X_val_resampled, y_val_resampled = undersampler.fit_resample(X_val_resampled, y_val_resampled)

# Convert the resampled data back to TensorFlow datasets
val_ds = tf.data.Dataset.from_tensor_slices((X_val_resampled.reshape(-1, 256, 256, 3), tf.keras.utils.to_categorical(y_val_resampled, 2)))
val_ds = val_ds.shuffle(len(X_val_resampled)).batch(10) #10*8/ 16*10

print("\nVAL\n")
for images, labels in val_ds:
    print(images.shape, labels.shape)  # This will help you confirm the shapes are as expected


X_train= []
y_train= []

for images, labels in train_ds:
    X_train.extend(images.numpy())
    y_train.extend(labels.numpy())

# # Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape X_train if needed (assuming images are 4D with shape [batch_size, height, width, channels])
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
print(X_train.shape[0])

class_names = train_ds.class_names
Depressed = class_names.index('Depressed')
Not = class_names.index('Not')

# Initialize the RandomOverSampler and RandomUnderSampler for validation data
# at 180 sec we chose them 200/200
# at 120 sec we chose them 270/270
oversampler = RandomOverSampler(sampling_strategy={Depressed: 200})
undersampler = RandomUnderSampler(sampling_strategy={Not: 200})

# Fit and transform the training data using oversampling and undersampling
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_reshaped, y_train)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_resampled, y_train_resampled)

# Convert the resampled data back to TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train_resampled.reshape(-1, 256, 256, 3),  tf.keras.utils.to_categorical(y_train_resampled, 2)))
train_ds = train_ds.shuffle(len(X_train_resampled)).batch(20) #320 (16*20)// #270*2=540 (27*20)
print(len(X_train_resampled))

print("\ntrain\n")
i=0
for images, labels in train_ds:
    print(images.shape, labels.shape)  # This will help you confirm the shapes are as expected
    i=i+1
print(i)
#
train_ds=train_ds.repeat()
val_ds=val_ds.repeat()


# base_model = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3), classes=2, classifier_activation='softmax')
# base_model.trainable = False

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(256, 256, 3), classes=2, classifier_activation='softmax', pooling='min')
base_model.trainable = True

def preprocess(images, labels):
  return preprocess_input(images), labels

train_ds = train_ds.map(preprocess)
test_ds = test_ds.map(preprocess)
val_ds = val_ds.map(preprocess)



# model = models.Sequential([
    # layers.preprocessing.RandomFlip("horizontal"),
    # layers.Rescaling(scale=1./127),
    # base_model,
    # tf.keras.layers.Conv2D(1024, 3, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.Conv2D(128, 3, padding='same',activation='elu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Conv2D(256, 3, padding='same',activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # tf.keras.layers.Dropout(0.25),

    # tf.keras.layers.Conv2D(128, 3, padding='same',activation='relu'),
    # # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.ConvLSTM1D(kernel_size=3, filters=16),

    # # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.5),
    # keras.layers.BatchNormalization(),
    # keras.layers.MaxPooling2D(pool_size=2),
    # keras.layers.Dropout(0.5),
    # # tf.keras.layers.Flatten(),
    # keras.layers.GlobalAveragePooling2D(),
    # dense_layer_1,
    # layers.GlobalAveragePooling2D(),  # Flatten the output before LSTM
    # layers.Reshape((1, -1)),  # Reshape to add a "time step" dimension for LSTM
    # layers.LSTM(64),  # LSTM layer with 32 units
    # layers.BatchNormalization(),
#     base_model,
#     tf.keras.layers.Dense(256, activation='relu'),
#     # tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     # tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#62% acc
# model = models.Sequential([
#     # tf.keras.layers.RandomFlip("horizontal"),
#     # tf.keras.layers.Normalization(),
#     # tf.keras.layers.Rescaling(scale=1./255),
#     tf.keras.layers.ActivityRegularization(input_shape=(256, 256, 3)),
#     base_model,
#
#     tf.keras.layers.BatchNormalization(),
#
#     tf.keras.layers.Conv2D(512, 5, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#     # tf.keras.layers.ActivityRegularization(l2=0.01),
#     tf.keras.layers.Conv2D(256, 5, padding='same', activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#
#     tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     layers.Dense(16, activation='relu'), #70 at 16 here and 0.2 dropout
#     tf.keras.layers.Dropout(0.2),
#     layers.Dense(2, activation='softmax')
# ])

model = models.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    # tf.keras.layers.Normalization(),
    tf.keras.layers.ActivityRegularization(input_shape=(256, 256, 3)),
    base_model,

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, 3, padding='same', activation='silu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'), #70 at 16 here and 0.2 dropout
    tf.keras.layers.Dropout(0.2),
    layers.Dense(2, activation='softmax')
])

# model = models.Sequential([
#     # # tf.keras.layers.preprocessing.RandomFlip("horizontal"),
#     # tf.keras.layers.LayerNormalization(),
#     # tf.keras.layers.Rescaling(scale=1./127),
#     # tf.keras.layers.ActivityRegularization(input_shape=(256, 256, 3)),
#     base_model,
#     tf.keras.layers.Conv2D(512, 5, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#
#     tf.keras.layers.Conv2D(512, 5, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#
#     tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     layers.Dense(16, activation='relu'),
#     layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.25),
#     layers.Dense(2, activation='softmax')
# ])
model.build(input_shape=(None,256, 256, 3))

model.compile(
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              loss='mse',
              # optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
              # optimizer=tf.keras.optimizers.Adadelta()
              optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4) ##we was working on this
              # optimizer='sgd'
              # optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
              ,metrics=['accuracy'])
model.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint("resnet-50.h5.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early = EarlyStopping(monitor='val_accuracy',  patience=15, verbose=1,  min_delta=0, mode='auto', restore_best_weights=True)

# def lr_schedule(epoch):
#     return 0.001 * np.exp(-epoch / 10)
# lr_scheduler = LearningRateScheduler(lr_schedule)

# class_weights = {class_names.index('Depressed'): 1.31, class_names.index('Not'): 4.89}

history= model.fit(train_ds
          , epochs=100
          , verbose=1
          # , shuffle= True
          ,batch_size=20
          , steps_per_epoch= 20
          , validation_data= val_ds
          , validation_steps= 8
          , validation_batch_size=10
          , callbacks=[checkpoint, early, reduce_lr]
          # , class_weight=class_weights
                   )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('MSE')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()



# for x, y in test_ds:
#   # Concatenate batch predictions and labels
#   predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=1)])
#   labels = np.append(labels,y)
#   images = np.append(images, x)

images= np.array([])
predictions = np.array([])
labels =  np.array([])
preds =  np.array([])

for x, y in test_ds:
  # Concatenate batch predictions and labels
  predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=1)])
  # preds=np.append(preds, model.predict(x))
  labels = np.append(labels, y)
  images = np.append(images, x)

# # # Convert lists to numpy arrays
# images = np.array(images)
# labels = np.array(labels)

# Reshape X_train if needed (assuming images are 4D with shape [batch_size, height, width, channels])
images = images.reshape(-1, 256, 256,3)


print(images.shape)
# images, labels = next(iter(test_ds))
#
# predictions= np.argmax(model.predict(X_test_reshaped), axis= 1)
# Code that may raise TypeError due to NoneType objects
np.save("y_true_audio_model", labels)
np.save("y_predicted_audio_model", predictions)
print(predictions)
print(labels)
# print(np.asarray(preds))
# print(len(images))
# images = images.reshape((-1, 256, 256, 3))
np.save("x_test_audio_model", images)
score = model.evaluate(test_ds,batch_size=1, verbose=1)
print('Test accuracy:', score[1])
print('Test MSE Loss:', score[0])

