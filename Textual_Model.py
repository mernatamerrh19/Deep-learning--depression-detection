import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import keras
import keras.optimizers
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.models import Sequential
from keras.utils import to_categorical
import tensorflow as tf
from keras.regularizers import l2
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant

# from numpy.random import seed
# seed(1)
# tf.random.set_seed(42)
# tf.keras.utils.set_random_seed(63)

# path_meta= 'F:/GP/metadata_mapped.csv'
# path_test_split=r'F:/GP/labels/test_split.csv'
# output_folder_1 = 'F:\GP\Preprocessed_Textual_Data\Preprocessed_training_files'
# output_folder_2 = 'F:\GP\Preprocessed_Textual_Data\Preprocessed_dev_files'
# output_folder_3 = 'F:\GP\Preprocessed_Textual_Data\Preprocessed_test_files'
# glove_file_path = 'F:\GP\glove.twitter.27B.200d.txt'
path_meta='G:/metadata_mapped.csv'
path_test_split=r'G:/labels/test_split.csv'
output_folder_1="G:/data/txt-preprocessed-train"
output_folder_2="G:/data/txt-preprocessed-dev"
output_folder_3="G:/data/txt-preprocessed-test"
glove_file_path = 'G:\Glove\glove.twitter.27B.200d.txt'


number_list_train=[]
phq8_array_train=[]
count_train= 0

number_list_dev=[]
phq8_array_dev=[]
count_dev=0
samples_number=[]

number_list_test=[]
phq8_array_test=[]
count_test=0

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# vectorizer = TfidfVectorizer()
# N-gram Vectorization
ngram_range = (1, 2)  # This example considers unigrams and bigrams
vectorizer = CountVectorizer(ngram_range=ngram_range)

mapping_data = pd.read_csv(path_meta, header=0)

for i in range(0,len(mapping_data)):
    number = mapping_data.iloc[i][0]
    split = mapping_data.iloc[i][1]
    phq8_array = mapping_data.iloc[i][3]
    samples_number.append(number)
    if number not in number_list_train and number not in number_list_dev:
        if 'training' in split:
            number_list_train.append(number)
            phq8_array_train.append(phq8_array)
            count_train +=1
        else:
            number_list_dev.append(number)
            phq8_array_dev.append(phq8_array)
            count_dev += 1
    else:
        pass

test_data = pd.read_csv(path_test_split, header=0)

for iter in range(0,len(test_data)):
    try:
        number = test_data.iloc[iter][0]
        phq8_array = test_data.iloc[iter][2]

        if number not in number_list_test:
            if number != 659 and number != 691:
                number_list_test.append(number)
                phq8_array_test.append(phq8_array)
                count_test += 1
        else:
            pass
    except:
        continue



# combined_lists = samples_number + number_list_test
# y = phq8_array_train + phq8_array_dev + phq8_array_test
# num_zeros = y.count(0)
# print(y)
# print(num_zeros)
# print(len(combined_lists))

print('Train IDs:'+str(number_list_train))
print('Train num:'+str(len(number_list_train)))
print('Dev IDs:'+str(number_list_dev))
print('Dev num:'+str(len(number_list_dev)))
# number_list_test.remove(659)
# number_list_test.remove(691)
# phq8_array_test.remove(phq8_array_test[31])
# phq8_array_test.remove(phq8_array_test[43])
print('Test IDs:'+str(number_list_test))
print('Test num:'+str(len(number_list_test)))
# print(len(phq8_array_test))

print(phq8_array_test)
print(count_test)

def clean_text(text):
    text = re.sub(r'[^A-Za-z]', ' ', text)  # Keep only alphabetical characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

nltk.download('punkt')
nltk.download('wordnet')
X_vectorized_train = []

for number in number_list_train:
    try:
        df = pd.read_csv('G:/data/' + str(number) + '_P/' + str(number) + '_TRANSCRIPT.csv', header=0)
        df = df.dropna()
        df['cleaned_text'] = df['Text'].apply(clean_text)
        df['tokenized_text'] = df['cleaned_text'].apply(word_tokenize)
        df['lowercased_text'] = df['tokenized_text'].apply(lambda tokens: [token.lower() for token in tokens])
        df['filtered_text'] = df['lowercased_text'].apply(
            lambda tokens: [token for token in tokens if token not in stop_words])
        df['lemmatized_text'] = df['filtered_text'].apply(
            lambda tokens: [lemmatizer.lemmatize(token, pos="a") for token in tokens])
        X_vectorized = vectorizer.fit_transform(df['lemmatized_text'].apply(lambda tokens: ' '.join(tokens)))
        X_vectorized_train.append(X_vectorized)

        output_file_path = os.path.join(output_folder_1, f'{number}_Preprocessed_Transcript.csv')
        df.to_csv(output_file_path, index=False)

    except:
        continue
print(X_vectorized_train)

X_vectorized_dev = []

for number in number_list_dev:
    try:
        df = pd.read_csv('G:/data/' + str(number) + '_P/' + str(number) + '_TRANSCRIPT.csv', header=0)
        df = df.dropna()
        df['cleaned_text'] = df['Text'].apply(clean_text)
        df['tokenized_text'] = df['cleaned_text'].apply(word_tokenize)
        df['lowercased_text'] = df['tokenized_text'].apply(lambda tokens: [token.lower() for token in tokens])
        df['filtered_text'] = df['lowercased_text'].apply(
            lambda tokens: [token for token in tokens if token not in stop_words])
        df['lemmatized_text'] = df['filtered_text'].apply(
            lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
        X_vectorized = vectorizer.fit_transform(df['lemmatized_text'].apply(lambda tokens: ' '.join(tokens)))
        X_vectorized_dev.append(X_vectorized)

        output_file_path = os.path.join(output_folder_2, f'{number}_Preprocessed_Transcript.csv')
        df.to_csv(output_file_path, index=False)

    except:
        continue


X_vectorized_test = []

for number in number_list_test:
    try:
        df = pd.read_csv('G:/data/' + str(number) + '_P/' + str(number) + '_TRANSCRIPT.csv', header=0)
        df = df.dropna()
        df['cleaned_text'] = df['Text'].apply(clean_text)
        df['tokenized_text'] = df['cleaned_text'].apply(word_tokenize)
        df['lowercased_text'] = df['tokenized_text'].apply(lambda tokens: [token.lower() for token in tokens])
        df['filtered_text'] = df['lowercased_text'].apply(
            lambda tokens: [token for token in tokens if token not in stop_words])
        df['lemmatized_text'] = df['filtered_text'].apply(
            lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
        X_vectorized = vectorizer.fit_transform(df['lemmatized_text'].apply(lambda tokens: ' '.join(tokens)))
        X_vectorized_test.append(X_vectorized)

        output_file_path = os.path.join(output_folder_3, f'{number}_Preprocessed_Transcript.csv')
        df.to_csv(output_file_path, index=False)

    except:
        continue


all_text = []
max_len_train = 2573

for X_vectorized in X_vectorized_train:
    non_zero_elements = X_vectorized.nonzero()

    words = [vectorizer.get_feature_names_out()[idx] for idx in non_zero_elements[1] if idx < len(vectorizer.get_feature_names_out())]

    sentence = ' '.join(words)

    all_text.append(sentence)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)


vocab_size = len(tokenizer.word_index) + 1
# max_len = max(len(sentence.split()) for sentence in all_text)

print(f'Vocabulary Size: {vocab_size}')
# print(f'Maximum Sequence Length: {max_len}')

sequences = tokenizer.texts_to_sequences(all_text)

X_padded_train = pad_sequences(sequences, maxlen=max_len_train, padding='post')


all_text = []
max_len_dev = 2573

for X_vectorized in X_vectorized_dev:
    non_zero_elements = X_vectorized.nonzero()

    words = [vectorizer.get_feature_names_out()[idx] for idx in non_zero_elements[1] if idx < len(vectorizer.get_feature_names_out())]

    sentence = ' '.join(words)

    all_text.append(sentence)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)


vocab_size = len(tokenizer.word_index) + 1
# max_len = max(len(sentence.split()) for sentence in all_text)

print(f'Vocabulary Size: {vocab_size}')
# print(f'Maximum Sequence Length: {max_len}')

sequences = tokenizer.texts_to_sequences(all_text)

#Using the max_len_train not max_len_dev to make all text have the same length using padding
X_padded_dev = pad_sequences(sequences, maxlen=max_len_train, padding='post')

all_text = []
max_len_test = 2586

for X_vectorized in X_vectorized_test:
    non_zero_elements = X_vectorized.nonzero()

    words = [vectorizer.get_feature_names_out()[idx] for idx in non_zero_elements[1] if idx < len(vectorizer.get_feature_names_out())]

    sentence = ' '.join(words)

    all_text.append(sentence)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)


vocab_size = len(tokenizer.word_index) + 1
# max_len = max(len(sentence.split()) for sentence in all_text)

print(f'Vocabulary Size: {vocab_size}')
# print(f'Maximum Sequence Length: {max_len}')

sequences = tokenizer.texts_to_sequences(all_text)


#Using the max_len_train not max_len_test to make all text have the same length using padding
X_padded_test = pad_sequences(sequences, maxlen=max_len_train, padding='post')


embedding_dim = 200

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings = load_glove_embeddings(glove_file_path)


embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Initialize embedding layer with pre-trained embeddings
# embedding_layer = Embedding(
#     input_dim=vocab_size,
#     output_dim=embedding_dim,
#     weights=[embedding_matrix],  # Set pre-trained embeddings here
#     input_length=max_len_train,
#     trainable=False  # Set to True if you want to fine-tune embeddings
# )

embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    embeddings_initializer=Constant(embedding_matrix),  # Use Constant initializer for pre-trained embeddings
    trainable=False  # Set to True if you want to fine-tune embeddings
)

X_train = np.vstack(X_padded_train)
y_train = phq8_array_train

X_dev = np.vstack(X_padded_dev)
y_dev = phq8_array_dev

X_test = np.vstack(X_padded_test)
y_test = phq8_array_test

def count_unique(y_train, y_dev, y_test):
    unique_train, count_train = np.unique(y_train, return_counts=True)
    print("Training Data:")
    for cls, count in zip(unique_train, count_train):
        print(f"Class {cls}: {count} samples")

    # Count of each class in the development data
    unique_dev, count_dev = np.unique(y_dev, return_counts=True)
    print("\nDevelopment Data:")
    for cls, count in zip(unique_dev, count_dev):
        print(f"Class {cls}: {count} samples")

    # Count of each class in the test data
    unique_test, count_test = np.unique(y_test, return_counts=True)
    print("\nTest Data:")
    for cls, count in zip(unique_test, count_test):
        print(f"Class {cls}: {count} samples")

count_unique(y_train, y_dev, y_test)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Assuming you have defined your classes as 'Depressed' and 'Not', and have imported RandomOverSampler and RandomUnderSampler

# Define the sampling strategy for oversampling and undersampling
oversampling_strategy = {1: 90}
undersampling_strategy = {0: 90}

# Initialize the RandomOverSampler and RandomUnderSampler with the specified sampling strategies
oversampler = RandomOverSampler(sampling_strategy=oversampling_strategy)
undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy)

# Perform oversampling on the validation data
X_train, y_train = oversampler.fit_resample(X_train, y_train)

# Perform undersampling on the resampled data
X_train, y_train = undersampler.fit_resample(X_train, y_train)

#For Validation Data
# Define the sampling strategy for oversampling and undersampling
oversampling_strategy = {1: 35}
undersampling_strategy = {0: 35}

# Initialize the RandomOverSampler and RandomUnderSampler with the specified sampling strategies
oversampler = RandomOverSampler(sampling_strategy=oversampling_strategy)
undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy)

# Perform oversampling on the validation data
X_dev, y_dev = oversampler.fit_resample(X_dev, y_dev)

# Perform undersampling on the resampled data
X_dev, y_dev = undersampler.fit_resample(X_dev, y_dev)


count_unique(y_train, y_dev, y_test)


max_len_train = 2659
max_len_dev = 2573
max_len_test = 2586

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# embedding_dim = 50
num_classes = 2
batch_size = 64
epochs = 300

# Adding early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint("textual.weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', save_weights_only=True)


# Learning rate scheduler
def lr_schedule(epoch):
    return 0.001 * np.exp(-epoch / 10)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Trying different optimizers:
# optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9) #53
# optimizer = keras.optimizers.Adagrad(learning_rate=0.001)
# optimizer = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
# optimizer = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9)
# optimizer = keras.optimizers.Ftrl(learning_rate=0.01, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
optimizer = keras.optimizers.Adam(learning_rate=0.0001) #59
# optimizer = keras.optimizers.Adamax(learning_rate=0.001)

model = Sequential()

# model.add(BatchNormalization())

model.add(embedding_layer)

model.add(BatchNormalization())
# model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(3))
model.add(Dropout(0.4))

model.add(Conv1D(128, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(3))
model.add(Dropout(0.3))

model.add(Conv1D(64, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(3))
model.add(Dropout(0.3))

# model.add(Conv1D(64, 3, activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(3))
# model.add(Dropout(0.3))

model.add(GlobalAveragePooling1D())

# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
#
# # model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.summary()
model.compile(loss='mse', optimizer= optimizer, metrics=['accuracy'])

X_train_reshaped = X_train.reshape(X_train.shape[0], max_len_dev)
X_dev_reshaped = X_dev.reshape(X_dev.shape[0], max_len_dev)
X_test_reshaped = X_test.reshape(X_test.shape[0], max_len_dev)

X_train_reshaped = np.array(X_train_reshaped)
X_dev_reshaped = np.array(X_dev_reshaped)
X_test_reshaped = np.array(X_test_reshaped)
y_train = tf.keras.utils.to_categorical(y_train,2)
y_dev = tf.keras.utils.to_categorical(y_dev,2)
y_test=np.array(y_test)
print(X_test_reshaped.shape)
print(y_test.shape)
np.save('y_true_textual_model', y_test)
np.save('x_test_textual_model',X_test_reshaped)
y_test = tf.keras.utils.to_categorical(y_test,2)


history = model.fit(X_train_reshaped, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_dev_reshaped, y_dev), callbacks=[early_stopping,lr_scheduler, checkpoint]) #, lr_scheduler,


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# predictions = np.array([])
# for x in X_test_reshaped:
#     predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=1)])
# for x, y in test_ds:
#   # Concatenate batch predictions and labels
#   predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=1)])
#   # preds=np.append(preds, model.predict(x))
#   labels = np.append(labels, y)
#   images = np.append(images, x)

predictions= np.argmax(model.predict(X_test_reshaped), axis= 1)

print(predictions)
np.save("y_predicted_textual_model", np.asarray(predictions))

eval_result = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}')
