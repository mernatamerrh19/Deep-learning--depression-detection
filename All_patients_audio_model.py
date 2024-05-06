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
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Embedding
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, LSTM, Bidirectional
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import itertools
import math
import time
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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


data = open(r'G:/metadata_mapped.csv', 'r', encoding='utf-8-sig')
mapping_data = pd.read_csv('G:/metadata_mapped.csv', header=0)

# def categorise(phq8_array):
#     if phq8_array <= 2:
#         phq8_array = 0
#     elif phq8_array >= 3 and phq8_array <= 5:
#         phq8_array = 1
#     elif phq8_array >= 6 and phq8_array <= 8:
#         phq8_array = 2
#     else:
#         phq8_array = 3
#     return phq8_array

for i in range(0,len(mapping_data)):
    number = mapping_data.iloc[i][0]
    split = mapping_data.iloc[i][1]
    phq8_array = mapping_data.iloc[i][3]
    samples_number.append(number)
    # phq8_array= categorise(phq8_array)
    if 'training' in split:
        number_list_train.append(number)
        phq8_array_train.append(phq8_array)
        count_train +=1
    else:
        number_list_dev.append(number)
        phq8_array_dev.append(phq8_array)
        count_dev += 1


dev_data = pd.read_csv(r'G:/labels/test_split.csv', header=0)

for iter in range(0,len(dev_data)):
    try:
        number = dev_data.iloc[iter][0]
        phq8_array = dev_data.iloc[iter][2]
        # phq8_array = categorise(phq8_array)

        df = pd.read_csv('G:/data/' + str(number) + '_P/'
                         + str(number) + '_TRANSCRIPT.csv', header=0)
        number_list_test.append(number)
        phq8_array_test.append(phq8_array)
        count_test += 1
    except:
        continue

print(phq8_array_train)
print(phq8_array_test)
print(phq8_array_dev)

# def get_short_time_fourier_transform(soundwave):
#     return librosa.stft(soundwave, n_fft=128)
#
# def short_time_fourier_transform_amplitude_to_db(stft):
#     return librosa.amplitude_to_db(np.abs(stft))
#
# def soundwave_to_np_spectogram(soundwave):
#     step1 = get_short_time_fourier_transform(soundwave)
#     step2 = short_time_fourier_transform_amplitude_to_db(step1)
#     return step2
#
# def inspect_data(sound):
#     # a = get_short_time_fourier_transform(sound)
#     Xdb = soundwave_to_np_spectogram(sound)
#     return Xdb
#
# updated_phq8_array_train=[]
# i = 0
# j = 0
# for audio_train in number_list_train:
#     if audio_train > 695:
#         audio = AudioSegment.from_wav('G:/data/Clipped/' + str(audio_train) + '.wav')
#         # audio,sr = librosa.load('G:/data/Clipped/' + str(audio_train) + '.wav')
#         length = int(len(audio)/1000)
#         intial = 0
#         end=120000
#         print(phq8_array_train[i])
#         print(int(length))
#         print(int(length / 120))
#         for j in range(0,int(length/120)):
#             updated_phq8_array_train.append(phq8_array_train[i])
#             audio_slice = audio[intial: end]
#             intial= intial+120000
#             end = end+ 120000
#             samples = audio_slice.get_array_of_samples()
#             samples = np.array(samples).astype(np.float64)
#             window_size = 128
#             window = np.hanning(window_size)
#             stft = librosa.stft(samples, n_fft=window_size, hop_length=32, window=window)
#             out = 2 * np.abs(stft) / np.sum(window)
#             # For plotting headlessly
#             fig = plt.Figure(frameon=False)
#             canvas = FigureCanvas(fig)
#             ax = fig.add_subplot()
#             # ax.patch.set_visible(False)
#             # ax.axis('off')
#             p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax)
#             if phq8_array_train[i] == 0:
#                 fig.savefig('G:/data/DP-preprocessed-train - Copy/Not/' + str(audio_train) + '_' + str(j) + '.png',
#                             transparent=True, bbox_inches='tight')
#                 plt.close(fig)
#             elif phq8_array_train[i] == 1:
#                 fig.savefig('G:/data/DP-preprocessed-train - Copy/Depressed/' + str(audio_train) + '_' + str(j) + '.png',
#                             transparent=True, bbox_inches='tight')
#                 plt.close(fig)
#             # elif phq8_array_train[i] == 2:
#             #     fig.savefig('G:/data/DP-preprocessed-train/Moderate/' + str(audio_train) + '_' + str(j) + '.png', transparent=True, bbox_inches='tight')
#             # elif phq8_array_train[i] == 3:
#             #     fig.savefig('G:/data/DP-preprocessed-train/Severe/' + str(audio_train) + '_' + str(j) + '.png', transparent=True, bbox_inches='tight')
#         i=i+1

#
#
updated_phq8_array_dev=[]
i = 0
j = 0
# for audio_dev in number_list_dev:
#     if audio_dev > 657:
#         audio = AudioSegment.from_wav('G:/data/Clipped/' + str(audio_dev) + '.wav')
#         # audio,sr = librosa.load('G:/data/Clipped/' + str(audio_train) + '.wav')
#         length = int(len(audio) / 1000)
#         intial = 0
#         end = 120000
#         for j in range(0, int(length / 120)):
#             updated_phq8_array_dev.append(phq8_array_dev[i])
#             audio_slice = audio[intial: end]
#             intial = intial + 120000
#             end = end + 120000
#             samples = audio_slice.get_array_of_samples()
#             samples = np.array(samples).astype(np.float64)
#             window_size = 128
#             window = np.hanning(window_size)
#             stft = librosa.stft(samples, n_fft=window_size, hop_length=32, window=window)
#             out = 2 * np.abs(stft) / np.sum(window)
#             # For plotting headlessly
#             fig = plt.Figure(frameon=False)
#             canvas = FigureCanvas(fig)
#             ax = fig.add_subplot()
#             ax.patch.set_visible(False)
#             ax.axis('off')
#             p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax)
#             if phq8_array_dev[i] == 0:
#                 fig.savefig('G:/data/DP-preprocessed-dev - Copy/Not/' + str(audio_dev) + '_' + str(j) + '.png',
#                             transparent=True, bbox_inches='tight')
#                 plt.close(fig)
#             elif phq8_array_dev[i] == 1:
#                 fig.savefig('G:/data/DP-preprocessed-dev - Copy/Depressed/' + str(audio_dev) + '_' + str(j) + '.png',
#                             transparent=True, bbox_inches='tight')
#                 plt.close(fig)
#             # elif phq8_array_dev[i] == 2:
#             #     fig.savefig('G:/data/DP-preprocessed-dev/Moderate/' + str(audio_dev) + '_' + str(j) + '.png', transparent=True, bbox_inches='tight')
#             # elif phq8_array_dev[i] == 3:
#             #     fig.savefig('G:/data/DP-preprocessed-dev/Severe/' + str(audio_dev) + '_' + str(j) + '.png', transparent=True, bbox_inches='tight')
#     i = i+1


updated_phq8_array_test=[]
i = 0
j = 0
for audio_test in number_list_test:
    if audio_test > 666:
        audio = AudioSegment.from_wav('G:/data/Clipped/' + str(audio_test) + '.wav')
        # audio,sr = librosa.load('G:/data/Clipped/' + str(audio_train) + '.wav')
        length = int(len(audio) / 1000)
        intial = 0
        end = 120000
        # for j in range(0, int(length / 180)):
        updated_phq8_array_test.append(phq8_array_test[i])
        audio_slice = audio[intial: end]
        intial = intial + 120000
        end = end + 120000
        samples = audio_slice.get_array_of_samples()
        samples = np.array(samples).astype(np.float32)
        window_size = 128
        window = np.hanning(window_size)
        stft = librosa.stft(samples, n_fft=window_size, hop_length=32, window=window)
        out = 2 * np.abs(stft) / np.sum(window)
        # For plotting headlessly
        fig = plt.Figure(frameon=False)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot()
        ax.patch.set_visible(False)
        ax.axis('off')
        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax)
        # p = librosa.display.specshow(out, ax=ax)
        if phq8_array_test[i] == 0:
            fig.savefig('G:/data/DP-preprocessed-test - Copy/Not/' + str(audio_test) + '_' + str(j) + '.png',
                        transparent=True, bbox_inches='tight')
            plt.close(fig)
        elif phq8_array_test[i] == 1:
            fig.savefig('G:/data/DP-preprocessed-test - Copy/Depressed/' + str(audio_test) + '_' + str(j) + '.png',
                        transparent=True, bbox_inches='tight')
            plt.close(fig)
        # elif phq8_array_test[i] == 2:
        #     fig.savefig('G:/data/DP-preprocessed-test/Moderate/' + str(audio_test) + '_' + str(j) + '.png', transparent=True, bbox_inches='tight')
        # elif phq8_array_test[i] == 3:
        #     fig.savefig('G:/data/DP-preprocessed-test/Severe/' + str(audio_test) + '_' + str(j) + '.png', transparent=True, bbox_inches='tight')
    i= i+1
