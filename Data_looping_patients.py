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
import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, LSTM, Bidirectional
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import itertools
import math
import time

global number_list_train, phq8_array_train, count_train, number_list_dev, phq8_array_dev, count_dev, samples_number
global  number_list_test, phq8_array_test, count_test

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
for i in range(0,len(mapping_data)):
    number = mapping_data.iloc[i][0]
    split = mapping_data.iloc[i][1]
    phq8_array = mapping_data.iloc[i][4]
    samples_number.append(number)
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

        df = pd.read_csv('G:/data/' + str(number) + '_P/'
                         + str(number) + '_TRANSCRIPT.csv', header=0)
        number_list_test.append(number)
        phq8_array_test.append(phq8_array)
        count_test += 1
    except:
        continue


print('Train IDs:'+str(number_list_train))
print('Train num:'+str(len(number_list_train)))
print('Dev IDs:'+str(number_list_dev))
print('Dev num:'+str(len(number_list_dev)))
print('Test IDs:'+str(number_list_test))
print('Test num:'+str(len(number_list_test)))