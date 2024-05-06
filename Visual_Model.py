import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras.optimizers
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import models, layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.regularizers import l2
import tensorflow as tf
from keras.callbacks import ModelCheckpoint


tf.random.set_seed(41)
tf.keras.utils.set_random_seed(49)

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

scaler_pose = StandardScaler()
scaler_gaze = StandardScaler()
scaler_au = StandardScaler()

mapping_data = pd.read_csv('G:/metadata_mapped.csv', header=0)

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

test_data = pd.read_csv(r'G:/labels/test_split.csv', header=0)

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
print(phq8_array_test)
print(count_test)

combined_lists = samples_number + number_list_test
y = phq8_array_train + phq8_array_dev + phq8_array_test
num_zeros = y.count(0)
print(y)
print(combined_lists)

print('Train IDs:'+str(number_list_train))
print('Train num:'+str(len(number_list_train)))
print('Dev IDs:'+str(number_list_dev))
print('Dev num:'+str(len(number_list_dev)))
print('Test IDs:'+str(number_list_test))
print('Test num:'+str(len(number_list_test)))
# number_list_test.remove(659)
# number_list_test.remove(691)
# phq8_array_test.remove(phq8_array_test[31])
# phq8_array_test.remove(phq8_array_test[43])
print(len(phq8_array_test))

# # Preprocessing training, dev, and test files separately
# data_path = 'G:/data/Visual Features/'
# output_folder_1 = 'G:/data/Preprocessed_Visual_Features/Preprocessed training files'
#
#
# for number in number_list_train:
#     # Reading each file as a dataframe
#     file_path = os.path.join(data_path, f'{number}_OpenFace2.1.0_Pose_gaze_AUs.csv')
#     openface_data = pd.read_csv(file_path, header=0)
#
#     # Averaging each 30 rows to have one feature row representing each second in the video
#     openface_data_timestamp = pd.to_datetime(openface_data['timestamp'])
#     averaged_data = openface_data.groupby([openface_data_timestamp]).mean()
#
#     # Extracting the features from the averaged data
#     pose_features = averaged_data[['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']]
#     gaze_features = averaged_data[
#         ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']]
#     au_features = averaged_data[
#         ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
#          'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']]
#
#     # Normalizing the features
#     pose_features_normalized = scaler_pose.fit_transform(pose_features)
#     gaze_features_normalized = scaler_gaze.fit_transform(gaze_features)
#     au_features_normalized = scaler_au.fit_transform(au_features)
#
#     # Creating a new DataFrame with normalized features
#     processed_data = pd.DataFrame({
#         'pose_Tx': pose_features_normalized[:, 0],
#         'pose_Ty': pose_features_normalized[:, 1],
#         'pose_Tz': pose_features_normalized[:, 2],
#         'pose_Rx': pose_features_normalized[:, 3],
#         'pose_Ry': pose_features_normalized[:, 4],
#         'pose_Rz': pose_features_normalized[:, 5],
#         'gaze_0_x': gaze_features_normalized[:, 0],
#         'gaze_0_y': gaze_features_normalized[:, 1],
#         'gaze_0_z': gaze_features_normalized[:, 2],
#         'gaze_1_x': gaze_features_normalized[:, 3],
#         'gaze_1_y': gaze_features_normalized[:, 4],
#         'gaze_1_z': gaze_features_normalized[:, 5],
#         'gaze_angle_x': gaze_features_normalized[:, 6],
#         'gaze_angle_y': gaze_features_normalized[:, 7],
#         'AU01_r': au_features_normalized[:, 0],
#         'AU02_r': au_features_normalized[:, 1],
#         'AU04_r': au_features_normalized[:, 2],
#         'AU05_r': au_features_normalized[:, 3],
#         'AU06_r': au_features_normalized[:, 4],
#         'AU07_r': au_features_normalized[:, 5],
#         'AU09_r': au_features_normalized[:, 6],
#         'AU10_r': au_features_normalized[:, 7],
#         'AU12_r': au_features_normalized[:, 8],
#         'AU14_r': au_features_normalized[:, 9],
#         'AU15_r': au_features_normalized[:, 10],
#         'AU17_r': au_features_normalized[:, 11],
#         'AU20_r': au_features_normalized[:, 12],
#         'AU23_r': au_features_normalized[:, 13],
#         'AU25_r': au_features_normalized[:, 14],
#         'AU26_r': au_features_normalized[:, 15]
#     })
#
#     # Saving the processed data to a new CSV file in the output folder
#     output_file_path = os.path.join(output_folder_1, f'{number}_Preprocessed_Visual_Features.csv')
#     processed_data.to_csv(output_file_path, index=False)
#
# output_folder_2 = 'G:/data/Preprocessed_Visual_Features/Preprocessed dev files'
#
# for number in number_list_dev:
#     # Reading each file as a dataframe
#     file_path = os.path.join(data_path, f'{number}_OpenFace2.1.0_'
#                                         f'Pose_gaze_AUs.csv')
#     openface_data = pd.read_csv(file_path, header=0)
#
#     # Averaging each 30 rows to have one feature row representing each second in the video
#     openface_data_timestamp = pd.to_datetime(openface_data['timestamp'])
#     averaged_data = openface_data.groupby([openface_data_timestamp]).mean()
#
#     # Extracting the features from the averaged data
#     pose_features = averaged_data[['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']]
#     gaze_features = averaged_data[
#         ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']]
#     au_features = averaged_data[
#         ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
#          'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']]
#
#     # Normalizing the features
#     pose_features_normalized = scaler_pose.transform(pose_features)
#     gaze_features_normalized = scaler_gaze.transform(gaze_features)
#     au_features_normalized = scaler_au.transform(au_features)
#
#     # Creating a new DataFrame with normalized features
#     processed_data = pd.DataFrame({
#         'pose_Tx': pose_features_normalized[:, 0],
#         'pose_Ty': pose_features_normalized[:, 1],
#         'pose_Tz': pose_features_normalized[:, 2],
#         'pose_Rx': pose_features_normalized[:, 3],
#         'pose_Ry': pose_features_normalized[:, 4],
#         'pose_Rz': pose_features_normalized[:, 5],
#         'gaze_0_x': gaze_features_normalized[:, 0],
#         'gaze_0_y': gaze_features_normalized[:, 1],
#         'gaze_0_z': gaze_features_normalized[:, 2],
#         'gaze_1_x': gaze_features_normalized[:, 3],
#         'gaze_1_y': gaze_features_normalized[:, 4],
#         'gaze_1_z': gaze_features_normalized[:, 5],
#         'gaze_angle_x': gaze_features_normalized[:, 6],
#         'gaze_angle_y': gaze_features_normalized[:, 7],
#         'AU01_r': au_features_normalized[:, 0],
#         'AU02_r': au_features_normalized[:, 1],
#         'AU04_r': au_features_normalized[:, 2],
#         'AU05_r': au_features_normalized[:, 3],
#         'AU06_r': au_features_normalized[:, 4],
#         'AU07_r': au_features_normalized[:, 5],
#         'AU09_r': au_features_normalized[:, 6],
#         'AU10_r': au_features_normalized[:, 7],
#         'AU12_r': au_features_normalized[:, 8],
#         'AU14_r': au_features_normalized[:, 9],
#         'AU15_r': au_features_normalized[:, 10],
#         'AU17_r': au_features_normalized[:, 11],
#         'AU20_r': au_features_normalized[:, 12],
#         'AU23_r': au_features_normalized[:, 13],
#         'AU25_r': au_features_normalized[:, 14],
#         'AU26_r': au_features_normalized[:, 15]
#     })
#
#     # Saving the processed data to a new CSV file in the output folder
#     output_file_path = os.path.join(output_folder_2, f'{number}_Preprocessed_Visual_Features.csv')
#     processed_data.to_csv(output_file_path, index=False)
#
# output_folder_3 = 'G:/data/Preprocessed_Visual_Features/Preprocessed test files'
#
# for number in number_list_test:
#     # Reading each file as a dataframe
#     file_path = os.path.join(data_path, f'{number}_OpenFace2.1.0_Pose_gaze_AUs.csv')
#     openface_data = pd.read_csv(file_path, header=0)
#
#     # Averaging each 30 rows to have one feature row representing each second in the video
#     openface_data_timestamp = pd.to_datetime(openface_data['timestamp'])
#     averaged_data = openface_data.groupby([openface_data_timestamp]).mean()
#
#     # Extracting the features from the averaged data
#     pose_features = averaged_data[['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']]
#     gaze_features = averaged_data[['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
#                                    'gaze_angle_x', 'gaze_angle_y']]
#     au_features = averaged_data[
#         ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r',
#          'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']]
#
#     # Normalizing the features
#     pose_features_normalized = scaler_pose.transform(pose_features)
#     gaze_features_normalized = scaler_gaze.transform(gaze_features)
#     au_features_normalized = scaler_au.transform(au_features)
#
#     # Creating a new DataFrame with normalized features
#     processed_data = pd.DataFrame({
#         'pose_Tx': pose_features_normalized[:, 0],
#         'pose_Ty': pose_features_normalized[:, 1],
#         'pose_Tz': pose_features_normalized[:, 2],
#         'pose_Rx': pose_features_normalized[:, 3],
#         'pose_Ry': pose_features_normalized[:, 4],
#         'pose_Rz': pose_features_normalized[:, 5],
#         'gaze_0_x': gaze_features_normalized[:, 0],
#         'gaze_0_y': gaze_features_normalized[:, 1],
#         'gaze_0_z': gaze_features_normalized[:, 2],
#         'gaze_1_x': gaze_features_normalized[:, 3],
#         'gaze_1_y': gaze_features_normalized[:, 4],
#         'gaze_1_z': gaze_features_normalized[:, 5],
#         'gaze_angle_x': gaze_features_normalized[:, 6],
#         'gaze_angle_y': gaze_features_normalized[:, 7],
#         'AU01_r': au_features_normalized[:, 0],
#         'AU02_r': au_features_normalized[:, 1],
#         'AU04_r': au_features_normalized[:, 2],
#         'AU05_r': au_features_normalized[:, 3],
#         'AU06_r': au_features_normalized[:, 4],
#         'AU07_r': au_features_normalized[:, 5],
#         'AU09_r': au_features_normalized[:, 6],
#         'AU10_r': au_features_normalized[:, 7],
#         'AU12_r': au_features_normalized[:, 8],
#         'AU14_r': au_features_normalized[:, 9],
#         'AU15_r': au_features_normalized[:, 10],
#         'AU17_r': au_features_normalized[:, 11],
#         'AU20_r': au_features_normalized[:, 12],
#         'AU23_r': au_features_normalized[:, 13],
#         'AU25_r': au_features_normalized[:, 14],
#         'AU26_r': au_features_normalized[:, 15],
#     })
#
#     # Saving the processed data to a new CSV file in the output folder
#     output_file_path = os.path.join(output_folder_3, f'{number}_Preprocessed_Visual_Features.csv')
#     processed_data.to_csv(output_file_path, index=False)
#
# #### Preprocessing Finished
#
# # Specify the common structure of the file names
#
# file_name_template = 'G:/data/Preprocessed_Visual_Features/Preprocessed training files/{}_Preprocessed_Visual_Features.csv'
#
# # Generate a list of file paths using a loop
# file_paths_train = [file_name_template.format(patient_number) for patient_number in number_list_train]
#
# # Specify the common structure of the file names
# file_name_template = 'G:/data/Preprocessed_Visual_Features/Preprocessed dev files/{}_Preprocessed_Visual_Features.csv'
#
# # Generate a list of file paths using a loop
# file_paths_dev = [file_name_template.format(patient_number) for patient_number in number_list_dev]
#
# # Specify the common structure of the file names
# file_name_template = 'G:/data/Preprocessed_Visual_Features/Preprocessed test files/{}_Preprocessed_Visual_Features.csv'
#
# # Generate a list of file paths using a loop
# file_paths_test = [file_name_template.format(patient_number) for patient_number in number_list_test]
#
#
# def load_and_concatenate(file_paths, labels):
#     dfs = []
#     concatenated_df=[]
#     for file_path, label in zip(file_paths, labels):
#         df = pd.read_csv(file_path)
#         # Assuming 'label_column' for your label
#         df['label_column'] = label
#         dfs.append(df)
#         # concatenated_df.append(df)
#     # concatenated_df = pd.concat(dfs, ignore_index=True)
#     return dfs
#
# # Load and concatenate train data
# train_data = load_and_concatenate(file_paths_train, phq8_array_train)
#
# # Load and concatenate dev data
# dev_data = load_and_concatenate(file_paths_dev, phq8_array_dev)
#
# # Load and concatenate test data
# test_data = load_and_concatenate(file_paths_test, phq8_array_test)
#
# print(len(train_data))
# print(len(dev_data))
# print(len(test_data))
#
#
# def extract_features_and_labels(data):
#     # Extract features (adjust columns based on your data)
#     X = data[['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
#               'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y',
#               'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
#               'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']]
#     # Extract labels
#     y = data['label_column']
#     return X, y
#
# # Extract features and labels for train, dev, and test sets
# X_train, y_train = extract_features_and_labels(train_data)
# X_dev, y_dev = extract_features_and_labels(dev_data)
# X_test, y_test = extract_features_and_labels(test_data)
#
# print(X_train)
# print(X_train.shape)
#
def load_and_concatenate(file_paths, labels):
    dfs = []
    for file_path, label in zip(file_paths, labels):
        df = pd.read_csv(file_path)
        # Assuming 'label_column' for your label
        df['label_column'] = label
        dfs.append(df)
    return dfs

# Generate a list of file paths for the train set using a loop
file_name_template = 'G:/data/Preprocessed_Visual_Features/Preprocessed training files/{}_Preprocessed_Visual_Features.csv'
file_paths_train = [file_name_template.format(patient_number) for patient_number in number_list_train]

# Generate a list of file paths for the dev set using a loop
file_name_template = 'G:/data/Preprocessed_Visual_Features/Preprocessed dev files/{}_Preprocessed_Visual_Features.csv'
file_paths_dev = [file_name_template.format(patient_number) for patient_number in number_list_dev]

# Generate a list of file paths for the test set using a loop
file_name_template = 'G:/data/Preprocessed_Visual_Features/Preprocessed test files/{}_Preprocessed_Visual_Features.csv'
file_paths_test = [file_name_template.format(patient_number) for patient_number in number_list_test]

# Load and concatenate train data
train_data = load_and_concatenate(file_paths_train, phq8_array_train)

# Load and concatenate dev data
dev_data = load_and_concatenate(file_paths_dev, phq8_array_dev)

# Load and concatenate test data
test_data = load_and_concatenate(file_paths_test, phq8_array_test)

print(len(train_data))
print(len(dev_data))
print(len(test_data))

min_dim_train= min(train_data[i].shape[0] for i in range(0, len(train_data)))
min_dim_dev= min(dev_data[i].shape[0] for i in range(0, len(dev_data)))
min_dim_test= min(test_data[i].shape[0] for i in range(0, len(test_data)))
min_dimension=min(min_dim_train, min_dim_dev, min_dim_test)
print(min_dim_train, min_dim_dev, min_dim_test)
# Function to extract features and labels from data
def extract_features_and_labels(data, min_dim):
    X = []  # List to store features
    y = []  # List to store labels
    for df in data:
        # Extract features (adjust columns based on your data)
        X_df = df[['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
                   'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                   'gaze_angle_x', 'gaze_angle_y', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r',
                   'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
                   'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']]
        X_df = X_df.iloc[:min_dim, :]  # Extract the first min_dim rows
        X.append(X_df.to_numpy())  # Append the numpy array to X
        # Extract labels
        y_df = df['label_column']
        if 0 in y_df.values:
            y.append(0)
        else:
            y.append(1)
    return X, y

# Extract features and labels for train, dev, and test sets
X_train, y_train = extract_features_and_labels(train_data, min_dimension)
X_dev, y_dev = extract_features_and_labels(dev_data, min_dimension)
X_test, y_test = extract_features_and_labels(test_data, min_dimension)



X_train_reshaped = np.array(X_train)
X_dev_reshaped = np.array(X_dev)
X_test_reshaped = np.array(X_test)

# X_dev_reshaped = np.array([x.reshape((min_dim_dev, -1)) for x in X_dev])
# X_test_reshaped = np.array([x.reshape((min_dim_test, -1)) for x in X_test])

print(X_train_reshaped.shape)
print(X_dev_reshaped.shape) # Assuming you want to check the shape of the first patient's data
print(X_test_reshaped.shape)


# # Reshape X_train, X_dev, and X_test to the desired shapes
# X_train_reshaped = np.array([x.reshape(30, 163) for x in X_train])
# X_dev_reshaped = np.array([x.reshape(30, 56) for x in X_dev])
# X_test_reshaped = np.array([x.reshape(30, 54) for x in X_test])
#
# Print shapes to verify
# print(X_train_reshaped.shape)
# print(X_dev_reshaped.shape)
# print(X_test_reshaped.shape)
# print(y_train.shape())
# print(y_dev.shape())
# print(y_test.shape())
# Function to load and concatenate data for each patient separately


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


X_train_reshaped = X_train_reshaped.reshape((X_train_reshaped.shape[0], -1))
X_dev_reshaped = X_dev_reshaped.reshape((X_dev_reshaped.shape[0], -1))

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Assuming you have defined your classes as 'Depressed' and 'Not', and have imported RandomOverSampler and RandomUnderSampler

# Define the sampling strategy for oversampling and undersampling
oversampling_strategy = {1: 70}
undersampling_strategy = {0: 70}

# Initialize the RandomOverSampler and RandomUnderSampler with the specified sampling strategies
oversampler = RandomOverSampler(sampling_strategy=oversampling_strategy)
undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy)

# Perform oversampling on the validation data
X_train_reshaped, y_train = oversampler.fit_resample(X_train_reshaped, y_train)

# Perform undersampling on the resampled data
X_train_reshaped, y_train = undersampler.fit_resample(X_train_reshaped, y_train)

#For Validation Data
# Define the sampling strategy for oversampling and undersampling
oversampling_strategy = {1: 30}
undersampling_strategy = {0: 30}

# Initialize the RandomOverSampler and RandomUnderSampler with the specified sampling strategies
oversampler = RandomOverSampler(sampling_strategy=oversampling_strategy)
undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy)

# Perform oversampling on the validation data
X_dev_reshaped, y_dev = oversampler.fit_resample(X_dev_reshaped, y_dev)

# Perform undersampling on the resampled data
X_dev_reshaped, y_dev = undersampler.fit_resample(X_dev_reshaped, y_dev)


count_unique(y_train, y_dev, y_test)


y_train_one_hot = to_categorical(y_train,2)
y_dev_one_hot = to_categorical(y_dev,2)
y_test_one_hot = to_categorical(y_test,2)
# y_train_one_hot = np.array(y_train)
# y_dev_one_hot = np.array(y_dev)
# y_test_one_hot = np.array(y_test)


X_train_reshaped = X_train_reshaped.reshape((X_train_reshaped.shape[0], min_dimension, X_train_reshaped.shape[1]//min_dimension))
X_dev_reshaped = X_dev_reshaped.reshape((X_dev_reshaped.shape[0], min_dimension, X_dev_reshaped.shape[1]//min_dimension))

print("\n"+str(X_train_reshaped.shape))

# Adding early stopping

checkpoint = ModelCheckpoint("visual.keras.weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy',  patience=15, verbose=1,  min_delta=0, mode='auto', restore_best_weights=True)


# Learning rate scheduler
def lr_schedule(epoch):
    return 0.001 * np.exp(-epoch / 10)
lr_scheduler = LearningRateScheduler(lr_schedule)

num_classes = 2
batch_size = 64
epochs = 300

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
# optimizer = keras.optimizers.Adamax(learning_rate=0.001)

# model.add(layers.Conv1D(256, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))


model = models.Sequential()
model.add(layers.Conv1D(1024, kernel_size=5, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(BatchNormalization())
model.add(layers.MaxPooling1D(2))
model.add(layers.Dropout(0.6))
model.add(layers.Conv1D(512, kernel_size=3, activation='tanh'))
model.add(BatchNormalization())
model.add(layers.MaxPooling1D(2))
model.add(layers.Dropout(0.6))
model.add(layers.Conv1D(256, kernel_size=3, activation='tanh'))
model.add(BatchNormalization())
model.add(layers.MaxPooling1D(2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling1D(2))
model.add(layers.Dropout(0.5))
# model.add(BatchNormalization())
model.add(layers.Flatten())
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dropout(0.25))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(BatchNormalization())
model.add(layers.Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train_one_hot, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_dev_reshaped, y_dev_one_hot), callbacks=[early_stopping, lr_scheduler, checkpoint])

np.save('x_test_visual_model',X_test_reshaped)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_one_hot)
print(f'Test Accuracy: {test_accuracy}')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predictions= np.argmax(model.predict(X_test_reshaped), axis= 1)
y_test_one_hot= np.argmax(y_test_one_hot, axis= 1)
#
print(predictions)
print(len(predictions))
np.save("y_predicted_visual_model", np.asarray(predictions))
# print(y_test_one_hot)
# print(len(y_test_one_hot))
np.save('y_true_visual_model', y_test_one_hot)
