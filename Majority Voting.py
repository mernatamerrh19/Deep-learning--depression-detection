import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#x_test_visual_model
# y_predicted_visual= np.load("y_predicted_visual_model.npy")
y_true_visual= np.load('y_true_visual_model.npy')
print("Visual model\n True:")
print(y_true_visual, "\n")

y_predicted_visual= np.load("y_predicted_visual_model.npy")
print(y_predicted_visual, "\n")
# print("Predicted:")
# print(y_predicted_visual, "\n")
# print(len(y_predicted_visual), "\n")


#x_test_textual_model
y_true_textual=np.load("y_true_textual_model.npy")
print("textual model")
print(y_true_textual, "\n")

y_predicted_textual=np.load("y_predicted_textual_model.npy")
print(y_predicted_textual, "\n")

#x_test_audio_model
y_true_audio=np.load("y_true_audio_model.npy")
print("audio model")
print(y_true_audio, "\n")

y_predicted_audio=np.load("y_predicted_audio_model.npy")
print(y_predicted_audio, "\n")

majority_predicted=[]
for x,y,j in zip(y_predicted_audio,y_predicted_textual,y_predicted_visual):
    majority=x+y+j
    if majority ==0 or majority==1:
        majority_predicted.append(0)
    else:
        majority_predicted.append(1)

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import f1_score
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
matplotlib.use('TkAgg')

#Textual Model Results
f1_1=f1_score(y_true_textual, y_predicted_textual, average='weighted')
acc1=balanced_accuracy_score(y_true_textual, y_predicted_textual)
mse1= mean_squared_error(y_true_textual, y_predicted_textual)
rmse1=np.sqrt(mse1)
print('Textual Test F1:', f1_1)
print('Textual Test Accuracy:', acc1)
print('Textual Test MSE:', mse1)
print('Textual Test RMSE:', rmse1)

#Visual Model Results
f1_2=f1_score(y_true_visual, y_predicted_visual, average='weighted')
acc2=balanced_accuracy_score(y_true_visual, y_predicted_visual)
mse2= mean_squared_error(y_true_visual, y_predicted_visual)
rmse2=np.sqrt(mse2)
print('\nVisual Test F1:', f1_2)
print('Visual Test Accuracy:', acc2)
print('Visual Test MSE:', mse2)
print('Visual Test RMSE:', rmse2)

#Audio Model Results
f1_3=f1_score(y_true_audio, y_predicted_audio, average='weighted')
acc3=balanced_accuracy_score(y_true_audio, y_predicted_audio)
mse3= mean_squared_error(y_true_audio, y_predicted_audio)
rmse3=np.sqrt(mse3)
print('\nAudio Test F1:', f1_3)
print('Audio Test Accuracy:', acc3)
print('Audio Test MSE:', mse3)
print('Audio Test RMSE:', rmse3)

#Majority Voting Results
f1_4=f1_score(y_true_textual, majority_predicted, average='weighted')
acc4=balanced_accuracy_score(y_true_textual, majority_predicted)
mse4= mean_squared_error(y_true_textual, majority_predicted)
rmse4=np.sqrt(mse4)
print('\nMajority Test F1:', f1_4)
print('Majority Test Accuracy:', acc4)
print('Majority Test MSE:', mse4)
print('Majority Test RMSE:', rmse4)

# corr = np.corrcoef(y_true_textual, majority_predicted, method="pearson")
# print(corr)

##Plotting Confusion Matrices

#Visual
cm1 = confusion_matrix(y_true_visual, y_predicted_visual)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp1.plot()
plt.show()
#Textual
cm2 = confusion_matrix(y_true_textual, y_predicted_textual)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()
plt.show()
#Audio
cm3 = confusion_matrix(y_true_audio, y_predicted_audio)
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3)
disp3.plot()
plt.show()
#Majority
cm4 = confusion_matrix(y_true_textual, majority_predicted)
disp4 = ConfusionMatrixDisplay(confusion_matrix=cm4)
disp4.plot()
plt.show()

# acc3=tf.keras.metrics.Accuracy()
# acc3.update_state(y_true_audio, y_predicted_audio)
# result3 = acc3.result()
# print('Audio Test Accuracy:', result3.numpy())
#
# acc4=tf.keras.metrics.Accuracy()
# acc4.update_state(y_true_textual, majority_predicted)
# result4 = acc4.result()
# print('Test Accuracy:', result4.numpy())


# f1_3=tf.keras.metrics.F1Score(threshold=0.5)
# f1_3.update_state(y_true_audio, y_predicted_audio)
# result_f1_3 = f1_3.result()
# print('Audio Test F1:', result_f1_3.numpy())
#
# f1_4=tf.keras.metrics.F1Score(threshold=0.5)
# f1_4.update_state(y_true_textual, majority_predicted)
# result_f1_4 = f1_4.result()
# print('Test F1:', result_f1_4.numpy())


# corr = np.corrcoef(y_true, y_pred, method="pearson")
# print(corr)