#
#   ML NN Template
#   Use Keras and TensorFlow
#
#   Created by Vasili Makouski
#
#   Copyleft
#

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend
import numpy
import numpy as np
import os
import h5py
import datetime
import time
import tensorflow as tf
import dtst

numpy.random.seed(2)


Epchs = 10 # The number of epochs is the number of complete passes through the training dataset
Btchsz = 2 #The batch size is a number of samples processed before the model is updated

start = time.time() # training start time

inptsD, otptsD = dtst.traindat() #training data

IWs = int(10) #input layer
OWs = int(5) #output layes
print("Size", IWs)
print('-----------------------------')
print("Start:")
print('-----------------------------')

model = Sequential()
model.add(Dense(IWs, input_dim=IWs, activation='relu'))  #input layer and type of activation function
model.add(Dense(15, activation='sigmoid')) #hiden layer and type of activation function
model.add(Dense(15, activation='sigmoid')) #hiden layer and type of activation function
model.add(Dense(OWs, activation='sigmoid')) #output layes and type of activation function

model.compile(loss="binary_crossentropy", optimizer="SGD", metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
chkpt = ModelCheckpoint('modelNN.json', 
                        monitor='val_loss', 
                        verbose=1, 
                        save_best_only=True,
                        mode='auto')
callbacks = [early_stop, chkpt]

model.fit(inptsD, otptsD, epochs = Epchs, batch_size=Btchsz) , callbacks==callbacks #model training

end = time.time() # training finish time
print('-----------------------------')
print("Stop:")
print('Execution time min: %0.0f' % ((end - start)/60))
print('Execution time h: %0.0f' % (((end - start)/60)/60))
print('-----------------------------')
print('check:') #Print model fit result

# summary model trainong results
scores = model.evaluate(inptsD, otptsD)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print('-----------------------------')
print('Save our model to file')
model.save('our_template_model') # Save our model to file

print('-----------------------------')
print('NN test result:')
dd = np.array([])
dd = np.genfromtxt('nn_test_data_final.csv',delimiter=',', dtype=int)
DD_t = np.array([dd])
print(DD_t) #print test data
OT_test = model.predict(DD_t, batch_size=1)
np.savetxt("nn_test_result.csv", OT_test, delimiter=",", fmt = '%.2f')                                                                  
print(OT_test) #print result
print('-----------------------------')


