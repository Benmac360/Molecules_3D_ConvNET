# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:50:32 2020

@author: Benjamin
"""

import numpy as np  
import matplotlib  
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv3D, Dense, MaxPooling3D, Flatten, Input, GlobalMaxPooling3D,AveragePooling3D, GlobalAveragePooling3D
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
import tensorflow as tf
import keras

csv_logger = CSVLogger('log.csv', append=True, separator=';')

from keras.models import model_from_json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


plt.ioff()              # Turn of plotting interactive mode so can run on cluster

#%%


internal_energy = np.loadtxt('internal_energies_per_atom_medium_atoms.txt') # Load internarl energy
#internal_energy =internal_energy[:60000]
potentials_raw = np.load('medium_atoms_potential.npy')    
#potentials_raw =potentials_raw[:60000]


l = len(potentials_raw[0,:,0,0])                # Load potentials
potentials = potentials_raw.reshape(len(potentials_raw[:,1,1,1]),l,l,l,1)           # Reshape for the CNN - RESHAPING IS CORRECT
del(potentials_raw)                                         # Delete the raw potential database to preserve RAM

    
no_of_samples = len(potentials[:,1,1,1])                    # Count the number of samples
length = len(potentials[1,:,1,1,0])                         # Get the length of the grid
input_shape = (length,length,length,1)                      # Define input shape

no_of_epochs = int(2000)                                       # Set the number of trianing epochs

 
optimizer = keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.02, patience=15, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

#%%


###### Construct Model ######

cnn = Sequential()
#Reducing layer
cnn.add(Conv3D(8, 4, strides = 2, padding ='valid', activation = 'elu',data_format = 'channels_last', input_shape = input_shape))
cnn.summary()

cnn.add(AveragePooling3D(2))
cnn.summary()
#Convolution
cnn.add(Conv3D(10, 3, strides = 1 , padding ='valid', activation = 'elu'))
cnn.summary()
cnn.add(AveragePooling3D(2))
cnn.summary()

cnn.add(Conv3D(30, 2, strides = 1 , padding ='valid', activation = 'elu'))
cnn.summary()
cnn.add(AveragePooling3D(2))
cnn.summary()

cnn.add(Flatten())
# Into a dense layer
cnn.add(Dense(units=20 , activation = 'elu'))
#cnn.add(Dense(units=10 , activation = 'elu'))
#cnn.add(Dense(units=10 , activation = 'elu'))

cnn.add(Dense(units=1))
cnn.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mae'])
cnn.summary()

#%%
###### Train the model and print the R2 score for the test and training ######

X_train, X_test, y_train, y_test = train_test_split(potentials, internal_energy, train_size = 0.9)

X_test_in = X_test[:int(len(X_test)*9/10)]
y_test_in = y_test[:int(len(y_test)*9/10)]

X_val_in = X_test[int(len(X_test)*9/10):]
y_val_in = y_test[int(len(y_test)*9/10):]

cnn.fit(X_train,y_train, epochs = no_of_epochs, batch_size = int(50), callbacks=[callback,csv_logger],validation_data=(X_val_in, y_val_in))

y_pred_test=cnn.predict(X_test_in)
msqe_test=mean_squared_error(y_test_in, y_pred_test)  
var_test=np.var(y_test,ddof=1) 
score_test=1-msqe_test/var_test


y_pred_train=cnn.predict(X_train) 
msqe_train=mean_squared_error(y_train, y_pred_train)  
var_train=np.var(y_train,ddof=1) 
score_train=1-msqe_train/var_train

#print(score_test)
#print(score_train)

file1 = open("Test_and_train_score.txt","a")
write_in = ["The test score is \n",str(score_test),"\nThe train score is \n",str(score_train)] 
file1.writelines(write_in)
file1.close() 


### Heat Map between correct energies and predicted energies ###
 
predict = cnn.predict(X_test[:int(no_of_samples/10)])
correct = y_test[:int(no_of_samples/10)]
predict.shape = (int(no_of_samples/10),)
correct.shape = (int(no_of_samples/10),)


plt.hist2d(correct, predict, (100, 100), cmap='hot')
plt.xlabel('True Energies') 
plt.ylabel('Predicted Energies')
plt.title('Heat Map Internal Energy')
plt.colorbar()
plt.savefig('Heat Map of Internal Energy.png')
plt.close()


# serialize model to JSON
cnn_json = cnn.to_json()
with open("cnn.json", "w") as json_file:
    json_file.write(cnn_json)
# serialize weights to HDF5
cnn.save_weights("3D_CNN_weights.h5")
cnn.save("3D_CNN_Model.h5")
