# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:39:50 2018

@author: asamiko
"""

from __future__ import print_function
import matplotlib.pyplot as plot
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input,GRU
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from keras.metrics import categorical_accuracy


K.set_image_data_format('channels_first')


def split_in_seqs(data, subdivs):
    """
    Splits a long sequence matrix into sub-sequences.
        Eg: input: data = MxN  sub-sequence length (subdivs) = 2
            output = M/2 x 2 x N
        
    :param data: Array of one or two dimensions 
    :param subdivs: integer value representing a sub-sequence length
    :return: array of dimension = input array dimension + 1 
    """
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, data.shape[1]))
    return data


def split_multi_channels(data, num_channels):
    """
    Split features into multiple channels
        Eg: input: data = MxNxP  num_channels = 2
        output = M x 2 x N x P/2
    
    :param data: 3-D array
    :param num_channels: integer value representing the number of channels
    :return: array of dimension = input array dimension + 1 
    """
    in_shape = data.shape
    if len(in_shape) == 3:
        hop = in_shape[2] // num_channels
        tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
        for i in range(num_channels):
            tmp[:, i, :, :] = data[:, :, i*hop:(i+1)*hop]
        return tmp
    else:
        print("ERROR: The input should be a 3D matrix but it seems to have dimensions ", in_shape)
        exit()


def get_rnn_model(in_data, out_data):
    #TODO: implement your RNN model here
    
    mel_start = Input(shape=(in_data.shape[-2], in_data.shape[-1]))
    x = GRU(32,activation='tanh',dropout=0.25)(mel_start)                   
    out = Dense(out_data.shape[-1], activation='sigmoid')(x)                 
    

    # leave the following unchanged
    _model = Model(inputs=mel_start, outputs=out)
    _model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = [categorical_accuracy])
    _model.summary()
    return _model


def get_model(in_data, out_data):
    """
    Keras model definition
    
    :param in_data: input data to the network (training data)
    :param out_data: output data to the network (training labels)
    :return: _model: keras model configuration
    """
    mel_pool_size = [1]
    mel_nb_filt = 32
    dropout_rate = 0.1

    mel_start = Input(shape=(in_data.shape[-3], in_data.shape[-2], in_data.shape[-1]))
    mel_x = mel_start
    for i, convCnt in enumerate(mel_pool_size):
        mel_x = Conv2D(filters=mel_nb_filt, kernel_size=(3, 3), padding='same')(mel_x)
        mel_x = BatchNormalization(axis=1)(mel_x)
        mel_x = Activation('relu')(mel_x)
        mel_x = MaxPooling2D(pool_size=(1, mel_pool_size[i]))(mel_x)
        mel_x = Dropout(dropout_rate)(mel_x)
    mel_x = Permute((2, 1, 3))(mel_x)
    mel_x = Reshape((in_data.shape[-2], (in_data.shape[-1] * mel_nb_filt) // np.prod(mel_pool_size)))(mel_x)

    mel_x = TimeDistributed(Dense(out_data.shape[-1]))(mel_x)
    out = Activation('sigmoid')(mel_x)

    _model = Model(inputs=mel_start, outputs=out)
    _model.compile(optimizer='Adam', loss='binary_crossentropy', metrics = [categorical_accuracy])
    _model.summary()
    return _model

# -------------------------------------------------------------------
#              Main script starts here
# -------------------------------------------------------------------

# TODO: Change the following three parameters as mentioned in the exercise

window_length = [4096, 2048, 1024, 512]
nb_mel_bands = 32
nb_frames = [320, 160, 80, 40]
Accuracy = np.zeros(16);
BestAccuracy = np.zeros(16);
CM = np.repeat([np.zeros((4,4))],16,axis = 0)
Best_ConfusionMatrix = np.repeat([np.zeros((4,4))],16,axis = 0)

k = 0;
for ind in range(0, 4):
    for j in range(0, 4):
        input_feat_name = 'four_genres_{}_{}_{}.npz'.format(nb_frames[ind], nb_mel_bands, window_length[j])
        print('input_feat_name: {}'.format(input_feat_name))
        # Load normalized features and pre-process them - splitting into sequence and multi-channels
        dmp = np.load(input_feat_name)
         
        # TODO: Change the data pre-processing based on the GRU requirements. Check the definition in the website. See what should be the input and output format
        train_data, train_labels, test_data, test_labels = \
            split_in_seqs(dmp['arr_0'], nb_frames[ind]), \
            split_in_seqs(dmp['arr_1'], nb_frames[ind]), \
            split_in_seqs(dmp['arr_2'], nb_frames[ind]), \
            split_in_seqs(dmp['arr_3'], nb_frames[ind])

        train_labels=train_labels[:,0,:]
        test_labels=test_labels[:,0,:]
        # Load the CNN model
        #model = get_model(train_data, train_labels)
        model = get_rnn_model(train_data, train_labels) #TODO: complete this function and then uncomment this line and use it
        
        nb_epoch = 300      # Maximum number of epochs for training
        batch_size =32      # Batch size
        
        patience = int(0.25 * nb_epoch)     # We stop training if the accuracy does not improve for 'patience' number of epochs
        patience_cnt = 0    # Variable to keep track of the patience
        
        best_accuracy = -999    # Variable to save the best accuracy of the model
        best_epoch = -1     # Variable to save the best epoch of the model
        train_loss = [0] * nb_epoch  # Variable to save the training loss of the model per epoch
        test_accuracy = [0] * nb_epoch  # Variable to save the training accuracy of the model per epoch
        
        # Training begins
        for i in range(nb_epoch):
            print('Epoch : {} '.format(i), end='')
        
            # Fit model for one epoch
            hist = model.fit(
                train_data,
                train_labels,
                batch_size=batch_size,
                epochs=1
            )
            # save the training loss for the epoch
            train_loss[i] = hist.history.get('loss')[-1]
        
            # Use the trained model on test data
            pred = model.predict(test_data, batch_size=batch_size)
        
            # Calculate the accuracy on the test data
            y_test_non_category = [ np.argmax(t) for t in test_labels ]
            y_predict_non_category = [ np.argmax(t) for t in pred ]
            test_accuracy[i] = metrics.accuracy_score(y_test_non_category, y_predict_non_category)
            patience_cnt = patience_cnt + 1
            # Calculate the confusion matrix
            ConfusionMatrix = confusion_matrix(y_test_non_category, y_predict_non_category)
            # Check if the test_accuracy for the epoch is better than the best_accuracy
            if test_accuracy[i] > best_accuracy:
                # Save the best accuracy and its respective epoch
                best_accuracy = test_accuracy[i]
                best_epoch = i
                patience_cnt = 0
                best_ConfusionMatrix = ConfusionMatrix
            print('Test accuracy: {}, best accuracy: {}, best epoch: {}, confusion matrix: {}'.format(test_accuracy[i], best_accuracy, best_epoch, ConfusionMatrix))
        
            # Early stopping, if the test_accuracy does not change for 'patience' number of epochs then we quit training
            if patience_cnt > patience:
                break
        
        print('The best_epoch: {} with best accuracy: {}'.format(best_epoch, best_accuracy))
        print('input_feat_name: {}'.format(input_feat_name))
        Accuracy[k] = test_accuracy[i]
        BestAccuracy[k] = best_accuracy
        CM[k] = ConfusionMatrix
        Best_ConfusionMatrix[k] = best_ConfusionMatrix        
        k = k + 1;
        
k = np.linspace(1,16,16);
plot.plot(k, Accuracy);
plot.plot(k, BestAccuracy);
print('FinalConfusionMatrices: {}'.format(CM))
print('ConfusionMatrices: {} with best accuracy'.format(Best_ConfusionMatrix))
