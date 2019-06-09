#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from __future__ import print_function
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, GRU, Activation
#from IPython import embed
#from keras import backend as K

__docformat__ = 'reStructuredText'
__all__ = [
    'get_model_1',
    'get_model_2',
    'get_model_3'
]


def get_model_1(input_shape, model_case):
    """Creates the FNN model for the Task 1.

    :param input_shape: The input shape to the model
    :type input_shape: (int, int)
    :param output_dimensions: The output dimensionality of the two layers
    :type output_dimensions: (int, int) | list[int]
    :param model_case: The case (e.g. source prediction)
    :type model_case: str
    :return: The model
    :rtype: keras.models.Sequential
    """
    model = Sequential()
    model.add(TimeDistributed(Dense(512),input_shape=input_shape))
    model.add(TimeDistributed(Dense(512)))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.add(Activation('relu'))
    model.compile(optimizer='Adam', loss='mse')
    
    # TODO: Implement a 2-3 layer FNN based auto-encoder
  
    # TODO: Use proper activation based on the range of the spectrogram magnitude

    _print_informative_message(
        model=model,
        model_case=model_case
    )

    return model


def get_model_2(input_shape, model_case):
    """Creates the FNN model for the Task 2.

    :param input_shape: The input shape to the model
    :type input_shape: (int, int)
    :param output_dimensions: The output dimensionality of the two layers
    :type output_dimensions: (int, int) | list[int]
    :param model_case: The case (e.g. source prediction)
    :type model_case: str
    :return: The model
    :rtype: keras.models.Sequential
    """
    model = Sequential()
    model.add(TimeDistributed(Dense(512),input_shape=input_shape))
    model.add(TimeDistributed(Dense(512)))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='Adam', loss='mse')

    # TODO: Implement a 2-3 layer FNN model
    # TODO: Use proper activation based on the range of the output mask magnitude

    _print_informative_message(
        model=model,
        model_case=model_case
    )

    return model


def get_model_3(input_shape, model_case):
    """Creates the RNN model for the Task 3.

    :param input_shape: The input shape to the model
    :type input_shape: (int, int)
    :param output_dimensions: The output dimensionality of the two layers
    :type output_dimensions: (int, int) | list[int]
    :param model_case: The case (e.g. source prediction)
    :type model_case: str
    :return: The model
    :rtype: keras.models.Sequential
    """
    model = Sequential()
    model.add(GRU(512, dropout=0.24, return_sequences=True,input_shape=input_shape))
    model.add(GRU(512, dropout=0.24, return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.add(Activation('relu'))
    model.compile(optimizer='Adam', loss='mse')
    # TODO: Implement a 2-4 layer GRU-RNN based model
    # TODO: Use FNN to reduce the dimension if necessary
    # TODO: Use proper activation based on the range of the spectrogram/mask magnitude

    _print_informative_message(
        model=model,
        model_case=model_case
    )

    return model


def _print_informative_message(model, model_case):
    """Prints the model case and summary of the model.

    :param model: The model.
    :type model: from keras.models.Sequential
    :param model_case: The model case
    :type model_case: str
    """
    print('Printing summary for the model that does {}'.format(
        model_case
    ))
    model.summary()

# EOF
# Using FNN predicting the magnitude spectrogram with DAE framework, 
# the mean SDR is 3.24, mean SIR is 8.44 and the Mean SAR is 8.85.
# Total parameters are 2363393.Training time is about 23000ms.(100 epochs)
# Using FNN predicting the mask with IRM, the mean SDR is 11, 
# the mean SIR is 14.97 and the mean SAR is 13.57.Total parameters are 2363393.
# Training time is about 24800ms(100 epochs).
# Dense layers are modeled both in Tasks1 and Tasks 2. For STFT,
# win_size = 4096, fft_size = 4096, hop = 1024.According to experiment 
# this is better win_size =2098.
# Using RNN predicting, the mean SDR is 1.37, the mean SIR is 8.69 and the mean
# SAR is 2.83.Total parameters are 6560769. Training time is rather long whih
# is 550000ms(50 epochs). The RNN doesn't perform as well as the FNN. Because 
# the FNN layer's output can have the vector with reduced dimensions.

