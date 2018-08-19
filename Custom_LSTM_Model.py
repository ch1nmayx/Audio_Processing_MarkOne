# -*- coding: utf-8 -*-
"""
@author: Custom LSTM Network

"""

# Configuration of the Custom LSTM Network
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Dropout, Activation, LSTM, \
        TimeDistributed, Convolution1D, MaxPooling1D
        
SEED = 100
NO_OF_LAYERS = 5
LENGTH_OF_FILTER = 5
CONV_FILTER_COUNT = 256
LSTM_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 10

n_features = x_train.shape[2]
input_shape = (None, n_features)
model_input = Input(input_shape, name='input')
layer = model_input
for i in range(NO_OF_LAYERS):
    # convolutional layer names are used by extract_filters.py
    layer = Convolution1D(
            nb_filter=CONV_FILTER_COUNT,
            LENGTH_OF_FILTER=LENGTH_OF_FILTER,
            name='convolution_' + str(i + 1)
        )(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(2)(layer)

layer = Dropout(0.5)(layer)
layer = LSTM(LSTM_COUNT, return_sequences=True)(layer)
layer = Dropout(0.5)(layer)
layer = TimeDistributed(Dense(len(GENRES)))(layer)
layer = Activation('softmax', name='output_realtime')(layer)
time_distributed_merge_layer = Lambda(
        function=lambda x: K.mean(x, axis=1), 
        output_shape=lambda shape: (shape[0],) + shape[2:],
        name='output_merged'
    )
model_output = time_distributed_merge_layer(layer)
model = Model(model_input, model_output)
opt = RMSprop(lr=0.0001)
model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

print(model.summary())