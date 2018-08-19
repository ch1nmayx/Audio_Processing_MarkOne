# -*- coding: utf-8 -*-
"""
@author: Chinmay Sinha

"""

'''
GPU command:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_model.py
'''

from common import GENRES
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Dropout, Activation, LSTM, \
        TimeDistributed, Convolution1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from optparse import OptionParser
from sys import stderr, argv
import os

SEED = 42
N_LAYERS = 5
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
LSTM_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 50

dat_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\genres'
out_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\python3-ver\out_genre\data_fin.pkl'

pkl_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\python3-ver\out_genre\data_fin.pkl'
model_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\python3-ver\models\model_conv3.yaml'
weights_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\python3-ver\models\weights_conv3.h5'

with open(out_path, 'rb') as f:
    data = pickle.load(f)


x = data['x']
y = data['y']
(x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.4,
        random_state=SEED)

#print('Building model...')

n_features = x_train.shape[2]
input_shape = (None, n_features)
model_input = Input(input_shape, name='input')
layer = model_input
for i in range(N_LAYERS):
    # convolutional layer names are used by extract_filters.py
    layer = Convolution1D(
            nb_filter=CONV_FILTER_COUNT,
            filter_length=FILTER_LENGTH,
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

print('Training...')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
          validation_data=(x_val, y_val), verbose=1)


print(model.summary())


validation_size = 200

X_validate = x_val[-validation_size:]
Y_validate = y_val[-validation_size:]
X_test = x_val[:-validation_size]
Y_test = y_val[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))



predictions = model.predict(x_val)
y_pred = (predictions > 0.5)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))


