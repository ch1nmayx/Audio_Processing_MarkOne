# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 20:50:39 2018

@author: Admin
"""

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

dat_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\test_dat'
out_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\python3-ver\out_genre\data_test.pkl'

pkl_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\python3-ver\out_genre\data_test.pkl'
model_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\python3-ver\models\model_test.yaml'
weights_path = r'G:\000. UCD- Data Science\Research Project\Deep-ListeningX\python3-ver\models\weights_test.h5'

SEED = 42
N_LAYERS = 4
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
LSTM_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 50

with open(out_path, 'rb') as f:
    data = pickle.load(f)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

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
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(2)(layer)

layer  = Dense(500, init='normal')(layer)
layer = BatchNormalization()(layer)
layer  = Activation("relu")(layer)
layer  = Dense(500, init='normal')(layer)
layer = BatchNormalization()(layer)
layer  = Activation("relu")(layer)
layer  = Dense(len(GENRES))(layer)
layer = BatchNormalization()(layer)
layer  = Activation("softmax")(layer)

# softmax classifier
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

