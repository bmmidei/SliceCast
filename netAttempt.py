#%%
#######################################
# Imports and function definitions
#######################################
import os, re
import tensorflow as tf
print(tf.__version__)

import tensorflow_hub as hub
import numpy as np 
from netUtils import BatchGen, docGen
from tensorflow.keras.layers import Layer, Dense, Input, Lambda, \
    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
from tensorflow.keras import Model, Sequential
import tensorflow.keras.backend as K
import h5py
import matplotlib.pyplot as plt

print('importing hub')
url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(url)

def UniversalEmbedding(input):
    # Explicitly cast the input tensor to strings
    x_str = tf.cast(input, tf.string)

    # Embed each sentence of the input tensor
    embedTens = tf.map_fn(lambda x: embed(x), x_str, dtype=tf.float32)
    return embedTens

def defineModel(classification=True):
    # Batch shape not included in Input shape
    encoderIn = Input(shape=[None,], dtype=tf.string, name='encoderIn')
    encoderOut = Lambda(UniversalEmbedding, output_shape=(None, None, 512), name='encoderOut')(encoderIn)
    lstm1 = Bidirectional(LSTM(128, return_sequences=True), name='lstm_1', input_shape=(None, 512))(encoderOut)
    lstm2 = Bidirectional(LSTM(128, return_sequences=True), name='lstm_2', input_shape=(None, 256))(lstm1)
    preds = Dense(1)(lstm2)

    model = Model(inputs=encoderIn, outputs=preds)
    if classification:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    model.summary()

    return model
#%%
#######################################
# Training
#######################################

# Define files and batch generator
# TODO - replace with a glob call that returns all HDF5s in a given directory
files = ['/Users/bmmidei/Projects/SliceCast/data/hdf5/batch0_1.hdf5',
        '/Users/bmmidei/Projects/SliceCast/data/hdf5/batch0_2.hdf5',
        '/Users/bmmidei/Projects/SliceCast/data/hdf5/batch0_3.hdf5']
gen = BatchGen(files)
save = True

# I think these need to sum to 1
weights = {0: 0.1,
           1: 0.9}

print('starting session')
with tf.Session() as sess:
    K.set_session(sess)
    model = defineModel()
    class_weight = weights
    initOp = [tf.global_variables_initializer(), tf.initializers.tables_initializer()]
    sess.run(initOp)

    # Shout out https://github.com/tensorflow/datasets/issues/233
    history = model.fit(gen.itr.get_next(), 
                        steps_per_epoch=20,
                        epochs=3,
                        verbose=1,
                        class_weight=class_weight)
    if save:
        # Save model metadata
        model_yaml = model.to_yaml()
        with open('./models/model.yaml', 'w') as yaml_file:
            yaml_file.write(model_yaml)

        # Serialize weights to HDF5
        model.save_weights('./models/model.h5')
        print("Saved model to disk")

#%%
#######################################
# Inference 
#######################################
fname = '/Users/bmmidei/Projects/SliceCast/data/hdf5/batch0_4.hdf5'
testGen = docGen(fname)
testx, testy = next(testGen)
#%%
from tensorflow.keras.models import model_from_yaml

with tf.Session() as sess:
    K.set_session(sess)
 
    # Load YAML and create model
    with open('./models/model.yaml', 'r') as yaml_file:
        loaded_model_yaml = yaml_file.read()
    loaded_model = model_from_yaml(loaded_model_yaml)

    # Initialize and then load weights
    initOp = [tf.global_variables_initializer(), tf.initializers.tables_initializer()]
    sess.run(initOp)
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    preds = loaded_model.predict(testx)

#%%
preds = np.squeeze(preds)
for i, j in zip(testy, preds):
    print(i, j)