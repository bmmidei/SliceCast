#%%
import os, re
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np 
from netUtils import BatchGen
from tensorflow.keras.layers import Layer, Dense, Input, Lambda, \
    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
from tensorflow.keras import Model, Sequential
import tensorflow.keras.backend as K
import h5py
#from tensorflow.keras.utils import to_categorical

'''Universal Sentence Embedding Layer'''
def UniversalEmbedding(input):
    # Explicitly cast the input tensor to strings
    x_str = tf.cast(input, tf.string)

    # Embed each sentence of the input tensor
    embedTens = tf.map_fn(lambda x: embed(x), x_str, dtype=tf.float32)
    return embedTens

def defineModel():
    # Batch shape not included in Input shape
    encoderIn = Input(shape=[None], dtype=tf.string, name='encoderIn')
    encoderOut = Lambda(UniversalEmbedding, output_shape=(None, None, 512), name='encoderOut')(encoderIn)
    lstm1 = Bidirectional(LSTM(128, return_sequences=True), name='lstm_1', input_shape=(None, 512))(encoderOut)
    lstm2 = Bidirectional(LSTM(128, return_sequences=True), name='lstm_2', input_shape=(None, 256))(lstm1)
    preds = Dense(1)(lstm2)
    #preds = Activation('softmax')(preds)

    model = Model(inputs=[encoderIn], outputs=[preds])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(url)

files = ['/Users/bmmidei/Projects/SliceCast/data/hdf5/batch0_1.hdf5',
        '/Users/bmmidei/Projects/SliceCast/data/hdf5/batch0_2.hdf5',
        '/Users/bmmidei/Projects/SliceCast/data/hdf5/batch0_3.hdf5']

gen = BatchGen(files)

with tf.Session() as sess:
    K.set_session(sess)
    model = defineModel()
    class_weight = {0: 1.,
                    1: 50.}

    initOp = [tf.global_variables_initializer(), tf.initializers.tables_initializer()]
    sess.run(initOp)

    # Shout out https://github.com/tensorflow/datasets/issues/233
    history = model.fit(gen.itr.get_next(), 
                        steps_per_epoch=20,
                        epochs=5,
                        verbose=2,
                        class_weight=class_weight)
    #model.save_weights('./model.h5')