#%%
# Small Change
import tensorflow as tf
import tensorflow_hub as hub
import os
import re
from tensorflow.keras.layers import Layer, Dense, Input, Lambda, \
    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
import numpy as np 
from netUtils import docGen
import h5py
from tensorflow.keras.utils import to_categorical

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
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()

    return model
#%%

with tf.Graph().as_default():
    url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    embed = hub.Module(url)
   
    files = ['/Users/bmmidei/Projects/SliceCast/data/hdf5/0.hdf5',
            '/Users/bmmidei/Projects/SliceCast/data/hdf5/1.hdf5',
            '/Users/bmmidei/Projects/SliceCast/data/hdf5/2.hdf5',
            '/Users/bmmidei/Projects/SliceCast/data/hdf5/3.hdf5']

    gen = docGen(files)
    with h5py.File('/Users/bmmidei/Projects/SliceCast/data/hdf5/0.hdf5', 'r') as hf:
        # Iterate through each document within the file
        x = 0
        for _, grp in hf.items():
            # Extract items and return
            sents = grp['sents'].value
            labels = grp['labels'].value
            #labels = to_categorical(labels)
            testx, testy =  sents, labels
            testx = np.expand_dims(testx, axis=0)
            testy = np.expand_dims(testy, axis=0)
            x += 1
            if x>3:
                break
    with tf.Session() as sess:
        initOp = [tf.global_variables_initializer(), tf.initializers.tables_initializer()]
        sess.run(initOp)
        model = defineModel()
        class_weight = {0: 1.,
                        1: 50.}
        model.fit_generator(gen, steps_per_epoch=40, verbose=2, class_weight=class_weight)

        layer_name = 'encoderOut'
        #intermediate_layer_model = Model(inputs=model.input,
        #                                outputs=model.get_layer(layer_name).output)
        #intermediate_output = intermediate_layer_model.predict(testx)
        print('generating predictions')
        preds = model.predict(testx)
with tf.Graph().as_default():
    url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    embed = hub.Module(url)
    embeddings = embed([
        "WTTE began operations on June 1, 1984 as the first general-entertainment independent station in central Ohio.",
        "The station was founded by the Commercial Radio Institute, a subsidiary of the Baltimore-based Sinclair Broadcast Group."
        ])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        testEmbed = sess.run(embeddings)
#%%
print(preds)
print(testy)
#%%
print(intermediate_output.shape)
print(testEmbed.shape)
#%%
import numpy as np
a = intermediate_output[0,0,:]
b = testEmbed[0,:]
print(a.shape)
print(b.shape)
print(a[:20])
print(b[:20])
for i, j in zip(a, b):
    if abs(i-j)>1e-7:
        print(i,j)