#%%
#######################################
# Imports and function definitions
#######################################
import os, re
import tensorflow as tf
print(tf.__version__)

import tensorflow_hub as hub
import numpy as np 
from netUtils import batchGen
from tensorflow.keras.layers import Layer, Dense, Input, Lambda, Dropout,\
    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
from tensorflow.keras import Model, Sequential
import tensorflow.keras.backend as K
import h5py
import matplotlib.pyplot as plt

print('importing hub')
url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(url)

class SliceNet():
    def __init__(self, pretrain=False,  model_path=None):
        self.pretrain = pretrain # Not implemented
        if self.pretrain:
            self.model_path = model_path # Not implemented
        self.model = self._defineModel()
    
    def map_docs(self, x):
        return embed(x, signature='default', as_dict=True)['default']

    def UniversalEmbedding(self, x):
        # Explicitly cast the input tensor to strings
        x_str = tf.cast(x, tf.string)
        
        # Embed each sentence of the input tensor
        embedTens = tf.map_fn(self.map_docs, x_str, dtype=tf.float32)
        return embedTens

    def _defineModel(self):
        # Define network structure
        encoderIn = Input(shape=[None,], dtype=tf.string, name='encoderIn')
        encoderOut = Lambda(self.UniversalEmbedding, name='encoderOut')(encoderIn)
        lstm1 = Bidirectional(LSTM(256, return_sequences=True), name='lstm_1')(encoderOut)
        lstm2 = Bidirectional(LSTM(256, return_sequences=True), name='lstm_2')(lstm1)
        output = Dropout(0.2)(lstm2)
        output = TimeDistributed(Dense(128, activation='relu'))(output)
        output = Dropout(0.2)(output)
        preds = Dense(1, activation='sigmoid')(output)

        model = Model(inputs=encoderIn, outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    #######################################
    # Training
    #######################################
    def train(self, train_files, val_files, batch_size=16, epochs=3, steps_per_epoch=1000, maxlen=None, save=True):
        
        # Define batch generator
        trainGen = batchGen(train_files, batch_size, maxlen)
        valGen = batchGen(val_files, 4, maxlen)
        
        # I think these need to sum to 1
        weights = [0.1, 0.9]
        self.model.summary()
        
        print('Starting Training')
        with tf.Session() as sess:
            K.set_session(sess)
            initOp = [tf.global_variables_initializer(),
                      tf.initializers.tables_initializer()]
            sess.run(initOp)
            
            history = self.model.fit_generator(trainGen,
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          verbose=1,
                                          class_weight=weights)
            
            #TODO add some sort of keras callback function
            #TODO add validation data to feed in to fit generator
                # validation_data = valGen
                # validation_steps = number of batches to yield from valGen
            #consider updating loss function from accuracy to precision recall or MSE
            
            if save:
                # Save model metadata
                model_yaml = self.model.to_yaml()
                with open('./models/model.yaml', 'w') as yaml_file:
                    yaml_file.write(model_yaml)

                # Serialize weights to HDF5
                self.model.save_weights('./models/model.h5')
                print("Saved model to disk")
                
        return history

#%%
#######################################
# Inference 
# Not yet implemented
#######################################
'''
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
'''