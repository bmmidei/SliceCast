#%%
#######################################
# Imports and function definitions
#######################################
import os, re
import tensorflow as tf
print(tf.__version__)

tf.logging.set_verbosity(tf.logging.ERROR)

import tensorflow_hub as hub
import numpy as np 
from netUtils import batchGen, getTestSet
#from tensorflow.keras.layers import Layer, Dense, Input, Lambda, Dropout,\
#    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
#from tensorflow.keras import Model, Sequential
#from tensorflow.keras.models import model_from_yaml
#import tensorflow.keras.backend as K

from keras.layers import Layer, Dense, Input, Lambda, Dropout,\
    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
from keras import Model, Sequential
from keras.models import model_from_json
import keras.backend as K
from keras.callbacks import ModelCheckpoint

import h5py
import matplotlib.pyplot as plt

print('importing hub')
url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(url)

class SliceNet():
    def __init__(self, pretrain=False, weights_path=None):
        self.pretrain = pretrain # Not implemented
        
        if self.pretrain:
            self.weights_path = weights_path # Not implemented
            
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
        encoderIn = Input(shape=[None,], dtype='string', name='encoderIn')
        encoderOut = Lambda(self.UniversalEmbedding, name='encoderOut')(encoderIn)
        lstm1 = Bidirectional(LSTM(256, return_sequences=True), name='lstm_1')(encoderOut)
        lstm2 = Bidirectional(LSTM(256, return_sequences=True), name='lstm_2')(lstm1)
        
        output = Dropout(0.2)(lstm2)
        output = TimeDistributed(Dense(256, activation='relu'))(output)
        output = Dropout(0.2)(output)
        output = TimeDistributed(Dense(64, activation='relu'))(output)
        output = Dropout(0.2)(output)
        preds = TimeDistributed(Dense(3, activation='softmax'))(output)
        
        model = Model(inputs=encoderIn, outputs=preds)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, train_files, val_files, batch_size=16, epochs=3, steps_per_epoch=1000, maxlen=None, save=True):
        
        # Define batch generator
        trainGen = batchGen(train_files, batch_size, maxlen)
        valGen = batchGen(val_files, 4, maxlen)
        
        # I think these need to sum to 1
        #weights = [0.3, 2.64, 0.06]
        weights = None
        self.model.summary()
        
        print('Starting Training')
        with tf.Session() as sess:
            K.set_session(sess)
            initOp = [tf.global_variables_initializer(),
                      tf.initializers.tables_initializer()]
            sess.run(initOp)
            
            
            cb = ModelCheckpoint('./models/weights_epoch{epoch:03d}.h5', 
                                         save_weights_only=True, period=2)
            
            history = self.model.fit_generator(trainGen,
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          verbose=1,
                                          validation_data=valGen,
                                          validation_steps=10,
                                          class_weight=weights,
                                          callbacks=[cb])
            
            #TODO add some sort of keras callback function
            #consider updating loss function from accuracy to precision recall or MSE
            
            if save:
                # Serialize weights to HDF5
                self.model.save_weights('./models/weights_final.h5')
                print("Saved weights to disk")
                
        return history

    def predict(self, test_file, num_samples, weights_path):
        # Get test data and test labels
        X_test, y_test = getTestSet(test_file, num_samples=num_samples)
        
        print('Starting Testing')
        with tf.Session() as sess:
            K.set_session(sess)
            
            initOp = [tf.global_variables_initializer(), tf.initializers.tables_initializer()]
            sess.run(initOp)
            
            # load weights into new model
            self.model.load_weights(weights_path)
            print("Loaded weights from disk")
            
            preds = self.model.predict(X_test)
            
        return preds, y_test
    
    def singlePredict(self, X_test, weights_path):
        
        print('Starting Testing')
        with tf.Session() as sess:
            K.set_session(sess)
            
            initOp = [tf.global_variables_initializer(), tf.initializers.tables_initializer()]
            sess.run(initOp)
            
            # load weights into new model
            self.model.load_weights(weights_path)
            print("Loaded weights from disk")
            
            preds = self.model.predict(X_test)
        
        return preds