#%%
#######################################
# Imports and function definitions
#######################################
import os, re
import tensorflow as tf
print(tf.__version__)

import tensorflow_hub as hub
import numpy as np 
from netUtils import batchGen, getTestSet, customCatLoss
from postprocess import pkHistory

from keras.layers import Layer, Dense, Input, Lambda, Dropout, Flatten,\
    Bidirectional, Activation, TimeDistributed, Concatenate, merge, Reshape
from keras.layers import CuDNNLSTM as LSTM

from keras import Model, Sequential
from keras import regularizers
from keras.models import model_from_json
import keras.backend as K
from keras.callbacks import ModelCheckpoint

import h5py
import matplotlib.pyplot as plt

print('importing hub')
url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(url, trainable=True)

class SliceNet():
    def __init__(self, classification,
                 class_weights,
                 pretrain=False,
                 weights_path=None,
                 maxlen=None,
                 drop_prob=0.2,
                 reg=1e-2):

        self.pretrain = pretrain # Not implemented
        self.classification = classification
        self.drop_prob = drop_prob
        self.reg = regularizers.l2(reg)
        self.class_weights = class_weights
        self.maxlen = maxlen
        if self.pretrain:
            self.weights_path = weights_path
            
        self.model = self._defineModel()
    
    def map_docs(self, x):
        """Helper function to embed a give list of sentences using
        the google universal sentence encoder from tfhub"""
        return embed(x, signature='default', as_dict=True)['default']
    
    def UniversalEmbedding(self, x):
        """Embedding layer that calls the map_docs function for each
        document in the batch. The result is an encoding for each sentence
        of each document in the mini-batch"""
        
        # Explicitly cast the input tensor to strings
        x_str = tf.cast(x, tf.string)

        # Embed each sentence of the input tensor
        embedTens = tf.map_fn(self.map_docs, x_str, dtype=tf.float32)
        return embedTens
     
    def _defineModel(self):
        # Define network structure
        encoderIn = Input(shape=(self.maxlen,), dtype='string', name='encoderIn')
        encoderOut = Lambda(self.UniversalEmbedding, name='encoderOut')(encoderIn)
        lstm1 = Bidirectional(LSTM(256, return_sequences=True), name='lstm_1')(encoderOut)
        activations = Bidirectional(LSTM(256, return_sequences=True), name='lstm_2')(lstm1)

        # compute importance for each step
        # https://github.com/keras-team/keras/issues/4962
        attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
        attention = Reshape(target_shape=[-1])(attention)
        attention = Activation('softmax')(attention)
        
        
        output = Lambda(lambda x: tf.einsum('bs,bsd->bsd',x[0], x[1]))([attention, activations])
        output = TimeDistributed(Dense(256, activation='relu',
                                           kernel_regularizer=self.reg))(output)
        output = Dropout(self.drop_prob)(output)
        output = TimeDistributed(Dense(128, activation='relu',
                                           kernel_regularizer=self.reg))(output)
        output = Dropout(self.drop_prob)(output)
        output = TimeDistributed(Dense(64, activation='relu', 
                                           kernel_regularizer=self.reg))(output)
        # Final output is different for classification and regression models
        if self.classification:
            preds = TimeDistributed(Dense(3, activation='softmax'))(output)
            model = Model(inputs=encoderIn, outputs=preds)
            model.compile(loss=customCatLoss(self.class_weights),
                          optimizer='adam',
                          metrics=['categorical_accuracy'])
        else:
            preds = TimeDistributed(Dense(1, activation='sigmoid'))(output)
            model = Model(inputs=encoderIn, outputs=preds)
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        return model

    def train(self, train_files,
                    val_files,
                    test_file,
                    batch_size=16,
                    epochs=3,
                    steps_per_epoch=1000,
                    save=True,
                    k=7):
        
        # Define batch generator
        trainGen = batchGen(train_files, batch_size, self.maxlen, classification=self.classification)
        valGen = batchGen(val_files, 4, self.maxlen, classification=self.classification)
        self.model.summary()
        
        print('Starting Training')
        with tf.Session() as sess:
            K.set_session(sess)
            initOp = [tf.global_variables_initializer(),
                      tf.initializers.tables_initializer()]
            sess.run(initOp)
            if self.pretrain:
                self.model.load_weights(self.weights_path)

            
            save_weights = ModelCheckpoint('./models/weights_epoch{epoch:03d}.h5', 
                                         save_weights_only=True, period=2)
            
            pkscores = pkHistory(test_file=test_file, num_samples=100, k=k)
            
            history = self.model.fit_generator(trainGen,
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          verbose=1,
                                          validation_data=valGen,
                                          validation_steps=10,
                                          callbacks=[save_weights, pkscores])
                        
            if save:
                # Serialize weights to HDF5
                self.model.save_weights('./models/weights_final.h5')
                print("Saved weights to disk")
                
        return history, pkscores

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
