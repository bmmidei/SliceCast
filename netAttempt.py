# Small Change
import tensorflow as tf
import tensorflow_hub as hub
import os
import re
from tensorflow.keras.layers import Layer, Dense, Input, Lambda, \
    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)
def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)
def call(self, x, mask=None):
        elmo_embds = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1), signature="default", as_dict=True)
        lstm1_embd = elmo_embds['lstm_outputs1'] #?xTXD
        lstm2_embd = elmo_embds['lstm_outputs2'] #?xTXD
        w_embd = tf.identity(elmo_embds['elmo'], name='elmo_word_embd') #?xTXD
        #taking index of last word in each sentence
        idx = elmo_embds['sequence_len']-1
        batch_idx = tf.stack([tf.range(0,tf.size(idx),1),idx],axis=1)
        # Concatenate first of backward with last of forward to get sentence embeddings
        dim = lstm1_embd.get_shape().as_list()[-1]
        sen_embd_1 = tf.concat([lstm1_embd[:,0,int(dim/2):],
        tf.gather_nd(lstm1_embd[:,:,:int(dim/2)],batch_idx)], axis=-1) #[batch,dim]
        sen_embd_2 = tf.concat([lstm2_embd[:,0,int(dim/2):],
        tf.gather_nd(lstm2_embd[:,:,:int(dim/2)],batch_idx)], axis=-1) #[batch,dim]
        sen_embd = tf.concat([tf.expand_dims(sen_embd_1,axis=2),
        tf.expand_dims(sen_embd_2,axis=2)], axis=2, name='elmo_sen_embd') #[batch,dim,2]
        print(sen_embd)
        e_s = tf.keras.layers.Dense(units=1,use_bias=False)(sen_embd) #?xDx1
        e_s = tf.squeeze(e_s,axis=2)
        return e_s
def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')
def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

model = Sequential()
model.add(ElmoEmbeddingLayer())
model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=(timesteps, data_dim))))
model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=(timesteps, data_dim))))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
