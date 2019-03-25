import tensorflow as tf

import tensorflow_hub as hub
import os
import re
from tensorflow.keras.layers import Layer, Dense, Input, Lambda, \
    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K

def ElmoLayer(x):
    #sentence placeholder - list of sentences
    text_batch = x
    #text_batch = tf.placeholder('string', shape=[None], name='text_input')
    #loading pre-trained ELMo
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    #getting ELMo embeddings
    elmo_embds = elmo(text_batch, signature="default", as_dict=True)
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

class ElmoEmbeddingLayer(Layer):
    def _init_(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self)._init_(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

def build_model(): 
    with tf.Graph().as_default():
        input_text = Input(shape=(None,), dtype=tf.string)
        embedding = Lambda(ElmoLayer, output_shape=(None, 1024))(input_text)
        x = Bidirectional(LSTM(units=512, return_sequences=True,
                            recurrent_dropout=0.2, dropout=0.2))(embedding)
        x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                                recurrent_dropout=0.2, dropout=0.2))(x)
        x = Concatenate([x, x_rnn])  # residual connection to the first biLSTM
        out = TimeDistributed(Dense(2, activation="softmax"))(x)
        model = Model(input_text, out)

        return model
    '''
    with tf.Graph().as_default():
        model = Sequential()
        model.add(Lambda(ElmoLayer))
        model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=(None, 1024))))
        model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=(None, 1024))))

        # paper suggests multiplying with a 2d matrix in the end, assuming this is for binary classification
        model.add(Dense(2))

        # activation function
        model.add(Activation('softmax'))

        # optimizer we can switch
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.build((None,))

        return model
    '''
'''
def Net():

    timesteps=None because each sentence has variable number of words
    data_dim=shape of vector representation of our words (depends on which GloVe we choose)
    original paper suggests 2 layers, 256 units and concatenation for sentence vector representation
    N fixed vector size for sentence representations
    model
    model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=(timesteps, data_dim))))
    model.add(Lambda(lambda x: x[:, -N:, :]))
    model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=(timesteps, data_dim)), merge_mode='concat'))


    timesteps=None because each document has variable number of sentences
    data_dim=N (shape of vector representation of our sentences)
    2 layers, 256 units
    model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=(timesteps, data_dim))))
    model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=(timesteps, data_dim))))

    # paper suggests multiplying with a 2d matrix in the end, assuming this is for binary classification
    model.add(Dense(2))

    # activation function
    model.add(Activation('softmax'))

    # optimizer we can switch
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model
'''