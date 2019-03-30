# Small Change
import tensorflow as tf
import tensorflow_hub as hub
import os
import re
from tensorflow.keras.layers import Layer, Dense, Input, Lambda, \
    Bidirectional, LSTM, Activation, TimeDistributed, Concatenate
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K

from netUtils import docGen
import h5py
from tensorflow.keras.utils import to_categorical

def defineModel():
    # Batch shape not included in Input shape
    encoderIn = Input(shape=[maxlen], dtype=tf.string, name='encoderIn')
    encoderOut = Lambda(UniversalEmbedding, output_shape=(None, maxlen, 512), name='encoderOut')(encoderInputs)
    merged = Bidirectional(LSTM(128, return_sequences=True), input_shape=(maxlen, 512))(encoded)
    preds = Dense(2)(merged)
    preds = Activation('softmax')(preds)

    model = Model(inputs=[encoderInputs], outputs=[preds, encoderIn, encoderOut])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()

    return model
 
g1 = tf.Graph()
with g1.as_default():
    url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    embed = hub.Module(url)
    embeddings = embed([
        "The quick brown fox jumps over the lazy dog.",
        "I am a sentence for which I would like to get its embedding"])
    print(embeddings)


    maxlen = None # we might need to set a max document length and pad
    def UniversalEmbedding(input):
        x_str = tf.cast(input, tf.string)
        print(x_str.shape)
        embedTens = tf.map_fn(lambda x: embed(x), x_str, dtype=tf.float32)
        # Flatten because the universal sentence encoder takes a 1 dimensional input
        #embedFlat = embed(tf.reshape(tf.cast(x, tf.string), shape=[-1]))
        print(embedTens.shape)
        # Reshape back into the correct dimensions
        # return tf.reshape(embedFlat, [-1, maxlen, 512])
        print(embedTens.dtype)
        return embedTens
    
    files = ['/Users/bmmidei/Projects/SliceCast/data/hdf5/0.hdf5']
    gen = docGen(files)
    with h5py.File('/Users/bmmidei/Projects/SliceCast/data/hdf5/0.hdf5', 'r') as hf:
        # Iterate through each document within the file
        for _, grp in hf.items():
            # Extract items and return
            sents = grp['sents'].value
            labels = grp['labels'].value
            labels = to_categorical(labels)
            testx, testy =  sents, labels
            break
    #files = ['/Users/bmmidei/Projects/SliceCast/data/hdf5/1.hdf5']
    #testGen = docGen(files)
    with tf.Session() as sess:
        initOp = [tf.global_variables_initializer(), tf.initializers.tables_initializer()]
        sess.run(initOp)
        model = defineModel()
        model.fit_generator(gen, steps_per_epoch=20, verbose=2)
        print('moving on')
        testx, testy
        print('generating predictions')
        print(model.predict(testx))
        print(testy)
'''
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
'''