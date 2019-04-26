import h5py
from pathlib import Path
from spacyOps import createInferencePipe, createSpacyPipe
import numpy as np
import tensorflow as tf
from keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

def batchGen(filenames, batch_size=16, maxlen=None, classification=True):
    """
    Generator function for batches of documents and labels
    from a list of HDF5 files
    Args:
        filenames: list of HDF5 filenames
        batch_size: size of each batch to yield from generator
        maxlen: maximum length of each example document
    Yields:
        padded_docs, padded_labels: A tuple of padded documents and
                                    corresponding labels.
    """
    while True:
        for fname in filenames:
            with h5py.File(fname, 'r') as hf:
                # Get a list of all examples in the file 
                groups = [item[1] for item in hf.items()]

                # Get lists of all sentences and all labels 
                docs = [grp['sents'][()] for grp in groups]
                docs = [docs.tolist() for docs in docs]
                labels = np.array([grp['labels'][()] for grp in groups])

                # Only get examples longer than 0 and less than maxlen
                if maxlen:
                    docs = [x for x in docs if len(x)<maxlen and len(x)>0]
                    labels = [x for x in labels if len(x)<maxlen and len(x)>0]

                # Only get examples longer than 0
                else:
                    docs = [x for x in docs if len(x)>0]
                    labels = [x for x in labels if len(x)>0]

                # Shuffle documents and labels
                docs, labels = shuffle(docs, labels)

                n = len(docs)
                assert n == len(labels) # Ensure docs and labels are same length
                num_batches = np.floor(n/batch_size).astype(np.int16)

                for idx in range(num_batches):
                    # Get each batch of documents and labels
                    batch_docs = docs[idx*batch_size: (idx+1)*batch_size]
                    batch_labels = labels[idx*batch_size: (idx+1)*batch_size]

                    # Pad docs and labels to the length of the longest sample in the batch
                    padded_docs = pad_sequences(batch_docs, dtype=object, value=' ')
                    
                    # 
                    if classification:
                        padded_labels = pad_sequences(batch_labels, dtype=int, value=2)
                        padded_labels = to_categorical(padded_labels, num_classes=3, dtype='int32')
                    else:
                        padded_labels = pad_sequences(batch_labels, dtype=int, value=0)
                        padded_labels = np.expand_dims(padded_labels, axis=-1)
                        
                    yield(padded_docs, padded_labels)   

def getTestSet(fname, num_samples=16, classification=True):
    """Generate a test set from a single HDF5 file
    Args:
        fname: Path to HDF5 file
        num_samples: Desired number of samples to return in the datset
        classification: Boolean indicating whether test set should be prepared
                        for classification or regression.
    Returns:
        padded_docs:
        padded_labels:
    """
    with h5py.File(fname, 'r') as hf:
        # Get a list of all examples in the file 
        groups = [item[1] for item in hf.items()]

        # Get lists of all sentences and all labels 
        docs = [grp['sents'][()] for grp in groups]
        docs = [docs.tolist() for docs in docs]
        labels = np.array([grp['labels'][()] for grp in groups])

        # Only get examples longer than 0
        docs = [x for x in docs if len(x)>0]
        labels = [x for x in labels if len(x)>0]

        # Shuffle documents and labels
        docs, labels = shuffle(docs, labels)

        assert len(docs) == len(labels) # ensure docs and labels are same length

        # Get each batch of documents and labels
        docs = docs[:num_samples]
        labels = labels[:num_samples]

        # pad docs and labels to the length of the longest sample in the batch
        padded_docs = pad_sequences(docs, dtype=object, value=' ')
        
        if classification:
            padded_labels = pad_sequences(labels, dtype=int, value=2)
            padded_labels = to_categorical(padded_labels, num_classes=3, dtype='int32')
        else:
            padded_labels = pad_sequences(labels, dtype=int, value=0)
            padded_labels = np.expand_dims(padded_labels, axis=-1)
        
        return padded_docs, padded_labels
    
# Credit: https://github.com/keras-team/keras/issues/3653
def customCatLoss(onehot_labels, logits):
    """Custom categorical loss function incorporating class weights
    Args:
        onehot_labels: onehot encoded labels - shape = [batch x doclength x numclasses]
        logits: logits from predictions from network
    Returns:
        loss: average loss for the mini-batch
    """
    class_weights = [1.0, 26.0, 0.2]
    # computer weights based on onehot labels
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)

    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits)

    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights

    # average to get final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss

def getSingleExample(fname, is_labeled=True):
    """Retrieve array of sentences and labels (if applicable)
    for a single text file to be used in inference"""
    # Run NLP pipeline on the single text file
    fo = Path(fname)
    nlp = createInferencePipe()
    doc = nlp(fo.read_text(encoding='utf-8'))
    
    # Extract sentences and labels from tokenized document
    sents = doc.user_data['sents']
    if is_labeled:
        labels = doc.user_data['labels']
    else:
        labels = None
        
    assert len(sents)>0, 'Document length is 0'
    
    sents = np.array(sents, dtype='object')
    return sents, labels