import h5py
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
                assert n == len(labels) # ensure docs and labels are same length
                num_batches = np.floor(n/batch_size).astype(np.int16)

                for idx in range(num_batches):
                    # Get each batch of documents and labels
                    batch_docs = docs[idx*batch_size: (idx+1)*batch_size]
                    batch_labels = labels[idx*batch_size: (idx+1)*batch_size]

                    # pad docs and labels to the length of the longest sample in the batch
                    padded_docs = pad_sequences(batch_docs, dtype=object, value=' ')
                    
                    if classification:
                        padded_labels = pad_sequences(batch_labels, dtype=int, value=2)
                        padded_labels = to_categorical(padded_labels, num_classes=3, dtype='int32')
                    else:
                        padded_labels = pad_sequences(batch_labels, dtype=int, value=0)
                        padded_labels = np.expand_dims(padded_labels, axis=-1)
                        
                    yield(padded_docs, padded_labels)   

def getTestSet(fname, num_samples=16, classification=True):
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
