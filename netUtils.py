import h5py
import tensorflow as tf
from keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def batchGen(filenames, batch_size=16, maxlen=None):
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
                
            n = len(docs)
            assert n == len(labels) # ensure docs and labels are same length
            num_batches = np.floor(n/batch_size).astype(np.int16)
            
            for idx in range(num_batches):
                # Get each batch of documents and labels
                batch_docs = docs[idx*batch_size: (idx+1)*batch_size]
                batch_labels = labels[idx*batch_size: (idx+1)*batch_size]
                
                # pad docs and labels to the length of the longest sample in the batch
                padded_docs = pad_sequences(batch_docs, dtype=object, value=' ')
                padded_labels = pad_sequences(batch_labels, dtype=int, value=0)
                padded_labels = np.expand_dims(padded_labels, axis=-1)
                yield(padded_docs, padded_labels)   
                
'''
def docGen(fname):
    """
    Generator function for a single HDF5 file
    Args:
        fname: filename for a single HDF5 file
    Yields:
        sents, labels: A tuple of sentences and corresponding
                       labels for a single example.
    """
    maxlen=30
    with h5py.File(fname, 'r') as hf:
        # Get a list of all examples in the file 
        groups = [item[1] for item in hf.items()]

        # Get lists of all sentences and all labels 
        docs = [grp['sents'][()] for grp in groups]
        docs = [docs.tolist() for docs in docs]
        labels = np.array([grp['labels'][()] for grp in groups])

        docs_mod = [x for x in docs if len(x)<maxlen]
        labels_mod = [x for x in labels if len(x)<maxlen]
        padded_docs = pad_sequences(docs_mod, maxlen=maxlen, dtype=object, value=' ')
        padded_labels = pad_sequences(labels_mod, maxlen=maxlen, dtype=int, value=0)
        print(type(padded_labels))
        #padded_labels = np.expand_dims(padded_labels, axis=-1)


    # Iterate through all sentences and corresponding labels
    # and yield one at a time
    for doc, label in zip(padded_docs, padded_labels):
        if len(label)==0:
            pass
        else:
            yield(doc, label)

class BatchGen(object):
    def __init__(self, filenames, batch_size=1, shuffle=True, repeat=True):
        # Create dataset from filenames
        ds = tf.data.Dataset.from_tensor_slices(filenames)

        # Use flat map to apply a transformation to each HDF5 file.
        # Each file creates a dataset of examples. Then all examples are combined
        # into a single dataset
        ds = ds.flat_map(lambda filename: tf.data.Dataset.from_generator(
                                            generator=docGen,
                                            output_types=(tf.string, tf.int64),
                                            output_shapes=([None,], [None,]),
                                            args=([filename])))
        # Apply dataset shuffling, batching, and repeating
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(100)
        if shuffle:
            ds = ds.shuffle(buffer_size=16)
        if repeat:
            ds = ds.repeat()

        self.ds = ds
        self.itr = self.ds.make_one_shot_iterator()

    def __next__(self):
        return self.itr.get_next()
'''
