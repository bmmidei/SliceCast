import h5py
import tensorflow as tf

def docGen(fname):
    """
    Generator function for a single HDF5 file
    Args:
        fname: filename for a single HDF5 file
    Yields:
        sents, labels: A tuple of sentences and corresponding
                       labels for a single example.
    """
    with h5py.File(fname, 'r') as hf:
        # Get a list of all examples in the file 
        groups = [item[1] for item in hf.items()]

        # Get lists of all sentences and all labels 
        sents = [grp['sents'][()] for grp in groups]
        labels = [grp['labels'][()] for grp in groups]
    
    # Iterate through all sentences and corresponding labels
    # and yield one at a time
    for sent, label in zip(sents, labels):
        if len(label)==0:
            print('encountered empty')
            pass
        else:
            yield(sent.astype(str), label)

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
        if shuffle:
            ds = ds.shuffle(buffer_size=16)
        if repeat:
            ds = ds.repeat()

        self.ds = ds
        self.itr = self.ds.make_one_shot_iterator()

        ''' Potentiall include bucketing later on for batches larger than 1
        ds = ds.apply(tf.data.experimental.bucket_by_sequence_length(
                        element_length_func=element_length_fn,
                        bucket_boundaries=[5, 10, 20, 40],
                        bucket_batch_sizes=[4, 4, 4, 4, 4]))
        '''

    def next_batch(self):
        return self.itr.get_next()