import numpy as np
from pathlib import Path
import h5py
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
'''
class Generator:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            # Iterate through each hdf5 file
            with h5py.File(file, 'r') as hf:
                # Iterate through each document within the file
                for _, grp in hf.items():
                    # Extract items and return
                    sents = grp['sents'].value
                    labels = grp['labels'].value

                    yield sents, labels
'''
def docGen(files):
    for f in files:
        # Iterate through each hdf5 file
        with h5py.File(f, 'r') as hf:
            # Iterate through each document within the file
            for _, grp in hf.items():
                # Extract items and return
                sents = grp['sents'].value
                labels = grp['labels'].value
                labels = to_categorical(labels)
                print('yield')
                yield sents, labels