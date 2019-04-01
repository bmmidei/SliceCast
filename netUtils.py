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
def docGen(files, batch_size=4):
    cnt = 0
    batchx = []
    batchy = []
    for f in files:
        # Iterate through each hdf5 file
        with h5py.File(f, 'r') as hf:
            # Iterate through each document within the file
            for _, grp in hf.items():
                # Extract items and return
                sents = grp['sents'].value
                labels = grp['labels'].value
                #labels = to_categorical(labels)
                sents = np.expand_dims(sents, axis=0)
                labels = np.expand_dims(labels, axis=0)
                batchx.append(sents.tolist)
                batchy.append(labels.tolist)
                cnt+=1
                if cnt==batch_size:
                    print('yield')
                    print(len(batchx))
                    print(len(batchy))
                    #batchx = np.array(batchx, dtype='object')
                    #batchy = np.array(batchy, dtype='object')
                    yield batchx, batchy
                    cnt=0
                    batchx = []
                    batchy = []
