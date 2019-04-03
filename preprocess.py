from __future__ import unicode_literals
from spacyOps import customLabeler, createSpacyPipe
from pathlib import Path
import spacy
import tensorflow as tf
import time, math
import h5py
import numpy as np

class Pipeline(object):

    def __init__(self, dataPath, ex_per_batch, ex_per_file):
        self.mainPath = Path(dataPath)
        self.trainPath = Path(dataPath).joinpath('train', 'train')
        self.testPath = Path(dataPath).joinpath('train', 'test')
        self.devPath = Path(dataPath).joinpath('train', 'dev')
        self.expPath = Path(dataPath).joinpath('test')
        self.hdf5Path = Path(dataPath).joinpath('hdf5')

        self.ex_per_batch = ex_per_batch
        self.ex_per_file = ex_per_file
        self.nlp = createSpacyPipe()

        print('\nUsing spacy pipeline components:')
        for name, proc in self.nlp.pipeline:
            print(name, proc)

    def processDirectory(self, dataPath, max_examples=None):
        # Get the file paths for every txt file in the directory
        self.getFilePaths(dataPath)

        print('There are {} documents in this directory'.format(self.num_examples))

        if max_examples:
            print('Processing a subset of size {}...'.format(max_examples))
            self.num_examples = max_examples

        # Determine number of batches to process
        self.num_batches = math.ceil(self.num_examples/self.ex_per_batch)

        # Process files in batches to reduce memory impact
        for batchIdx in range(self.num_batches):
            # Initialize docs at beginning of each batch
            docs = []
            self.docs = []
            # Determine start and stop indices in file list for current batch
            startIdx = batchIdx * self.ex_per_batch
            if batchIdx-1 == self.num_batches:
                stopIdx = self.num_examples
            else:
                stopIdx = startIdx + self.ex_per_batch
            
            # Run spacy nlp on each document
            for file in self.files[startIdx:stopIdx]:
                try:
                    doc = self.nlp(file.read_text(encoding='utf-8'))
                    docs.append(doc)
                except Exception as e:
                    print(file)
                    print(e)

            self.docs = docs
            
            # Generate HDF5s for current batch
            self._genHDF5s(batchIdx, startIdx, stopIdx)

    def getFilePaths(self, dataPath):
        self.files = [x for x in dataPath.glob('**/*') if x.is_file()]
        self.num_examples = len(self.files)

    @staticmethod
    def genSingleExample(doc):
        labels = np.array(doc.user_data['labels'])
        sents = np.array(doc.user_data['sents'], dtype=object)
        return sents, labels

    def _genHDF5s(self, batchIdx, startIdx, stopIdx, prnt_interval=10):
        # Create directory for processed audio files
        if not self.hdf5Path.exists():
            print('Adding the hdf5 data directory')
            self.hdf5Path.mkdir()

        num_files = math.floor(len(self.docs)/self.ex_per_file)
        print('There are {} examples in batch {}'.format(len(self.docs), batchIdx))
        print('Creating {} hdf5 files...'.format(num_files))

        for i in range(num_files):
            fname = str(self.hdf5Path.joinpath('batch'+str(batchIdx)+'_'+str(i)).with_suffix('.hdf5'))
            with h5py.File(str(fname), 'w') as f:
                for j in range(self.ex_per_file):
                    docIdx = i*self.ex_per_file + j
                    sents, labels = self.genSingleExample(self.docs[docIdx])
                    if len(labels)>0:
                        group = f.create_group(str(j))

                        group.create_dataset(name='sents', data=sents, dtype=h5py.special_dtype(vlen=str))
                        group.create_dataset(name='labels', data=labels)

            print('Generated {} files in batch {}...'.format(i+1, batchIdx))