from __future__ import unicode_literals
from spacyOps import customLabeler, createSpacyPipe
from pathlib import Path
import spacy
import tensorflow as tf
import time, math
import h5py
import numpy as np

class Pipeline(object):

    def __init__(self, dataPath, ex_per_batch, ex_per_file, min_segs=3, wiki=True):
        """Preprocessing pipeline class to extract and condition training data and labels from 
        text files. To be used with the SliceCast neural network and associated
        batch generator
        Args:
            dataPath: Path to high level data directory
            ex_per_batch: Batching is used because the entire wiki dataset cannot be
                          loaded into memory. Instead it must be processed in batches
                          This parameter determines the number of text files to be
                          processed in a single batch
            ex_per_file: Number of processed documents to place in each hdf5 file
            min_segs: Minumum number of segments in a document for it to be included
                      in the training set.
            wiki: Boolean indicating if the Wiki dataset is being processed
                  If false, then podcast dataset is being processed.
        """
        self.wiki = wiki # We preprocess slightly differently based on wiki or podcast
        self.mainPath = Path(dataPath)
        if wiki:
            self.trainPath = Path(dataPath).joinpath('train')
            self.testPath = Path(dataPath).joinpath('test')
            self.devPath = Path(dataPath).joinpath('dev')

        self.ex_per_batch = ex_per_batch
        self.ex_per_file = ex_per_file
        self.nlp = createSpacyPipe()
        self.min_segs = min_segs
        print('\nUsing spacy pipeline components:')
        for name, proc in self.nlp.pipeline:
            print(name, proc)

    def processDirectory(self, dataPath, max_examples=None):
        """Main Processing function"""
        # Get the file paths for every txt file in the directory
        self.getFilePaths(dataPath)
        self.hdf5Path = Path(dataPath).joinpath('hdf5')

        print('There are {} documents in this directory'.format(self.num_examples))

        if max_examples:
            self.num_examples = min(max_examples, self.num_examples)
            print('Processing a subset of size {}...'.format(self.num_examples))

        # Determine number of batches to process
        self.num_batches = math.ceil(self.num_examples/self.ex_per_batch)
        print('There are {} batches'.format(self.num_batches))

        #Process files in batches to reduce memory impact
        for batchIdx in range(self.num_batches):
            # Initialize docs at beginning of each batch
            docs = []
            self.docs = []
            # Determine start and stop indices in file list for current batch
            startIdx = batchIdx * self.ex_per_batch
            if batchIdx+1 == self.num_batches:
                stopIdx = self.num_examples
            else:
                stopIdx = startIdx + self.ex_per_batch
            # Run spacy nlp on each document
            print(len(self.files))
            for file in self.files[startIdx:stopIdx]:
                try:
                    doc = self.nlp(file.read_text(encoding='utf-8'))
                    if len(doc.user_data['labels']) > 0 and sum(doc.user_data['labels'])>=self.min_segs:
                        docs.append(doc)
                except Exception as e:
                    print(file)
                    print(e)

            self.docs = docs
            
            # Generate HDF5s for current batch
            self._genHDF5s(batchIdx, startIdx, stopIdx)

    def getFilePaths(self, dataPath):
        """Get all text files in a directory, recursively"""
        self.files = [x for x in dataPath.glob('**/*') if x.is_file()]
        print(dataPath)
        self.num_examples = len(self.files)

    @staticmethod
    def genSingleExample(doc):
        """Extract sentences and corresponding labels from spacy
        document object
        """
        labels = np.array(doc.user_data['labels'])
        sents = np.array(doc.user_data['sents'], dtype=object)
        return sents, labels

    def _genHDF5s(self, batchIdx, startIdx, stopIdx, prnt_interval=10):
        """Generate hdf5 files from processed documents"""
        # Create directory for processed audio files

        if not self.hdf5Path.exists():
            print('Adding the hdf5 data directory')
            self.hdf5Path.mkdir()

        num_files = math.floor(len(self.docs)/self.ex_per_file)
        num_files = max(1, num_files) # Ensure generation of at least 1 file
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