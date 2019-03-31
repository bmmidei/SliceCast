from __future__ import unicode_literals
from spacyOps import customLabeler, createSpacyPipe
from pathlib import Path
import spacy
import tensorflow as tf
import time, math
import h5py
import numpy as np

class Pipeline(object):

    def __init__(self, dataPath):
        self.mainPath = Path(dataPath)
        self.trainPath = Path(dataPath).joinpath('train', 'train')
        self.testPath = Path(dataPath).joinpath('train', 'test')
        self.devPath = Path(dataPath).joinpath('train', 'dev')
        self.expPath = Path(dataPath).joinpath('test')
        self.hdf5Path = Path(dataPath).joinpath('hdf5')

        self.nlp = createSpacyPipe()

        print('\nUsing spacy pipeline components:')
        for name, proc in self.nlp.pipeline:
            print(name, proc)

    def processDirectory(self, dataPath, maxExamples=None):
        self.getFilePaths(dataPath)
        print('\nThere are {} documents in this directory'.format(self.numExamples))
        print('Processing a subset of size {}...'.format(maxExamples))

        if maxExamples:
            self.numExamples = min(maxExamples, self.numExamples)

        docs = []
        for file in self.files[:self.numExamples]:
            try:
                doc = self.nlp(file.read_text(encoding='utf-8'))
                if len(doc.user_data['labels']) != 0:
                    docs.append(doc)
                else:
                    print('Ignoring file due to short segments: ' + str(file))
            except:
                print('Ignoring file due to invalid type: ' + str(file))

        # Update number of examples
        # (We have dropped some because they have no significant segments)
        self.numExamples = len(docs)
        self.docs = docs

    def getFilePaths(self, dataPath):
        self.files = [x for x in dataPath.glob('**/*') if x.is_file()]
        self.numExamples = len(self.files)

    @staticmethod
    def genSingleExample(doc):
        labels = np.array(doc.user_data['labels'])
        sents = np.array(doc.user_data['sents'], dtype=object)
        return sents, labels

    def genHDF5s(self, exPerFile=256, prntInterval=10):
        # Create directory for processed audio files
        if not self.hdf5Path.exists():
            print('Adding the hdf5 data directory')
            self.hdf5Path.mkdir()

        numFiles = math.floor(self.numExamples/exPerFile)
        print('There are {} examples in the directory'.format(self.numExamples))
        print('Creating {} hdf5 files...'.format(numFiles))

        for i in range(numFiles):
            fname = str(self.hdf5Path.joinpath(str(i)).with_suffix('.hdf5'))
            with h5py.File(str(fname), 'w') as f:
                for j in range(exPerFile):
                    docIdx = i*exPerFile + j
                    sents, labels = self.genSingleExample(self.docs[docIdx])
                    group = f.create_group(str(j))

                    group.create_dataset(name='sents', data=sents, dtype=h5py.special_dtype(vlen=str))
                    group.create_dataset(name='labels', data=labels)

            if i%prntInterval==0:
                print('Generated {} files...'.format(i))