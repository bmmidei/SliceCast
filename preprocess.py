from __future__ import unicode_literals
#from spacyOps import Collection
from pathlib import Path
import spacy
import time
nlp = spacy.load('en')
nlp.remove_pipe('ner')
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')
#tokenizer = nlp.create_pipe('tokenizer')
sentencizer = nlp.create_pipe('sentencizer')
#nlp.add_pipe(tokenizer)
nlp.add_pipe(sentencizer)

class Pipeline(object):

    def __init__(self, dataPath):
        self.trainPath = Path(dataPath).joinpath('train', 'train')
        self.testPath = Path(dataPath).joinpath('train', 'test')
        self.devPath = Path(dataPath).joinpath('train', 'dev')
        self.expPath = Path(dataPath).joinpath('test')
        for name, proc in nlp.pipeline:
            print(name, proc)

    def getFilePaths(self, dataPath):
        return [x for x in dataPath.glob('**/*') if x.is_file()]

    def processDirectory(self, dataPath):
        files = self.getFilePaths(dataPath)
        self.docs = [nlp(file.read_text(encoding='utf-8')) for file in files]

def main(dataPath):
    pipe = Pipeline(dataPath=dataPath)

    pipe.processDirectory(pipe.expPath)

if __name__ == '__main__':
    mainPath = '/Users/bmmidei/Projects/SliceCast/data'

    main(dataPath=mainPath)