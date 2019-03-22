from __future__ import unicode_literals
from spacyOps import customLabeler
from pathlib import Path
import spacy
import tensorflow as tf
import time
nlp = spacy.load('en')
nlp.remove_pipe('ner')
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')
#tokenizer = nlp.create_pipe('tokenizer')
sentencizer = nlp.create_pipe('sentencizer')
#nlp.add_pipe(tokenizer)
nlp.add_pipe(sentencizer)
nlp.add_pipe(customLabeler)

class Pipeline(object):

    def __init__(self, dataPath):
        self.mainPath = Path(dataPath)
        self.trainPath = Path(dataPath).joinpath('train', 'train')
        self.testPath = Path(dataPath).joinpath('train', 'test')
        self.devPath = Path(dataPath).joinpath('train', 'dev')
        self.expPath = Path(dataPath).joinpath('test')
        for name, proc in nlp.pipeline:
            print(name, proc)

    def processDirectory(self, dataPath):
        files = self.getFilePaths(dataPath)
        self.docs = [nlp(file.read_text(encoding='utf-8')) for file in files]

    def getFilePaths(self, dataPath):
        return [x for x in dataPath.glob('**/*') if x.is_file()]

    def genSingleRecord(self, fpath, doc, lemma=True):

        # Helper functions for creating tfrecords
        def _bytes_feature(value):
            encoded = [x.encode('utf8') for x in value]
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded))
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        if not fpath.exists():
            try:
                # Initialize tensorflow writer
                #writer = tf.data.experimental.TFRecordWriter(str(fpath)) # tf 2.x version
                writer = tf.io.TFRecordWriter(str(fpath)) # tf 1.x version

                # Create features of example from spacy Document object
                feature = {}
                for i, sent in enumerate(doc.user_data['sents']):
                    if lemma:
                        feature['sent_'+str(i)] = _bytes_feature([x.lemma_ for x in sent])
                    else:
                        feature['sent_'+str(i)] = _bytes_feature([x.lower_ for x in sent])
                feature['labels'] = _int64_feature(doc.user_data['labels'])

                # Construct the Example proto object
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                
                # Serialize the example to a string
                serialized = example.SerializeToString()

                # write the serialized object to the disk
                writer.write(serialized)
                writer.close()

            except Exception as e:
                print('Error creating tf record for ' + str(fpath))
                print(e)

def main(dataPath):
    pipe = Pipeline(dataPath=dataPath)

    pipe.processDirectory(pipe.expPath)

if __name__ == '__main__':
    mainPath = '/Users/bmmidei/Projects/SliceCast/data'

    main(dataPath=mainPath)