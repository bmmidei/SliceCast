import argparse
from preprocess import Pipeline
from pathlib import Path
import time

data_path = '/home/bmmidei/SliceCast/data/podcasts/katy'
ex_per_file = 2
ex_per_batch = 1000
max_examples = 10000
wiki=False

s = time.time()

pipe = Pipeline(dataPath=data_path,
                ex_per_batch=ex_per_batch,
                ex_per_file=ex_per_file,
                wiki=wiki)

pipe.processDirectory(pipe.mainPath, max_examples=max_examples)
e = time.time()

print('\nProcessing time: {} seconds'.format(e - s))
