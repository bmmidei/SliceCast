import argparse
from preprocess import Pipeline
from pathlib import Path
import time

parser = argparse.ArgumentParser()

parser.add_argument('-dp', '--data_path', action='store', type=str,
                    help='path to high level data directory')

parser.add_argument('-v', '--verbose', help='modify output verbosity',
                    action = 'store_true')

parser.add_argument('-m', '--max_examples', action='store', default=None, type=int,
                    help='maximum number of examples to consider')

parser.add_argument('-n', '--exPerFile', action='store', default=1000, type=int,
                    help='number of examples to store in each HDF5 file')

args = parser.parse_args()

s = time.time()
try:
    pipe = Pipeline(dataPath=args.data_path,
                    ex_per_batch=10000,
                    ex_per_file=args.exPerFile,
                    wiki=True)
except TypeError:
    print('You must enter a data path to process with the -dp flag')
    print('exiting...')
    raise

pipe.processDirectory(pipe.trainPath, max_examples=args.max_examples)
e = time.time()
if args.verbose:
    print('\nProcessing time: {} seconds'.format(e - s))
