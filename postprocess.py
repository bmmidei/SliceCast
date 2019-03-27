import numpy as np
from gensim.summarization import summarize

def createSummaries(doc, labels):
    summaries = np.empty(shape = labels.count(1), dtype=string)
    segment = ''
    k = 0
    for i, sent in enumerate(doc):
        if labels[i] == 1 and segment != '':
            summaries[k] = summerize(segment)
            k = k+1
            segment = sent
        else:
            segment = segment + sent
    summaries[k] = summerize(segment)
    return summaries
