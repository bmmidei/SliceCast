# -*- coding: utf-8 -*-
import numpy as np
import re
import objectpath
import json
from gensim.summarization import summarize, keywords

def getSummaries(doc, labels):
    summaries = np.empty(shape=labels.count(1), dtype=(str, []))
    segment = ''
    numSent = 0
    k = 0
    for i, sent in enumerate(doc):
        if labels[i] and segment != '':
            summaries[k] = (summarize(segment, ratio=1/numSent), keywords(segment, words = 3, lemmatize=True, split=True))
            k = k+1
            segment = sent
            numSent = 1
        else:
            segment = segment + " " + sent
            numSent = numSent + 1
    summaries[k] = (summarize(segment, ratio=1/numSent), keywords(segment, words = 3, lemmatize=True, split=True))
    return summaries

def getTimeStamps(doc, pathJSON, labels):
    keywords = findKeywords(doc, labels)
    timestamps = findTimeStamps(pathJSON, keywords)
    return timeStamps

def findKeywords(doc, labels):
    keywords = []
    for i, sent in enumerate(doc):
        if labels[i]:
            keywords.append((re.sub("[^\w]", " ",  sent).split()[0], re.sub("[^\w]", " ",  sent).split()[1]))
    return keywords

def findTimeStamps(pathJSON, keywords):
    with open (pathJSON) as f:
        data = json.load(f)
    jsonnn_tree = objectpath.Tree(data['response'])
    words = tuple(jsonnn_tree.execute('$..words'))
    k = 0
    timestamps = []

    for idx, word in enumerate(words):
        if word['word'] == keywords[k][0] and idx < len(result_tuple) - 1 and result_tuple[idx + 1]['word'] == keywords[k][1]:
            timestamps.append(word['startTime'])
            k = k + 1
            if k == len(keywords):
                break
    return timestamps
