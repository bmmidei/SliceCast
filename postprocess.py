# -*- coding: utf-8 -*-
import numpy as np
import re
from gensim.summarization import summarize, keywords

def createSummaries(doc, labels):
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

def getTimeStamps(doc, docJSON, labels):
    keywords = findKeywords(doc, labels)
    timestamps = findTimeStamps(docJSON, keywords)
    return timeStamps

def findTimeStamps(docJSON, keywords):
    timestamps = np.empty(shape=len(keywords), dtype=str)
    k=0
    data = json.load(docJSON)
    for i in data['response']['results']:
        for j in i['alternatives']:
            for idx, o in enumerate(j['words']):
                if(o['word'] == keywords[k][0] and j['words'][idx+1]['word'] == keywords[k][1]):
                    timestamps[k] = o['startTime']
                    k = k+1
    return timestamps

def findKeywords(doc, labels):
    keywords = np.empty(shape=labels.count(1), dtype=(str,str))
    k = 0
    for i, sent in enumerate(doc):
        if labels[i]:
            keywords[k] = (re.sub("[^\w]", " ",  sent).split()[0], re.sub("[^\w]", " ",  sent).split()[1])
            k = k+1
    return keywords
