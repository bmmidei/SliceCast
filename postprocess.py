# -*- coding: utf-8 -*-
import numpy as np
from gensim.summarization import summarize, keywords

text = "While most people associate me, I suspect, with thinking about longevity, we probably don’t spend enough time talking about what longevity means. " + \
"But the way I talk about it with my patience is, it’s both enhancing lifespan but also health span. " + \
"Lifespan is the easier of those two to understand, because enhancing lifespan just means not dying, which is not to say that that’s easy, but it’s conceptually easy. " + \
"I think the health span stuff is harder to understand and as I have come to learn in the past three or four years, I believe for most people it actually matters more. " + \
"Many people think if you helping me doesn’t add one day to the length of my life, but improves the quality of my life, especially at the end, that would be sufficient. " + \
"So in many ways what I want to talk about today is one piece of health span that I know the least about by far, but also I think is the one that we are least likely to talk about as a society, which is mental health. " + \
"Now you’ve spoken really publicly about your interest in that. " + \
"I knew a lot of this before you talked about it at TED, but can you tell me a little bit about that? " + \
"I can, and I’m thrilled you’re doing a podcast because I do think just as a bit of overlay on what you said, there is so much focus on extending lifespan, rightly so, but the equal level of obsession that you bring to performance and health span, I think creates a compelling combination that I don’t find in many places. " + \
"I find the combination of those interests very common, but the combination of competencies broadly speaking in both of those domains is very uncommon."

print(summarize(text, ratio=0.15))
print(keywords(text, words = 3, lemmatize=True))
summaries = np.empty(shape=(23,), dtype=(str, []))

def createSummaries(doc, labels):
    summaries = np.empty(shape=labels.count(1), dtype=(str, []))
    segment = ''
    numSent = 0
    k = 0
    for i, sent in enumerate(doc):
        if labels[i] == 1 and segment != '':
            summaries[k] = (summarize(segment, ratio=1/numSent), keywords(segment, words = 3, lemmatize=True, split=True))
            k = k+1
            segment = sent
            numSent = 1
        else:
            segment = segment + " " + sent
            numSent = numSent + 1
    summaries[k] = (summarize(segment, ratio=1/numSent), keywords(segment, words = 5, lemmatize=True, split=True))
    return summaries
