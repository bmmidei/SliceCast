import re
import spacy


def createSpacyPipe():
    # Load baseline Spacy pipeline
    nlp = spacy.load('en')

    # Remove unneeded elements
    nlp.remove_pipe('ner')
    nlp.remove_pipe('tagger')
    nlp.remove_pipe('parser')

    # Add sentancizer and custom labeler
    sentencizer = nlp.create_pipe('sentencizer')
    nlp.add_pipe(sentencizer)
    nlp.add_pipe(customLabeler)

    return nlp

def customLabeler(doc):
    # Custom slicecast pipeline component to add to spacy pipeline
    sents = [sent.text.strip() for sent in doc.sents] # Remove whitespace
    sents = [sent for sent in sents if sent] # Remove empty strings

    numSents = len(sents)
    labels = [0] * numSents

    for i, sent in enumerate(sents):
        # Search for the split line and label it -1
        if re.search('========,[0-9]+,.+\.', sent):
            labels[i] = -1
            labels[i+1] = 1

    # Remove split lines and corresponding labels
    sents = [x for i,x in enumerate(sents) if labels[i]!=-1]
    labels = [x for x in labels if x!=-1]

    data = {'sents':sents,
            'labels':labels}
    doc.user_data = data
    
    return doc
