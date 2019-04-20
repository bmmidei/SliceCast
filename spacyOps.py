import re
import spacy

MIN_SENT = 5

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

    # Index of the start of the current segment
    startIdx = 0
    for i, sent in enumerate(sents):
        # Search for the split line and label it -1
        if re.search('========,[0-9]+,.+\.', sent):
            if i-startIdx <= MIN_SENT:
                # Label all sentences in last segment for removal
                for j in range(startIdx, i):
                    labels[j] = -1

            # label split as -1 and next sentence as 1 for the start of new seg
            labels[i] = -1
            labels[i+1] = 1

            # move the starting index of the segment to i
            startIdx = i
        
        # Handle last segment in document
        if i+1==numSents:
            if i-startIdx <= MIN_SENT:
                # Label all sentences in last segment for removal
                for j in range(startIdx, i+1):
                    labels[j] = -1

    # Remove split lines/short segments and corresponding labels
    sents = [x for i,x in enumerate(sents) if labels[i]!=-1]
    labels = [x for x in labels if x!=-1]

    data = {'sents':sents,
            'labels':labels}
    doc.user_data = data
    
    return doc

def edaLabeler(doc):
    """Custome SliceCast pipeline component to add to spacy pipeline.
    In contrast with the "customLabeler" component, this component does not
    remove short segments. It is intended for exploratory data analysis only
    """
    sents = [sent.text.strip() for sent in doc.sents] # Remove whitespace
    sents = [sent for sent in sents if sent] # Remove empty strings

    numSents = len(sents)
    labels = [0] * numSents

    # Index of the start of the current segment
    startIdx = 0
    for i, sent in enumerate(sents):
        # Search for the split line and label it -1
        if re.search('========,[0-9]+,.+\.', sent):
            # label split as -1 and next sentence as 1 for the start of new seg
            labels[i] = -1
            labels[i+1] = 1

            # move the starting index of the segment to i
            startIdx = i
        
        # Handle last segment in document
        if i+1==numSents:
            if i-startIdx <= MIN_SENT:
                # Label all sentences in last segment for removal
                for j in range(startIdx, i+1):
                    labels[j] = -1

    # Remove split lines/short segments and corresponding labels
    sents = [x for i,x in enumerate(sents) if labels[i]!=-1]
    labels = [x for x in labels if x!=-1]

    data = {'sents':sents,
            'labels':labels}
    doc.user_data = data
    
    return doc
