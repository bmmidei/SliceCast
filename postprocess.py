import numpy as np
import h5py

def pkmetric(ytrue, ypred, k=10):
    """Calculate the pk score for a minibatch.
    Args:
        labels: OneHot encoded array of labels for a minibatch of examples
                shape = [batch_size, doc_length, num_classes]
        preds: Softmaxed array of predictions for a minibatch of examples
                shape = [batch_size, doc_length, num_classes]
    Yields:
        pkscore: Average pkscore for the minibatch
    """
    ytrue = np.argmax(ytrue, axis=-1)
    ypred = np.argmax(ypred, axis=-1)
    scores = [pkscore(x,y,k) for x,y in zip(ytrue, ypred)]
    score = np.mean(scores)
    return score

def pkscore(labels, preds, k=10):
    """Calculate the pk score for a single document.
    Score is calculated by moving a sliding window across the predictions
    and labels. A correct score is recorded if the labels and predictions
    are in agreement that there is or is not a segment change.

    eg.     sliding window:    |-------| -->
        Labels: 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 
        Preds:  1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 
    The example above would return an incorrect score because the labels
    and predictions disagree. 

    Args:
        labels: 1-D array of labels. May include padding.
        preds: 1-D array of predictions.
    Yields:
        pkscore: pkscore for this given document
    """
    # Remove padding from labels and preds
    mask = np.where(labels<=1, True, False)
    labels = labels[mask]
    preds = preds[mask]

    num_windows = len(labels) - k + 1
    assert num_windows>0, 'Choose a smaller k value'

    correct = 0
    for i in range(num_windows):
        # calculate index of window close
        j = i + k

        # Get number of segment splits in labels and preds
        label_diff = sum(labels[i:j])
        pred_diff = sum(preds[i:j])

        # Check for agreement between labels and preds
        if (label_diff and pred_diff) or (not label_diff and not pred_diff):
            correct += 1
    return correct/(num_windows)
