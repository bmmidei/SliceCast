import numpy as np
import h5py
from netUtils import getTestSet
import keras

class pkHistory(keras.callbacks.Callback):
    def __init__(self, test_file, num_samples, k=10):
        self.test_file = test_file
        self.num_samples = num_samples
        self.k = k
        
    def on_train_begin(self, logs={}):
        self.pk = []
 
    def on_train_end(self, logs={}):
        return self.pk
 
    def on_epoch_end(self, epoch, logs={}):
        X_test, y_test = getTestSet(self.test_file, self.num_samples)
        preds = self.model.predict(X_test)
        score = pkmetric(y_test, preds, k=self.k)
        self.pk.append(score)
        print('PK Score for epoch {} is {:0.4f}'.format(epoch+1, score))
        

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
    return 1-(correct/(num_windows))
