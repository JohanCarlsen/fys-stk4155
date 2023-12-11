import numpy as np
from abc import ABC, abstractmethod

class Metrics:
    r'''
    Class for calculating the accuracy, precision, and recall between 
    predicted and true labels.

    .. math::
        \begin{align}
        \mathrm{Accuracy}&=\mathrm{\frac{Correct\,predictions}{All\,predictions}}\\
        \mathrm{Precision}&=\mathrm{\frac{TP}{TP+FP}}\\
        \mathrm{Recall}&=\mathrm{\frac{TP}{TP+FN}}
        \end{align}

    where :math:`TP,FP,FN` are true positive, false positive, and false
    negative, respectively.

    Parameters
    ----------
    y_true : array_like
        True labels.

    y_pred : array_like
        Predicted labels.

    classes : array_like
        Class or classes to evaluate the metrics against.

    Attributes
    ----------
    accuracy : array_like
        Accuracy score.

    precision : ndarray
        Precision score.

    recall : ndarray
        Recall score.

    mean_precision : array_like
        Mean precision. 

    mean_recall : array_like
        Mean recall.
    '''
    def __init__(self, y_true, y_pred, classes):
        self.ytrue = y_true
        self.ypred = y_pred
        self.classes = classes[:, np.newaxis]

        self.accuracy = self._accuracy()
        self.precision = self._precision()
        self.mean_precision = np.mean(self.precision)
        self.recall = self._recall()
        self.mean_recall = np.mean(self.recall)

    def _accuracy(self):
        return np.sum(self.ypred == self.ytrue) / self.ytrue.size
    
    def _precision(self):

        self.TP = np.sum((self.ytrue == self.classes) & \
                         (self.ypred == self.classes), axis=-1)
        
        self.FP = np.sum((self.ytrue != self.classes) & \
                         (self.ypred == self.classes), axis=-1)
        
        TPFP = self.TP + self.FP 
        
        return np.where(TPFP != 0, self.TP / TPFP, 0)
    
    def _recall(self):

        self.FN = np.sum((self.ytrue == self.classes) & \
                         (self.ypred != self.classes), axis=-1)
        
        TPFN = self.TP + self.FN 

        return np.where(TPFN != 0, self.TP / TPFN, 0)
    
    def print_metrics(self):
        r'''
        Print the mean scores. 
        '''
        print('\nMean metrics')
        print('-' * 12)
        print(f'Accuracy: {self.accuracy:10.5f}')
        print(f'Precision: {self.mean_precision:9.5f}')
        print(f'Recall: {self.mean_recall:12.5f}')
    
def confusion_matrix(y_true, y_pred):
    r'''
    Calculate the confusion matrix for true and predicted labels.

    Parameters
    ----------
    y_true : array_like
        True labels. 

    y_pred : array_like
        Predicted labels.

    Returns
    -------
    cont_mat : ndarray
        Confusion matrix.
    '''
    n_classes = np.max([np.max(y_true), np.max(y_pred)]) + 1
    conf_mat = np.zeros((n_classes, n_classes))

    for true, pred in zip(y_true, y_pred):
        conf_mat[true, pred] += 1

    return conf_mat

