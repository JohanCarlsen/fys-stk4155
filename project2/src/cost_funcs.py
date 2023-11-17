import autograd.numpy as np

class CostFunctions:
    r'''
    Parent class for cost functions. Should not be used directly.
    '''
    def __init__(self):
        pass 

    def loss(self, y_true, y_pred):
        raise NotImplementedError
    
class MeanSquaredError(CostFunctions):
    r'''
    Mean squared error (MSE) cost function.

    .. math:: C(\beta)=\frac{1}{n}\sum_n(\tilde y_n-y_n)^2
    '''
    @staticmethod
    def loss(y_true, y_pred):
        r'''
        Return the MSE.

        Parameters
        ----------
        y_true : array_like
            Target values.

        y_pred : array_like
            Predicted values.

        Returns
        -------
        array_like : 
            MSE between target and predicted values.
        '''
        n = y_true.shape[0]

        return (1 / n) * np.sum((y_pred - y_true)**2)

class CrossEntropy(CostFunctions):
    r'''
    Cross entropy cost function.

    .. math:: C(\beta)=-\frac{1}{n}\sum_n\left[y_n\log_{10}(\tilde y+\delta)\right]
    '''
    @staticmethod
    def loss(y_true, y_pred):
        r'''
        Return the cross entropy.

        Parameters
        ----------
        y_true : array_like
            Target values.

        y_pred : array_like
            Predicted values.

        Returns
        -------
        array_like : 
            Cross entropy between target and predicted values.
        '''
        delta = 1e-9
        n = y_true.size

        return -(1.0 / n) * np.sum(y_true * np.log10(y_pred + delta))

class LogLoss(CostFunctions):
    r'''
    Log-loss cost function.

    .. math::
        C(\beta)=-\frac{1}{n}\sum_n\left[y_n\log(\tilde y_n+\delta)+(1-\tilde y_n)\log(1-\tilde y_n+\delta)\right]
    '''
    @staticmethod
    def loss(y_true, y_pred):
        r'''
        Return the log-loss.

        Parameters
        ----------
        y_true : array_like
            Target values.

        y_pred : array_like
            Predicted values.

        Returns
        -------
        array_like : 
            Log-loss between target and predicted values.
        '''
        delta = 1e-9
        n = y_true.shape[0]

        return -(1.0 / n) * np.sum((y_true * np.log(y_pred + delta) + \
                                    ((1 - y_pred) * np.log(1 - y_pred + delta))))
