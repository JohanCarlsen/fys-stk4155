import autograd.numpy as np

class CostFunctions:
    def __init__(self):
        pass 

    def loss(self, y_true, y_pred):
        raise NotImplementedError
    
class MeanSquaredError(CostFunctions):
    def loss(y_true, y_pred):
        n = y_true.shape[0]

        return (1 / n) * np.sum((y_pred - y_true)**2)

class CrossEntropy(CostFunctions):
    def loss(y_true, y_pred):
        delta = 1e-9
        n = y_true.size

        return -(1.0 / n) * np.sum(y_true * np.log10(y_pred + delta))

class LogLoss(CostFunctions):
    def loss(y_true, y_pred):
        delta = 1e-9
        n = y_true.shape[0]

        return -(1.0 / n) * np.sum((y_true * np.log(y_pred + delta) + \
                                    ((1 - y_pred) * np.log(1 - y_pred + delta))))
