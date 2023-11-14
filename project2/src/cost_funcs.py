import autograd.numpy as np

class CostFunctions:
    def __init__(self):
        pass 

    def loss(self, y_true, y_pred):
        raise NotImplementedError
    
    def gradient(self, y_true, y_pred):
        raise NotImplementedError
    
class MeanSquaredError(CostFunctions):
    def loss(y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred)**2)
    
    def gradient(y_true, y_pred):
        return y_pred - y_true

class CrossEntropy(CostFunctions):
    def loss(y_true, y_pred):
        delta = 1e-10
        return -np.mean(y_true * np.log(y_pred + delta))
    
    def gradien(y_true, y_pred):
        delta = 1e-10
        return -y_true / (y_pred + delta)
