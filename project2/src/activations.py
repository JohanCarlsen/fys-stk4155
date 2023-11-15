import autograd.numpy as np 

class Activations:
    def __init__(self):
        pass

    def function(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

class Linear(Activations):
    def function(x):
        return x

    def derivative(x):
        return np.ones(x.shape)

class Sigmoid(Activations):
    def function(x):
        return 1 / (1 + np.exp(-x))

    def derivative(x):
        return x * (1 - x)

class ReLU(Activations):
    def function(x):
        return np.maximum(0, x)

    def derivative(x):
        return np.where(x > 0, 1, 0)

class LeakyReLU(Activations):
    def function(x):
        alpha = 0.01  
        return np.where(x > 0, x, alpha * x)

    def derivative(x):
        alpha = 0.01
        return np.where(x > 0, 1, alpha)

class Softmax(Activations):
    def function(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(x):
        raise NotImplementedError

class WeightInitializers:
    def __init__(self):
        pass

    def initialize(self, size_in, size_out):
        raise NotImplementedError

class XavierInitializer(WeightInitializers):
    def initialize(size_in, size_out):
        std = np.sqrt(2 / (size_in + size_out))
        weights = np.random.normal(0, std, size=(size_in, size_out))
        return weights

class HeInitializer(WeightInitializers):
    def initialize(size_in, size_out):
        std = np.sqrt(2 / size_in)
        weights = np.random.normal(0, std, size=(size_in, size_out))
        return weights

class LeCunInitializer(WeightInitializers):
    def initialize(size_in, size_out):
        std = np.sqrt(1 / size_in)
        weights = np.random.normal(0, std, size=(size_in, size_out))
        return weights

class RandomInitializer(WeightInitializers):
    def initialize(size_in, size_out):
        return np.random.randn(size_in, size_out)
