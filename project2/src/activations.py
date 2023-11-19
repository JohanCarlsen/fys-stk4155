import autograd.numpy as np 

class Activations:
    r'''
    Parent class for the activation functions. Should not be used
    directly.
    '''
    def __init__(self):
        pass

    def function(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

class Linear(Activations):
    r'''
    Linear activation function.

    .. math:: f(x)=x
    '''
    @staticmethod
    def function(x):
        r'''
        Activation function.

        Parameters
        ----------
        x : array_like
            Input variable.

        Returns
        -------
        x : array_like
            Linear relation.
        '''
        return x
    
    @staticmethod
    def derivative(x):
        r'''
        Derivative of the activation function.

        .. math:: f'(x)=1

        Parameters
        ----------
        x : array_like
            Input variable.

        Returns
        -------
        array_like : 
            Returns an array of the same shape as ``x`` filled with ones.
        '''
        return np.ones(x.shape)

class Sigmoid(Activations):
    r'''
    Sigmoid activation function.

    .. math:: \sigma(x)=\frac{1}{1+e^{-x}}
    '''
    @staticmethod
    def function(x):
        r'''
        Activation function.

        Parameters
        ----------
        x : array_like
            Input variable.

        Returns
        -------
        array_like : 
            The Sigmoid function :math:`\sigma(x)`.
        '''
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        r'''
        Derivative of activation function.

        .. math:: \sigma'(x)=\sigma(x)[1-\sigma(x)]

        Parameters
        ----------
        x : array_like
            Input variable.

        Returns
        -------
        array_like :
            The derivative of the Sigmoid function.
        '''
        return x * (1 - x)

class ReLU(Activations):
    r'''
    Rectified Linear Unit activation function.
    '''
    @staticmethod
    def function(x):
        r'''
        Activation function.

        Parameters
        ----------
        x : array_like
            Input variable.

        Returns
        -------
        array_like :
            The maximum value of ``x`` and 0.
        '''
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        r'''
        Derivative of the ReLU activation function.

        Parameters
        ----------
        x : array_like
            Input variable.

        Returns
        -------
        array_like :
            1 if ``x > 0``, else 0.
        '''
        return np.where(x > 0, 1, 0)

class LeakyReLU(Activations):
    @staticmethod
    def function(x):
        alpha = 0.01  
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def derivative(x):
        alpha = 0.01
        return np.where(x > 0, 1, alpha)

class Softmax(Activations):
    @staticmethod
    def function(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def derivative(x):
        raise NotImplementedError

class WeightInitializers:
    def __init__(self):
        pass

    def initialize(self, size_in, size_out):
        raise NotImplementedError

class XavierInitializer(WeightInitializers):
    @staticmethod
    def initialize(size_in, size_out):
        std = np.sqrt(2 / (size_in + size_out))
        weights = np.random.normal(0, std, size=(size_in, size_out))
        return weights

class HeInitializer(WeightInitializers):
    @staticmethod
    def initialize(size_in, size_out):
        std = np.sqrt(2 / size_in)
        weights = np.random.normal(0, std, size=(size_in, size_out))
        return weights

class LeCunInitializer(WeightInitializers):
    @staticmethod
    def initialize(size_in, size_out):
        std = np.sqrt(1 / size_in)
        weights = np.random.normal(0, std, size=(size_in, size_out))
        return weights

class RandomInitializer(WeightInitializers):
    @staticmethod
    def initialize(size_in, size_out):
        return np.random.randn(size_in, size_out)
