import sys
import os 
sys.path.insert(0, '../project1/props')
from calc import Calculate as calc
import autograd.numpy as np 
from autograd import grad
from numpy import random
import matplotlib.pyplot as plt 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor 
import seaborn as sns 
from alive_progress import alive_bar
from itertools import product

sns.set_theme()
warnings.filterwarnings('ignore')
random.seed(2023)

class NeuralNetwork:

    def __init__(self, input_size, hidden_layers=1, hidden_neurons=100,
                 output_size=1, activation='sigmoid', epochs=int(1e4),
                 batch_size=15,  eta=1e-5, alpha=1e-2,
                 random_weights=False, cost='mse', **kwargs):
        
        self.input_size = input_size
        self.n_layers = hidden_layers
        self.n_neurons = hidden_neurons
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.alpha = alpha
        self.kwargs = kwargs

        self.set_activation(activation)
        self.create_weights_and_bias(activation, random_weights)
        self.set_cost_function(cost)

        self.optimal_layers = None

    def set_activation(self, activation):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
    
        def sigmoid_derivative(x):
            return x * (1 - x)
        
        def ReLU(x):
            f = np.where(x > 0, x, 0.0)
            return f
        
        def ReLU_derivative(x):
            f = np.where(x > 0, 1.0, 0.0)
            return f
        
        def leaky_ReLU(x):
            f = np.where(x > 0, x, 1e-4 * x)
            return f
        
        def leaky_ReLU_derivative(x):
            f = np.where(x > 0, 1.0, 1e-4)
            return f

        def tanh(x):
            return np.tanh(x)
        
        def tanh_derivative(x):
            return 1 - np.tanh(x)**2
        
        activations = {'sigmoid': (sigmoid, sigmoid_derivative),
                    'relu': (ReLU, ReLU_derivative),
                    'lrelu': (leaky_ReLU, leaky_ReLU_derivative),
                    'tanh': (tanh, tanh_derivative)}
        
        msg = f'Valid activation functions are {list(activations.keys())}'
        assert activation in activations, msg

        self.activation, self.activation_diff = activations[activation]

    def create_weights_and_bias(self, activation, random_weights):
        def Xavier_weights(size_in, size_out):
            r'''
            Xavier/Glorot weights initialization. Helps aviod the 
            vanishing gradient problem. Suited for sigmoid and 
            hyperbolic activation functions.
            '''
            std = np.sqrt(2 / (size_in + size_out))
            weights = random.normal(0, std, size=(size_in, size_out))

            return weights
        
        def He_weights(size_in, size_out):
            r'''
            He weights initialization. Helps avoid dead nodes and 
            exploding gradients in deep networks. Suited for ReLU 
            activation functions.
            '''
            std = np.sqrt(2 / size_in)
            weights = random.normal(0, std, size=(size_in, size_out))

            return weights 
        
        def LeCun_weights(size_in, size_out):
            r'''
            LeCun weights initialization. Good for activation functions 
            with varying slopes. Suited for Leaky ReLU activation 
            functions.
            '''
            std = np.sqrt(1 / size_in)
            weights = random.normal(0, std, size=(size_in, size_out))

            return weights
        
        def rand_weights(size_in, size_out):
            r'''
            Randomized weights initialization. 
            '''
            return random.randn(size_in, size_out)
        
        weight_funcs = {'sigmoid': Xavier_weights,
                       'tanh': Xavier_weights,
                       'relu': He_weights,
                       'lrelu': LeCun_weights}
        
        if random_weights:
            init_weights = rand_weights
        
        else:
            init_weights = weight_funcs[activation]

        self.hidden_layers = []

        input_size = self.input_size

        for _ in range(self.n_layers):
            weights = init_weights(input_size, self.n_neurons)
            bias = np.zeros((self.n_neurons))

            self.hidden_layers.append([weights, bias])
            input_size = self.n_neurons

        self.hidden_layers.append([init_weights(self.n_neurons, 1),
                                   np.zeros((self.output_size))])

    def set_cost_function(self, cost):
        def cost_ols(x):
            return x - self.y_data
        
        if cost == 'mse':
            self.loss = cost_ols

    def feed_forward(self):
        self.z_h = []
        self.a_h = []

        # Input data is the input layer
        input_layer = self.X_data

        # Feed forward for the hidden layers
        for i in range(self.n_layers):
            weights = self.hidden_layers[i][0]
            bias = self.hidden_layers[i][1]

            z_h = (input_layer @ weights) + bias
            a_h = self.activation(z_h)

            self.z_h.append(z_h)
            self.a_h.append(a_h)

            # Update the input layer
            input_layer = a_h

        # Feed forward to the output layer
        self.z_o = (input_layer @ self.hidden_layers[-1][0]) \
                 + self.hidden_layers[-1][1]
        
        # self.z_h.append(self.z_o)
        # self.a_h.append(self.z_o)

    def predict(self, X):

        input_layer = X

        for i in range(self.n_layers):
            weights = self.hidden_layers[i][0]
            bias = self.hidden_layers[i][1]

            z_h = (input_layer @ weights) + bias
            a_h = self.activation(z_h)
            input_layer = a_h 
        
        z_o = (input_layer @ self.hidden_layers[-1][0]) \
            + self.hidden_layers[-1][1]
        
        return z_o
    
    def backpropagation(self):
        loss = self.loss(self.z_o)
        delta_o = loss # Add derivative of activation of final layer
        delta_h = delta_o

        for i in range(self.n_layers - 1, -1, -1):
            if i == self.n_layers - 1:
                layer = self.hidden_layers[i+1]
                weights = layer[0]
                delta_h = delta_o @ weights.T
            
            else: 
                layer = self.hidden_layers[i]
                weights = layer[0]
                delta_h = delta_h @ weights.T \
                        * self.activation_diff(self.a_h[i])
            
            if i == 0:
                input_layer = self.X_data
            
            else: 
                input_layer = self.a_h[i-1]
        
            w_grad = input_layer.T @ delta_h
            b_grad = np.sum(delta_h, axis=0)

            if self.alpha > 0.0:
                w_grad += weights * self.alpha
            
            self.hidden_layers[i][0] -= self.eta * w_grad
            self.hidden_layers[i][1] -= self.eta * b_grad
    
    def fit(self, X, y, X_val, y_val, patience=500, centered=False):
        if not centered:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)

            self.X_data_full = (X - mean) / std

            has_validation = False
            if not X_val is None and not y_val is None:
                mean_val = np.mean(X_val, axis=0)
                std_val = np.mean(X_val, axis=0)

                self.X_val = (X_val - mean_val) / std_val
                self.y_val = y_val
                has_validation = True
        
        else:
            self.X_data_full = X 
            if not X_val is None and not y_val is None:
                self.X_val = X_val 
                self.y_val = y_val 
                has_validation = True
        
        self.y_data_full = y

        n = len(self.X_data_full)
        indices = np.arange(n)
        iterations = int(n // self.batch_size)
        best_mse = np.inf
        counter = 0

        for i in range(self.epochs):
            for j in range(iterations):
                rand_inds = random.choice(indices, size=self.batch_size,
                                          replace=False)
                
                self.X_data = self.X_data_full[rand_inds]
                self.y_data = self.y_data_full[rand_inds]

                self.feed_forward()
                self.backpropagation()

                if has_validation:
                    ypred = self.predict(self.X_val)
                    mse = np.mean((self.y_val - ypred)**2)
                    # print(mse)
                else:
                    mse = np.mean((self.y_data - self.z_o)**2)
                
                if mse < best_mse:
                    best_mse = mse
                    self.optimal_layers = self.hidden_layers
                    counter = 0
                
                else: 
                    counter += 1

                if counter == patience:
                    self.hidden_layers = self.optimal_layers

                    print(f'Early stopping at epoch {i} with MSE: {best_mse}')
                    break

        self.hidden_layers = self.optimal_layers


def test_func(x):
    a_0 = 1
    a_1 = 2
    a_2 = -5
    a_3 = 3
    f = a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3
    # f = 2 * np.sin(2 * x) + - 0.5 * np.cos(3 * x) + 0.3 * x**3

    return f

n = int(1e2)
x = np.linspace(-4, 4, n)[:, np.newaxis]
y_true = test_func(x)
y = y_true + random.normal(0, 1, x.shape)

X = calc.create_X(x, poly_deg=1, include_ones=False)

def min_max(data, low=-1, high=1):
    norm = (high - low) * (data - np.min(data)) \
         / (np.max(data) - np.min(data)) + high
    
    return norm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

NN = NeuralNetwork(1, hidden_layers=1, hidden_neurons=105, alpha=0, eta=1e-6, batch_size=25)
NN.fit(X_train, y_train, X_test, y_test)
ypred = NN.predict(X)
mse = np.mean((y - ypred)**2)
print(mse)

fig, ax = plt.subplots()
ax.scatter(x, y, color='black', s=3, label='Data', alpha=0.5)
ax.plot(x, ypred, color='red', ls='dashdot', label='FNN')
ax.plot(x, y_true, color='black', ls='dotted', label='True')
# ax.plot(x, ypred_scikit, color='blue', label='MPLSciKit')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()
plt.show()