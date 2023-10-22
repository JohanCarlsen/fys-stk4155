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
sns.set_theme()
warnings.filterwarnings('ignore')
random.seed(2023)

class NeuralNetwork:

    def __init__(self, X_data, y_data, X_val=None, y_val=None,
                 n_hidden_neurons=12, n_hidden_layers=3,
                 n_epochs=int(1e3), batch_size=5, learning_rate=0.001,
                 momentum=0.9, lamb=0.0001, activation='sigmoid',
                 centered_data=False, random_weights=False):

        self.validation_set = False
        if not centered_data:
            self.X_data_full = self.center_data(X_data)

            if not X_val is None:
                self.X_val = self.center_data(X_val)
                self.y_val = y_val
                self.validation_set = True
        else: 
            self.X_data_full = X_data 

            if not X_val is None:
                self.X_val = X_val
                self.y_val = y_val
                self.validation_set = True       

        self.y_data_full = y_data

        self.n_inputs, self.n_features = self.X_data_full.shape
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_iterations = self.n_inputs // batch_size
        self.eta = learning_rate
        self.mom = momentum
        self.lamb = lamb
        self.activation = activation

        self.create_biases_and_weights(random_weights)
        self.set_activation()

        # Momentum values 
        self.output_weights_momentum = np.zeros(self.output_weights.shape)
        self.hidden_weights_momentum = [np.zeros(w.shape) for w in self.hidden_weights]
    
    def center_data(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        X_centered = (X - mean) / std

        return X_centered[:, 1:]
        
    def create_biases_and_weights(self, random_weights):
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
            init_weights = weight_funcs[self.activation]

        n_features = self.n_features
        n_hidden_neurons = self.n_hidden_neurons
        n_hidden_layers = self.n_hidden_layers

        hidden_weights = []
        hidden_biases = []

        # Input size of first hidden layer
        input_size = n_features
        
        for _ in range(n_hidden_layers):
            weights = init_weights(input_size, n_hidden_neurons)
            # weights = random.randn(input_size, n_hidden_neurons)
            biases = np.zeros((1, n_hidden_neurons))
            hidden_weights.append(weights)
            hidden_biases.append(biases)

            # Update the input size for the next layer
            input_size = n_hidden_neurons
        
        self.hidden_weights = np.array(hidden_weights)
        self.hidden_biases = np.array(hidden_biases)

        self.output_weights = init_weights(input_size, 1)
        # self.output_weights = random.randn(input_size, 1)
        self.output_biases = np.zeros(1)
    
    def set_activation(self):        

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
            f = np.where(x > 0, x, 0.01 * x)
            return f
        
        def leaky_ReLU_derivative(x):
            f = np.where(x > 0, 1.0, 0.0)
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
        assert self.activation in activations, msg

        self.act_func, self.act_func_diff = activations[self.activation]
    
    def feed_forward(self):
        self.z_h = []
        self.a_h = []

        # X_data is the input layer
        input_layer = self.X_data 

        for i in range(self.n_hidden_layers):
            # Feed forward for hidden layers
            z_h = (input_layer @ self.hidden_weights[i]) \
                + self.hidden_biases[i]
            
            a_h = self.act_func(z_h)

            self.z_h.append(z_h)
            self.a_h.append(a_h)

            # Update input layer
            input_layer = a_h
        
        # Feed forward for the output layer
        self.z_o = (a_h @ self.output_weights) + self.output_biases

    def feed_forward_out(self, X):

        # X is the input layer
        input_layer = X

        for i in range(self.n_hidden_layers):
            z_h = (input_layer @ self.hidden_weights[i]) \
                + self.hidden_biases[i]
            a_h = self.act_func(z_h)

            input_layer = a_h
        
        z_o = (a_h @ self.output_weights) + self.output_biases
        
        return z_o
    
    def predict(self, X=None, centered_data=False):
        if not X is None:
            if not centered_data:
                X = self.center_data(X)
            
            else:
                X = X 
        
        else:
            X = self.X_val

        z_o = self.feed_forward_out(X)

        return z_o

    def backpropagation(self):
        error_output = self.z_o - self.y_data
        error_hidden = None 

        self.output_weights_gradient = self.a_h[-1].T @ error_output
        self.output_biases_gradient = np.sum(error_output, axis=0)
        
        for i in range(self.n_hidden_layers - 1, -1, -1):
            if i == self.n_hidden_layers - 1:
                error_hidden = error_output @ self.output_weights.T \
                             * self.act_func_diff(self.a_h[i])
            
            else:
                error_hidden = error_hidden @ self.hidden_weights[i+1].T \
                             * self.act_func_diff(self.a_h[i])
            
            if i == 0:
                input_layer = self.X_data
            
            else:
                input_layer = self.a_h[i-1]
            
            self.hidden_weights_gradient = input_layer.T @ error_hidden
            self.hidden_biases_gradient = np.sum(error_hidden, axis=0)

            if self.lamb > 0.0:
                self.output_weights_gradient += self.lamb \
                                              * self.output_weights

                self.hidden_weights_gradient += self.lamb \
                                              * self.hidden_weights[i]

            self.output_weights_momentum = self.mom * self.output_weights_momentum \
                                         + self.eta * self.output_weights_gradient

            self.hidden_weights_momentum[i] = self.mom * self.hidden_weights_momentum[i] \
                                            + self.eta * self.hidden_weights_gradient

            self.output_weights -= self.output_weights_momentum
            self.hidden_weights[i] -= self.hidden_weights_momentum[i]
            self.output_biases -= self.eta * self.output_biases_gradient
            self.hidden_biases[i] -= self.eta * self.hidden_biases_gradient

    def train(self, patience=1000):
        indices = np.arange(self.n_inputs)
        best_mse = np.inf
        counter = 0

        for i in range(self.n_epochs):
            for j in range(self.n_iterations):
                rand_inds = random.choice(indices, size=self.batch_size,
                                          replace=False)

                self.X_data = self.X_data_full[rand_inds] 
                self.y_data = self.y_data_full[rand_inds] 

                self.feed_forward()
                self.backpropagation()
            
            if self.validation_set:
                ypred = self.feed_forward_out(self.X_val)
                mse = np.mean((self.y_val - ypred)**2)
            
            else:
                mse = np.mean((self.y_data - self.z_o)**2)

            if mse < best_mse:
                best_mse = mse 
                counter = 0
            
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping at epoch {i} with MSE: {best_mse}')
                break

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

X = calc.create_X(x, poly_deg=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
activation=['sigmoid', 'relu', 'lrelu']
NN = NeuralNetwork(X_train, y_train, X_val=X_test, y_val=y_test,
                n_epochs=int(1e4), batch_size=5, n_hidden_layers=2,
                n_hidden_neurons=15, centered_data=False,
                activation='lrelu', learning_rate=1e-5, lamb=1e-2)
# NN2 = NeuralNetwork(X_train, y_train, X_val=X_test, y_val=y_test,
#                 n_epochs=int(1e4), batch_size=5, n_hidden_layers=2,
#                 n_hidden_neurons=15, centered_data=False,
#                 activation='tanh', learning_rate=1e-4, lamb=1e-3,
#                 random_weights=True)
# fig, axes = plt.subplots(3, 1, figsize=(8,7), sharex=True)
# fig.suptitle('Batch size: 5 | Hidden: 2 | Neurons: 15')
# axes[2].set_xlabel(r'$x$')
# for active, ax in zip(activation, axes):

#     NN = NeuralNetwork(X_train, y_train, X_val=X_test, y_val=y_test,
#                     n_epochs=int(1e4), batch_size=5, n_hidden_layers=2,
#                     n_hidden_neurons=15, centered_data=False,
#                     activation=active, learning_rate=1e-4, lamb=1e-3)
#     NN.train()
#     ypred = NN.predict(X)
#     mse = np.mean((ypred - y)**2)

#     ax.set_title(f'MSE: {mse:.3f}, activation: {active}')
#     ax.scatter(x, y, color='black', s=2, label='Data', alpha=0.5)
#     ax.plot(x, ypred, color='red', ls='dashed', label='FFNN')
#     ax.plot(x, y_true, color='blue', ls='dotted', label='True')
#     ax.set_ylabel(r'$y$')
#     ax.legend()

# fig.tight_layout()

# ypred = NN.predict(X_test)
# print(mse)
NN.train()
# NN2.train()
ypred = NN.predict(X)
mse = np.mean((ypred - y)**2)

fig, ax = plt.subplots()
# ax.set_title(f'FFNN MSE: {mse:.3f}, activation: lrelu')
# ax.set_title('Relative error')
ax.scatter(x, y, color='black', s=3, label='Data', alpha=0.5)
ax.plot(x, ypred, color='red', ls='dashdot', label='FNN')
ax.plot(x, y_true, color='black', ls='dotted', label='True')
# abserr = np.abs(y_true - ypred) / np.abs(y_true)
# ax.plot(x, abserr, color='red', ls='solid', label='FFNN Xavier/Golorot weights')
# ypred = NN2.predict(X)
# abserr = np.abs(y_true - ypred) / np.abs(y_true)
# ax.plot(x, abserr, color='blue', ls='dashed', label='FFNN Random weights')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()
plt.show()