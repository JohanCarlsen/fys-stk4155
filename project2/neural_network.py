import sys
import os 
sys.path.insert(0, '../project1/props')
from calc import Calculate as calc
import autograd.numpy as np 
from autograd import grad
# from numpy import random
import matplotlib.pyplot as plt 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor 
import seaborn as sns 
from alive_progress import alive_bar
from itertools import product
from matplotlib.ticker import FormatStrFormatter

sns.set_theme()
# warnings.filterwarnings('ignore')
np.random.seed(2023)

class NeuralNetwork:

    def __init__(self, input_size, hidden_sizes=[1], output_size=1, 
                 activation='sigmoid', epochs=int(1e4),
                 batch_size=80,  eta=1e-4, alpha=1e-3,
                 random_weights=False, cost='mse',
                 output_activation='linear', **kwargs):
        
        self.input_size = input_size
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.alpha = alpha
        self.kwargs = kwargs

        self.set_activation(activation, output_activation)
        self.set_weights(activation, random_weights)
        self.set_cost(cost)

        self.optimal_weights = None

    def set_activation(self, activation, output_activation):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
    
        def sigmoid_derivative(x):
            return x * (1 - x)
        
        def ReLU(x):
            f = np.where(x > 0, x, np.zeros(x.shape))
            return f
        
        def ReLU_derivative(x):
            f = np.where(x > 0, np.ones(x.shape), np.zeros(x.shape))
            return f
        
        def leaky_ReLU(x):
            f = np.where(x > 0, x, 1e-4 * x)
            return f
        
        def leaky_ReLU_derivative(x):
            f = np.where(x > 0, np.ones(x.shape), 1e-4 * np.ones(x.shape))
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

        if output_activation == 'linear':
            self.output_activation = lambda x: x
            self.output_activation_diff = lambda x: 1
    
    def set_weights(self, activation, random_weights=False):
        def Xavier_weights(size_in, size_out):
            r'''
            Xavier/Glorot weights initialization. Helps aviod the 
            vanishing gradient problem. Suited for sigmoid and 
            hyperbolic activation functions.
            '''
            std = np.sqrt(2 / (size_in + size_out))
            weights = np.random.normal(0, std, size=(size_in, size_out))

            return weights
        
        def He_weights(size_in, size_out):
            r'''
            He weights initialization. Helps avoid dead nodes and 
            exploding gradients in deep networks. Suited for ReLU 
            activation functions.
            '''
            std = np.sqrt(2 / size_in)
            weights = np.random.normal(0, std, size=(size_in, size_out))

            return weights 
        
        def LeCun_weights(size_in, size_out):
            r'''
            LeCun weights initialization. Good for activation functions 
            with varying slopes. Suited for Leaky ReLU activation 
            functions.
            '''
            std = np.sqrt(1 / size_in)
            weights = np.random.normal(0, std, size=(size_in, size_out))

            return weights
        
        def rand_weights(size_in, size_out):
            r'''
            Randomized weights initialization. 
            '''
            return np.random.randn(size_in, size_out)
        
        weight_funcs = {'sigmoid': Xavier_weights,
                       'tanh': Xavier_weights,
                       'relu': He_weights,
                       'lrelu': LeCun_weights}
        
        if random_weights:
            init_weights = rand_weights
        
        else:
            init_weights = weight_funcs[activation]
        
        self.weights = []
        self.bias = []
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i+1]
            
            weights = init_weights(input_size, output_size)
            bias = np.ones((1, output_size)) * 0.01
            self.weights.append(weights)
            self.bias.append(bias)

    def set_cost(self, cost):
        def CostOLS(target):

            def func(X):
                return (1.0 / target.shape[0]) * np.sum((target - X)**2)
            
            return func
    
        if cost == 'mse': 
            self.cost_func = CostOLS
   
    def feed_forward(self, X):
        self.a = []
        self.z = []

        input_layer = X
        self.a.append(input_layer)
        self.z.append(input_layer)

        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                z = input_layer @ self.weights[i] + self.bias[i]
                a = self.activation(z)

                self.z.append(z)
                self.a.append(a)
            
            else:
                try:
                    z = input_layer @ self.weights[i]
                    a = self.output_activation(z)
                    self.z.append(z)
                    self.a.append(a)

                except Exception as OverflowError:
                    print('Overflow in fit()')
            
            input_layer = a
        
        return input_layer
    
    def backpropagate(self, X):
        dOut = self.output_activation_diff
        dHidden = self.activation_diff

        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                dCost = grad(self.cost_func(self.y_data))
                delta = dOut(self.z[i+1]) * dCost(self.a[i+1])

            else:
                delta = (self.weights[i+1] @ delta.T).T \
                      * dHidden(self.z[i+1])

            w_grad = self.a[i].T @ delta
            b_grad = np.sum(delta, axis=0)

            w_grad += self.weights[i] * self.alpha

            self.weights[i] -= self.eta * w_grad
            self.bias[i] -= self.eta * b_grad
    
    def fit(self, X, y, X_val=None, y_val=None, centered=False,
            patience=200):
        
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
        n_batches = int(np.ceil(n / self.batch_size))
        indices = np.arange(n)
        shuffled_inds = np.random.choice(indices, size=n, replace=False)

        rand_inds = []
        for idx in range(n_batches):
            start = idx * self.batch_size
            stop = (idx + 1) * self.batch_size
            inds = shuffled_inds[start:stop]
            rand_inds.append(inds)

        best_mse = np.inf
        counter = 0

        for i in range(self.epochs):
            for inds in rand_inds:                
                self.X_data = self.X_data_full[inds]
                self.y_data = self.y_data_full[inds]

                self.feed_forward(self.X_data)
                self.backpropagate(self.X_data)

                if has_validation:
                    ypred = self.predict(self.X_val)
                    mse = np.mean((self.y_val - ypred)**2)
                    # print(mse)

                else:
                    mse = np.mean((self.y_data - self.z_o)**2)

                if mse < best_mse:
                    best_mse = mse 
                    self.optimal_weights = self.weights
                    counter = 0

                else:
                    counter += 1
                
            if counter >= patience:
                self.weights = self.optimal_weights
                print(f'Early stopping at epoch {i} with MSE: {best_mse:.3f}')
                break 
        
        self.weights = self.optimal_weights
                    
    
    def predict(self, X):
        return self.feed_forward(X)


def test_func(x):
    a_0 = 1
    a_1 = 0.09
    a_2 = -0.3
    a_3 = 0.1
    # f = a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3
    f = 2 * np.sin(2 * x) + - 0.5 * np.cos(3 * x) + 0.3 * x**3

    return f

n = int(2e2)
x = np.linspace(-4, 4, n)[:, np.newaxis]
y_true = test_func(x)
y = y_true + np.random.normal(0, 1, x.shape)

X = calc.create_X(x, y, poly_deg=1, include_ones=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

alphas = np.logspace(-8, -5, 4)
etas = np.logspace(-8, -5, 4)
alpha_labels = [f'{alphas[i]:.1e}' for i in range(len(alphas))]
eta_labels = [f'{etas[i]:.1e}' for i in range(len(etas))]

MSEs = np.zeros((len(alphas), len(etas)))

tot = len(alphas) * len(etas)
best_alpha = 0
best_eta = 0
best_mse = np.inf
ypred = None

with alive_bar(tot, length=20, title='Processing...') as bar:
    for i in range(len(alphas)):
        for j in range(len(etas)):
            alpha = alphas[i]
            eta = etas[j]

            NN = NeuralNetwork(2, [50], activation='relu', eta=eta,
                               alpha=alpha, batch_size=60)
            NN.fit(X_train, y_train, X_test, y_test)
            _ypred = NN.predict(X)
            mse = np.mean((y - _ypred)**2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha 
                best_eta = eta
                ypred = _ypred

            MSEs[i, j] = mse 

            bar()
print(best_alpha, best_eta)

fig, ax = plt.subplots()
sns.heatmap(MSEs, annot=True, ax=ax, cmap='viridis', xticklabels=eta_labels,
            yticklabels=alpha_labels)
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\alpha$')
fig.tight_layout()

# NN = NeuralNetwork(2, [100], activation='relu', eta=1e-8, alpha=1e-10)
# NN.fit(X_train, y_train, X_test, y_test)

# ypred = NN.predict(X)
mse = np.mean((y - ypred)**2)
print(mse)
scikit = MLPRegressor(hidden_layer_sizes=(50), activation='relu', solver='sgd',
                      alpha=1e-7, batch_size=60, learning_rate_init=1e-5, momentum=0)
scikit.fit(X_train, y_train.ravel())
ypred_scikit = scikit.predict(X)
mse_scikit = np.mean((y - ypred_scikit)**2)
fig, ax = plt.subplots()
ax.set_title(f'MSE own: {mse:.2f} | MSE SciKit: {mse_scikit:.2f}')
ax.scatter(x, y, color='black', s=1, label='Data', alpha=0.75)
ax.plot(x, ypred, color='red', label='FNN')
ax.plot(x, y_true, color='black', ls='dashdot', label='True')
ax.plot(x, ypred_scikit, color='blue', ls='dashed', label='MPLSciKit')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()
plt.show()