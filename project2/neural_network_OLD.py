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
from copy import copy

sns.set_theme()
# warnings.filterwarnings('ignore')
np.random.seed(2023)

class Solver:
    def __init__(self, eta):
        self.eta = eta 
    
    def update_change(self, gradient):
        raise NotImplementedError
    
    def reset(self):
        pass 

class Constant(Solver):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient 
    
    def reset(self):
        pass 

class ADAM(Solver):
    def __init__(self, eta, rho1=0.9, rho2=0.999):
        super().__init__(eta)
        self.rho1 = rho1 
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8

        self.moment = self.rho1 * self.moment \
                    + (1 - self.rho1) * gradient 
        
        self.second = self.rho2 * self.second \
                    + (1 - self.rho2) * gradient * gradient 
        
        moment_corrected = self.moment / (1 - self.rho1**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))
    
    def reset(self):
        self.n_epochs += 1
        self.moment = 0 
        self.second = 0

class NeuralNetwork:

    def __init__(self, input_size, hidden_sizes=[1], output_size=1, 
                 activation='sigmoid', epochs=int(1e4),
                 batch_size=15,  eta=1e-4, alpha=1e-3,
                 random_weights=False, cost='mse',
                 output_activation='linear', solver='adam'):
        
        self.input_size = input_size
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.alpha = alpha

        if solver == 'adam':
            self.solver = ADAM(self.eta)

        elif solver == 'constant':
            self.solver = Constant(self.eta)

        self.set_activation(activation, output_activation)
        self.set_weights(activation, random_weights)
        self.set_cost(cost)

        self.optimal_weights = None
        self.optimal_bias = None

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
        
        self.w_solver = []
        self.b_solver = []
        for i in range(len(self.weights)):
            self.w_solver.append(copy(self.solver))
            self.b_solver.append(copy(self.solver))

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
                    z = input_layer @ self.weights[i] + self.bias[i]
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
            tot_grad = w_grad + b_grad
            self.norm = np.linalg.norm(tot_grad)

            w_grad += self.weights[i] * self.alpha

            w_change = self.w_solver[i].update_change(w_grad)
            b_change = self.b_solver[i].update_change(b_grad)
            
            self.weights[i] -= w_change
            self.bias[i] -= b_change
    
    def fit(self, X, y, X_val=None, y_val=None, centered=False,
            patience=10, tol=1e-3):
        
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
        rand_inds = np.array_split(shuffled_inds, n_batches)

        self.norm = np.inf
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
                    mse = np.mean((self.y_data - self.a[-1])**2)

                if mse < best_mse:
                    best_mse = mse 
                    self.optimal_weights = self.weights
                    self.optimal_bias = self.bias
                    counter = 0

                else:
                    counter += 1
            
            for w_solver, b_solver in zip(self.w_solver, self.b_solver):
                w_solver.reset()
                b_solver.reset()

            if counter >= patience:
                self.weights = self.optimal_weights
                self.bias = self.optimal_bias
                print(f'Early stopping at epoch {i} with MSE: {best_mse:.3f}')
                break 

            if self.norm <= tol:
                print(f'Converged after {i} epochs with MSE: {best_mse:.3f}')
                break
        
        # self.weights = self.optimal_weights
        # self.bias = self.optimal_bias                    
    
    def predict(self, X):
        return self.feed_forward(X)


def test_func(x):
    a_0 = 1
    a_1 = 0.09
    a_2 = -0.3
    a_3 = 0.1
    f = a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3
    # f = 2 * np.sin(2 * x) + - 0.5 * np.cos(3 * x) + 0.3 * x**3

    return f

n = int(1e2)
x = np.linspace(-4, 4, n)[:, np.newaxis]
y_true = test_func(x)
y_true = (y_true - np.min(y_true)) / (np.max(y_true - np.min(y_true)))
y = y_true + 0.1 * np.random.normal(0, 1, x.shape)

X = calc.create_X(x, poly_deg=1, include_ones=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

alphas = np.logspace(-5, -2, 4)
etas = np.logspace(-6, -3, 4)
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

            NN = NeuralNetwork(1, [5, 5], activation='relu', eta=eta,
                               alpha=alpha, batch_size=60)
            NN.fit(X_train, y_train, X_test, y_test)
            _ypred = NN.predict(X_test)
            mse = np.mean((y_test - _ypred)**2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha 
                best_eta = eta
                ypred = _ypred

            MSEs[i, j] = mse 

            bar()
print(best_alpha, best_eta)


# fig, ax = plt.subplots()
# sns.heatmap(MSEs, annot=True, ax=ax, cmap='viridis', xticklabels=eta_labels,
#             yticklabels=alpha_labels)

# ax.set_xlabel(r'$\eta$')
# ax.set_ylabel(r'$\alpha$')
# fig.tight_layout()

# NN = NeuralNetwork(1, [100], activation='relu', eta=1e-3, alpha=1e-3)
# NN.fit(X_train, y_train, X_test, y_test)

ypred = NN.predict(X)
mse = np.mean((y - ypred)**2)
print(mse)
scikit = MLPRegressor(hidden_layer_sizes=(100), activation='relu', solver='adam',
                      alpha=1e-5, batch_size=60, learning_rate_init=1e-3, momentum=1e-3)
scikit.fit(X_train, y_train.ravel())
ypred_scikit = scikit.predict(X)
mse_scikit = np.mean((y - ypred_scikit)**2)
fig, ax = plt.subplots()
ax.set_title(f'MSE own: {mse:.2f} | MSE SciKit: {mse_scikit:.2f}')
ax.scatter(x, y, color='black', s=2, label='Data', alpha=0.75)
ax.plot(x, ypred, color='red', label='FNN')
ax.plot(x, y_true, color='black', ls='dashdot', label='True')
ax.plot(x, ypred_scikit, color='blue', ls='dashed', label='MPLSciKit')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()
plt.show()