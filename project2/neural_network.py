import autograd.numpy as np 
from autograd import grad, elementwise_grad
from sklearn.model_selection import train_test_split
from copy import copy
import matplotlib.pyplot as plt 
from alive_progress import alive_bar
import seaborn as sns 
import sys
import os 
sys.path.insert(0, '../project1/props')
from calc import Calculate as calc

sns.set_theme()
np.random.seed(2023)

class Solvers:
    def __init__(self, eta):
        self.eta = eta 
    
    def update_change(self, gradient):
        raise NotImplementedError
    
    def reset(self):
        pass 

class Constant(Solvers):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient 
    
    def reset(self):
        pass 

class ADAM(Solvers):

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
        return 1

class Sigmoid(Activations):
    def function(x):
        return 1 / (1 + np.exp(-x))

    def derivative(x):
        return Sigmoid.function(x) * (1 - Sigmoid.function(x))

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
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(x):
        # The derivative of Softmax is a bit more complex and involves Jacobian matrix
        raise NotImplementedError  # You may need to implement this specifically

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
    
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, eta, alpha,
                 hidden_activation, output_activation, cost_function,
                 epochs, batch_size, solver, random_weights=False):
        
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.eta = eta 
        self.alpha = alpha 
        self.epochs = epochs
        self.batch_size = batch_size

        if hidden_activation == 'sigmoid':
            self.hidden_func = Sigmoid
            self.weight_func = XavierInitializer
        
        elif hidden_activation == 'relu':
            self.hidden_func = ReLU
            self.weight_func = HeInitializer

        elif hidden_activation == 'lrelu':
            self.hidden_func = LeakyReLU
            self.weight_func = LeCunInitializer

        if random_weights:
            self.weight_func = RandomInitializer

        if output_activation == 'linear':
            self.output_func = Linear

        if cost_function == 'mse':
            self.cost_func = MeanSquaredError

        if solver == 'adam':
            self.solver = ADAM(self.eta)

        self.create_weights_and_bias()

    def create_weights_and_bias(self):
        self.weights = []
        self.bias = []

        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i+1]

            weights = self.weight_func.initialize(input_size, output_size)
            bias = np.ones((1, output_size)) * 0.01

            self.weights.append(weights)
            self.bias.append(bias)
        
        self.w_solver = []
        self.b_solver = []

        for i in range(len(self.weights)):
            self.w_solver.append(copy(self.solver))
            self.b_solver.append(copy(self.solver))

    def _feed_forward(self, X):
        self.a = [X]
        self.z = [X]

        input_layer = X 
        for i in range(len(self.weights)):
            z = (input_layer @ self.weights[i]) + self.bias[i]

            if i < len(self.weights) - 1:
                a = self.hidden_func.function(z)
            
            else:
                a = self.output_func.function(z)

            self.z.append(z)
            self.a.append(a)
            input_layer = a

        return a 
    
    def _backpropagate(self, X, y):
        y_pred = self._feed_forward(X)
        dOut = self.output_func.derivative
        dCost = self.cost_func.gradient
        dHidden = self.hidden_func.derivative

        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                delta = dOut(self.a[-1]) * dCost(y, y_pred)

            else:
                # print(dHidden(self.a[i+1]).shape)
                delta = (delta @ self.weights[i+1].T) * dHidden(self.a[i+1])

            
            w_grad = (self.a[i].T @ delta) + self.weights[i] * self.alpha
            b_grad = np.sum(delta, axis=0)

            w_change = self.w_solver[i].update_change(w_grad)
            b_change = self.b_solver[i].update_change(b_grad)

            self.weights[i] -= w_change
            self.bias[i] -= b_change

    def predict(self, X):
        y_pred = self._feed_forward(X)

        return y_pred

    def fit(self, X, y, X_val, y_val, tol=1e-4, patience=1000):
        n = len(X)
        n_batches = int(np.ceil(n / self.batch_size))
        indices = np.arange(n)
        shuffled_inds = np.random.choice(indices, size=n, replace=False)
        rand_inds = np.array_split(shuffled_inds, n_batches)

        best_loss = np.inf
        best_weights = None 
        best_bias = None
        counter = 0

        for i in range(self.epochs):
            for inds in rand_inds:
                xi = X[inds]
                yi = y[inds]

                self._backpropagate(xi, yi)
                y_pred = self.predict(X_val)
                loss = self.cost_func.loss(y_val, y_pred)

                if loss < best_loss:
                    counter = 0
                    if abs(loss - best_loss) <= tol:
                        break

                    # print(loss)

                    best_loss = loss
                    best_weights = self.weights
                    best_bias = self.bias 

                else:
                    counter += 1
                
            for w_solver, b_solver in zip(self.w_solver, self.b_solver):
                w_solver.reset()
                b_solver.reset()

            if counter >= patience:
                print(f'Early stopping at epoch {i} with loss: {best_loss}')
                break            

        if best_loss < np.inf:
            self.weights = best_weights
            self.bias = best_bias

        else:
            print('Loss did not improve.')


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
# y_true = np.sin(x)
y_true = (y_true - np.min(y_true)) / (np.max(y_true) - np.min(y_true))
y = y_true + 0.1 * np.random.normal(0, 1, x.shape)

X = calc.create_X(x, poly_deg=1, include_ones=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

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

            NN = NeuralNetwork(1, [50, 50], 1, eta, alpha, 'relu', 'linear', 'mse', int(1e4), 80, 'adam')
            NN.fit(X_train, y_train, X_test, y_test)
            _ypred = NN.predict(X_test)
            mse = MeanSquaredError.loss(y_test, _ypred)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha 
                best_eta = eta
                ypred = _ypred

            MSEs[i, j] = mse 
            bar()

fig, ax = plt.subplots()
sns.heatmap(MSEs, annot=True, ax=ax, cmap='viridis', cbar_kws={'label': 'MSE'},
            xticklabels=eta_labels, yticklabels=alpha_labels)

ax.set_title('Test MSEs')
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\alpha$')

NN = NeuralNetwork(1, [100, 100], 1, best_eta, best_alpha, 'relu', 'linear', 'mse', int(1e4), 80, 'adam')
NN.fit(X_train, y_train, X_test, y_test)
ypred = NN.predict(X)
mse = MeanSquaredError.loss(y, ypred)

fig, ax = plt.subplots()
ax.set_title(f'MSE own: {mse:.2f}')
# ax.scatter(x, y, color='black', s=2, label='Data', alpha=0.75)
ax.plot(x, ypred, color='red', label='FNN')
ax.plot(x, y_true, color='black', ls='dashed', label='Target')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()
plt.show()


    

