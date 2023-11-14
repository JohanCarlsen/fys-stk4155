import autograd.numpy as np 
from copy import copy
from cost_funcs import *
from solvers import *
from activations import *

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, eta, alpha,
                 hidden_activation, output_activation, cost_function,
                 epochs, batch_size, solver, random_weights=False):
        
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.eta = eta 
        self.alpha = alpha 
        self.epochs = epochs
        self.batch_size = batch_size

        activs = {'sigmoid': [Sigmoid, XavierInitializer],
                  'relu': [ReLU, HeInitializer],
                  'lrelu': [LeakyReLU, LeCunInitializer],
                  'linear': Linear}
        
        costs = {'mse': MeanSquaredError,
                 'log': CrossEntropy}
        
        solvers = {'adam': ADAM(self.eta),
                   'constant': Constant(self.eta)}
        
        self.hidden_func, self.weight_func = activs[hidden_activation]
        self.output_func = activs[output_activation]
        self.cost_func = costs[cost_function]
        self.solver = solvers[solver]

        if random_weights:
            self.weight_func = RandomInitializer

        self.create_weights_and_bias()

    def create_weights_and_bias(self):
        self.weights = []
        self.bias = []

        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i+1]

            weights = self.weight_func.initialize(input_size, output_size)
            bias = np.zeros((1, output_size))

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

    def fit(self, X, y, X_val, y_val, tol=1e-4, patience=100):
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