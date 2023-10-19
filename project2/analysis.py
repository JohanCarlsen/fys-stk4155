import sys
import os 
sys.path.insert(0, '../project1/props')
from calc import Calculate as calc
import autograd.numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_theme()

np.random.seed(2023)

class RegressionAnalysis:

    def __init__(self, X, y):
        self.X, self.y = X, y

        self.n, self.p = X.shape
        self.calculate_mse = calc.mean_sq_err

        self.valid_optimizers = ['AdaGrad', 'RMSprop', 'ADAM']
        self.valid_grad_descents = ['GD', 'SGD']
        self.best_mse = np.inf
        
    def set_gradient(self):
        def grad_ols(n, **kwargs):
            return (2 / n) * self.X.T @ (self.X @ self.beta - self.y)
        
        def grad_ridge(n, lamb, **kwargs):
            return 2 * (self.X.T @ (self.X @ self.beta - self.y) + lamb * self.beta)

        if self.params.get('method') == 'OLS':
            self.grad_func = grad_ols
        
        else: 
            self.grad_func = grad_ridge
    
    def set_optimizer(self):

        def AdaGrad(gradient, eta):
            delta = 1e-8
            self.G_outer += gradient @ gradient.T
            G_diag = np.diagonal(self.G_outer)
            G_inv = np.c_[eta / (delta + np.sqrt(G_diag))]
            update = G_inv * gradient 
            self.beta -= update

        optimizer = self.params.get('optimizer')
        if optimizer is None:
            self.optimizer = lambda x, y, **nonargs: None
        
        elif optimizer == 'AdaGrad':
            self.optimizer = AdaGrad

    
    def gradient_descent(self, max_iter=int(1e5), **kwargs):
        self.params.update({'n_iterations': 0})
        for momentum in self.mom:
            for eta in self.eta:
                self.change = 0.0
                self.G_outer = np.zeros((self.p, self.p))
                self.beta = np.random.randn(self.p, 1)
                i = 0
                while i < max_iter:
                    gradient = self.grad_func(self.n, **kwargs)
                    self.beta -= eta * gradient
                    self.optimizer(gradient, eta)
                    self.momentum(gradient, eta, momentum)
                    
                    ypred = self.X @ self.beta
                    mse = self.calculate_mse(self.y, ypred)
                    
                    if mse < self.best_mse:
                        self.best_mse = mse
                        self.params.update({'eta': eta})
                        self.ypred = ypred
                    
                    elif (mse - self.best_mse) > 0:
                        print(i+1)
                        self.params['n_iterations'] = i+1
                        break
                    
                    i += 1
                    self.params['n_iterations'] = i
            
        
        self.params.update({'mse': self.best_mse})
    
    def set_momentum(self):

        if self.params.get('momentum') is None:
            self.momentum = lambda x, y, z: None 
        
        else:
            self.momentum = momentun()
        
        def momentum(gradient, eta, momentum):
            new_change = eta * gradient + momentum * self.change
            self.beta -= new_change
            self.change = new_change
    
    def set_params(self, method, gradient_descent, optimizer=None):
        if gradient_descent == 'GD':
            self.solver = self.gradient_descent

        self.params = {'method': method,
                       'gradient-descent': gradient_descent,
                       'optimizer': optimizer}
        
        self.set_gradient()
        self.set_optimizer()
    
    def set_hyper_params(self, learning_rate, momentum=None, **kwargs):
        if isinstance(learning_rate, float):
            self.eta = np.array([learning_rate])
        
        else:
            self.eta = learning_rate
        
        if momentum is None:
            self.mom = [0]
        
        else:
            self.mom = momentum

        self.params.update(kwargs)
        self.set_momentum()
    
    def run(self):
        self.solver(**self.params)


def test_func(x):
    a_0 = 1
    a_1 = 2
    a_2 = -5
    a_3 = 3
    f = a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3

    return f

n = int(1e2)
x = np.linspace(-2, 2, n)[:, np.newaxis]
y_true = test_func(x)
y = y_true + np.random.normal(0, 3, x.shape)
X = calc.create_X(x, poly_deg=3)

plt.scatter(x, y, color='black', s=3, label='Data')
plt.plot(x, y_true, color='blue', label='True')

reg = RegressionAnalysis(X, y)
reg.set_params(method='OLS', gradient_descent='GD', optimizer='AdaGrad')
eta = np.logspace(-3, -2, 2)
reg.set_hyper_params(learning_rate=eta)
reg.run()
plt.plot(x, reg.ypred)
print(reg.best_mse)
print(reg.params)

plt.legend()
plt.show()