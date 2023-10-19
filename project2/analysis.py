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
        
    def set_gradient(self):
        def grad_ols(n, X, y, *arg):
            return (2 / n) * X.T @ (X @ self.beta - y)
        
        def grad_ridge(n, X, y, lamb):
            return 2 * (X.T @ (X @ self.beta - y) + lamb * self.beta)

        if self.params.get('method') == 'OLS':
            self.grad_func = grad_ols
        
        else: 
            self.grad_func = grad_ridge
    
    def set_optimizer(self):

        def AdaGrad(gradient, eta, **kwargs):
            delta = 1e-8
            self.G_outer += gradient @ gradient.T
            G_diag = np.diagonal(self.G_outer)
            G_inv = np.c_[eta / (delta + np.sqrt(G_diag))]
            update = G_inv * gradient 
            self.beta -= update
        
        def RMSprop(gradient, eta, **kwargs):
            delta = 1e-8
            rho = 0.99
            G_prev = self.G_outer
            self.G_outer += gradient @ gradient.T 
            G_new = (rho * G_prev + (1 - rho) * self.G_outer)
            G_diag = np.diagonal(G_new)
            G_inv = np.c_[eta / (delta + np.sqrt(G_diag))]
            update = G_inv * gradient 
            self.beta -= update
        
        def ADAM(gradient, eta, **kwargs):
            delta = 1e-8
            rho1 = 0.9
            rho2 = 0.999
            self.first_moment = rho1 * self.first_moment + \
                                (1 - rho1) * gradient
            
            self.second_moment = rho2 * self.second_moment + \
                                 (1 - rho2) * gradient**2
            
            first_term = self.first_moment / (1.0 - rho1**self.iter)
            second_term = self.second_moment / (1.0 - rho2**self.iter)
            update = eta * first_term / (delta + np.sqrt(second_term))
            self.beta -= update

        optimizer = self.params.get('optimizer')
        msg = f'Available optimizers are: {self.valid_optimizers}'
        assert optimizer in self.valid_optimizers, msg 

        if optimizer is None:
            self.optimizer = lambda x, y, **nonargs: None
        
        elif optimizer == 'AdaGrad':
            self.optimizer = AdaGrad
        
        elif optimizer == 'RMSprop':
            self.optimizer = RMSprop
        
        else:
            self.optimizer = ADAM
    
    def gradient_descent(self, max_iter=int(1e5), **kwargs):
        self.params.update({'n_iterations': 0})

        def descent(m, stochastic=False, **kwargs):
            i = 0

            if not stochastic:
                X, y = self.X, self.y
                n = self.n
                n_iter = i
            
            else:
                rand_ind = np.random.randint(m)
                X = self.X[rand_ind:rand_ind + M]
                y = self.y[rand_ind:rand_ind + M]
                n = M
                n_iter = epoch

            while i < m:
                self.iter += 1
                gradient = self.grad_func(n, X, y, lamb)
                self.beta -= eta * gradient 
                self.momentum(gradient, eta, momentum)
                self.optimizer(gradient, eta)

                ypred = self.X @ self.beta 
                mse = self.calculate_mse(self.y, ypred)
                
                if mse < self.best_mse:
                    self.best_mse = mse
                    self.params.update({'eta': eta})
                    self.ypred = ypred
                    self.params['n_iterations'] = n_iter
                    self.params['momentum'] = momentum
            
                i += 1
                n_iter += 1

        for lamb in self.lamb:
            for momentum in self.mom:
                for eta in self.eta:
                    self.change = 0.0
                    self.G_outer = np.zeros((self.p, self.p))
                    self.beta = np.random.randn(self.p, 1)
                    self.first_moment = 0.0
                    self.second_moment = 0.0
                    self.iter = 0

                    if self.params['gradient-descent'] == 'GD':
                        m = max_iter
                        descent(m)

                    else:
                        M = self.params['minibatch_size']
                        n_epochs = self.params['n_epochs']
                        m = int(self.n / M)

                        for epoch in range(n_epochs):
                            descent(m, stochastic=True, M=M)

                    if self.params['n_iterations'] == 0:
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
    
    def set_params(self, method, gradient_descent, optimizer=None, **params):
        self.best_mse = np.inf
        self.solver = self.gradient_descent

        self.params = {'method': method,
                       'gradient-descent': gradient_descent,
                       'optimizer': optimizer}
        
        self.params.update(params)
        self.set_gradient()
        self.set_optimizer()
    
    def set_hyper_params(self, learning_rate, momentum=None, **hyperparams):
        if isinstance(learning_rate, float):
            self.eta = np.array([learning_rate])
        
        else:
            self.eta = learning_rate
        
        if momentum is None:
            self.mom = [0]
        
        else:
            self.mom = momentum
        
        lamb = hyperparams.get('lamb')
        if lamb is not None:

            if isinstance(lamb, float):
                self.lamb = [lamb]
            
            else:
                self.lamb = lamb
        else:
            self.lamb = [0]

        self.set_momentum()
    
    def run(self):
        self.solver(**self.params)


def test_func(x):
    a_0 = 1
    a_1 = 2
    a_2 = -5
    a_3 = 3
    # f = a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3
    f = 2 * np.sin(2 * x) + - 0.5 * np.cos(3 * x) + 0.3 * x**3

    return f

n = int(1e2)
x = np.linspace(-4, 4, n)[:, np.newaxis]
y_true = test_func(x)
y = y_true + np.random.normal(0, 1, x.shape)

X = calc.create_X(x, poly_deg=3)
optimizer = 'ADAM'
title = f'Optimizer: {optimizer}\n'
fig = plt.figure(figsize=(8,6))
plt.scatter(x, y, color='black', s=3, label='Data', alpha=0.5)
plt.plot(x, y_true, color='blue', ls='dotted', label='True')

reg = RegressionAnalysis(X, y)
reg.set_params(method='OLS', gradient_descent='GD', optimizer=optimizer, max_iter=int(1e4))

eta = np.logspace(-5, -2, 4)
mom = [None] + list(np.linspace(0.1, 0.9, 9))

reg.set_hyper_params(learning_rate=1e-3, momentum=mom)
reg.run()
print(reg.params)
title += r'MSE$_\mathrm{OLS}$: ' + f'{reg.best_mse:.3f} | '
plt.plot(x, reg.ypred, ls='dashdot', color='green', label='OLS GD')

reg.set_params(method='Ridge', gradient_descent='GD', optimizer=optimizer, max_iter=int(2.5e4))
reg.set_hyper_params(learning_rate=eta, lamb=1e-3)
reg.run()
print(reg.params)
title += r'MSE$_\mathrm{Ridge}$: ' + f'{reg.best_mse:.3f}'
plt.plot(x, reg.ypred, ls='dashed', color='red', label='Ridge GD')

reg.set_params(method='OLS', gradient_descent='SGD', optimizer=optimizer, n_epochs=int(1e4), minibatch_size=10)
reg.set_hyper_params(learning_rate=eta)
reg.run()
plt.plot(x, reg.ypred, color='magenta', label='OLS SGD')
print(reg.params)
title += '\n' + r'MSE$_\mathrm{OLS, SGD}$: ' + f'{reg.best_mse:.3f} | '

reg.set_params(method='Ridge', gradient_descent='SGD', optimizer=optimizer, n_epochs=int(1e3), minibatch_size=10)
reg.set_hyper_params(learning_rate=eta, lamb=1e-3)
reg.run()
plt.plot(x, reg.ypred, color='orange', ls=(0, (3,5,1,5,1,5)), label='Ridge SGD')
print(reg.params)
title += r'MSE$_\mathrm{Ridge, SLGD}$: ' + f'{reg.best_mse:.3f}'

fig.tight_layout()
plt.title(title)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.show()