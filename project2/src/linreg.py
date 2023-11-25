import sys
import os 
sys.path.insert(0, '../project1/props')
sys.path.insert(0, 'src')
from calc import Calculate as calc
import autograd.numpy as np 
from autograd import grad
import matplotlib.pyplot as plt 
import warnings
from sklearn.model_selection import train_test_split
from alive_progress import alive_bar
import pandas as pd
from itertools import product
from solvers import *

# import seaborn as sns 
# sns.set_theme()
warnings.filterwarnings('ignore')
np.random.seed(2023)

class RegressionAnalysis:
    r'''
    Regression class.

    Attributes
    ----------
    X : array_like
        Feature matrix.

    y : array_like
        Target values.
    '''
    def __init__(self, X, y):
        self._split(X, y)

        self.n, self.p = X.shape
        self.calculate_mse = calc.mean_sq_err

        self.valid_optimizers = ['AdaGrad', 'RMSprop', 'ADAM', 'const']
        self.valid_grad_descents = ['GD', 'SGD']

        self.score_evol = []
        self.add_mom = False
    
    def _split(self, X, y):
        r'''
        Split the data into training (80%) and test (20%) datasets.

        Parameters
        ----------
        X : array_like
            Feature matrix.

        y : array_like
            Target values.
        '''
        self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=0.2)
        
    def _set_gradient(self, autograd=False, **kwargs):
        r'''
        Set the gradient for the problem from the cost function.

        Parameters
        ----------
        autograd : boolean, optional
            If ``True``, uses the ``autograd`` package to calculate 
            the gradient. 
        '''
        def grad_ols(n, X, y, beta, *arg):
            return (2 / n) * X.T @ (X @ beta - y)
        
        def grad_ridge(n, X, y, beta, lamb):
            return 2 * (X.T @ (X @ beta - y) + lamb * beta)
        
        def cost_ols(n, X, y, beta, *arg):
            return (1.0 / n) * np.sum((y - X @ beta)**2)
        
        def cost_ridge(n, X, y, beta, lamb):
            return (1.0 / n) * np.sum((y - X @ beta)**2) + lamb * np.sum(np.abs(beta))

        if self.params.get('method') == 'OLS':
            if autograd:
                autograd_ols = grad(cost_ols, 3)
                self.grad_func = autograd_ols
            
            else:                
                self.grad_func = grad_ols
        
        else: 
            if autograd:
                autograd_ridge = grad(cost_ridge, 3)
                self.grad_func = autograd_ridge
            
            else:
                self.grad_func = grad_ridge
    
    def _set_optimizer(self):
        r'''
        Set the optimizer to use with the solver. 
        '''
        def AdaGrad(gradient, eta, *arg, **kwargs):
            delta = 1e-8
            self.G_outer += gradient @ gradient.T
            G_diag = np.diagonal(self.G_outer)
            G_inv = np.c_[eta / (delta + np.sqrt(G_diag))]
            update = G_inv * gradient 
            self.beta -= update
        
        def RMSprop(gradient, eta, *arg, **kwargs):
            delta = 1e-8
            rho = 0.99
            G_prev = self.G_outer
            self.G_outer += gradient @ gradient.T 
            G_new = (rho * G_prev + (1 - rho) * self.G_outer)
            G_diag = np.diagonal(G_new)
            G_inv = np.c_[eta / (delta + np.sqrt(G_diag))]
            update = G_inv * gradient 
            self.beta -= update
        
        def ADAM(gradient, eta, *arg, **kwargs):
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
        
        def constant(gradient, eta, *arg, **kwargs):
            if self.add_mom:
                momentum = arg[0]
                new_change = eta * gradient + momentum * self.change
                self.beta -= new_change
                self.change = new_change

            else:
                update = eta * gradient 
                self.beta -= update

        optimizer = self.params.get('optimizer')
        msg = f'Available optimizers are: {self.valid_optimizers}'
        assert optimizer in self.valid_optimizers, msg 

        if optimizer is None or optimizer == 'const':
            self.add_mom = True
            self.optimizer = constant
        
        elif optimizer == 'AdaGrad':
            self.optimizer = AdaGrad
        
        elif optimizer == 'RMSprop':
            self.optimizer = RMSprop
        
        else:
            self.optimizer = ADAM
    
    def _gradient_descent(self, max_iter=int(1e4), **kwargs):
        r'''
        Gradient descent solver method.

        Parameters
        ----------
        max_iter : int, optional
            Maximum numbers of iterations to perform (defaul: 10^5).
        '''
        self.params.update({'n_iterations': 0})

        def descent(m, eta, stochastic=False, **kwargs):
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
                if stochastic:
                    t0 = 1
                    t1 = 10
                    schedule = lambda t: t0/(t + t1)
                    t = epoch * m + i 
                    eta = schedule(t)

                self.iter += 1

                gradient = self.grad_func(n, X, y, self.beta, lamb)
                # self.momentum(gradient, eta, momentum)
                self.optimizer(gradient, eta, momentum)

                ypred = self.X_test @ self.beta 
                mse = self.calculate_mse(self.y_test, ypred)
                self.score_evol.append(mse)
                
                if mse < self.best_mse:
                    self.best_mse = mse
                    self.best_beta = self.beta
                    self.params.update({'eta': eta})
                    self.ypred = ypred
                    self.params['n_iterations'] = n_iter
                    self.params.update({'momentum': momentum})
                    self.params.update({'lamb': lamb})
                    self.counter = 0
                
                else:
                    self.counter += 1
            
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
                    self.counter = 0

                    if self.params['gradient-descent'] == 'GD':
                        m = max_iter
                        descent(m, eta)

                    else:
                        M = self.params['minibatch_size']
                        n_epochs = self.params['n_epochs']
                        m = int(self.n / M)

                        for epoch in range(n_epochs):
                            descent(m, eta, stochastic=True, M=M)
                    
                    if self.counter >= self.patience:
                        break
        
        self.params.update({'mse': self.best_mse})
    
    def _set_momentum(self):
        r'''
        Set the momentum for the solver.
        '''
        def momentum(gradient, eta, momentum):
            if self.add_mom:
                new_change = eta * gradient + momentum * self.change

            else:
                new_change = 0

            self.beta -= new_change
            self.change = new_change

        self.momentum = momentum        
    
    def set_params(self, method, gradient_descent, patience=200, optimizer=None, **params):
        r'''
        Set the parameters to use for the problem.

        Parameters
        ----------
        method : {'OLS', 'Ridge'}, str
            Which method to use. This also sets the cost function.

        gradient_descent : {'GD', 'SGD'}, str
            Weather to use plain gradiend descent or stochastic gradient
            descent.
        '''
        self.best_mse = np.inf
        self.solver = self._gradient_descent
        self.patience = patience

        self.params = {'method': method,
                       'gradient-descent': gradient_descent,
                       'optimizer': optimizer}
        
        self.params.update(params)
        self._set_gradient(**params)
        self._set_optimizer()
    
    def set_hyper_params(self, learning_rate, momentum=None, **hyperparams):
        r'''
        Set the hyper parameters to use.

        Parameters
        ----------
        learning_rate : float
            Learning rate (step size) to use.

        momentum : float, optional
            Momentum for the solver. 
        '''
        if isinstance(learning_rate, float):
            self.eta = np.array([learning_rate])
        
        else:
            self.eta = learning_rate
        
        if momentum is None:
            self.mom = [0]
        
        elif isinstance(momentum, float):
            self.mom = [momentum]
        
        else:
            self.mom = momentum
        
        lamb = hyperparams.get('lamb')
        if lamb is not None and self.params.get('method') != 'OLS':
            if isinstance(lamb, float):
                self.lamb = [lamb]
            
            else:
                self.lamb = lamb
        else:
            self.lamb = [0]

        self._set_momentum()
    
    def run(self):
        r'''
        Run the model. 
        '''
        self.score_evol = []
        self.solver(**self.params)

    def get_score_evol(self, limit=True):
        r'''
        Return the evolution of the score.

        Parameters
        ----------
        limit : boolean, optional
            If ``True`` (default), return the evolution up until the
            best score.

        Returns
        -------
        score_eval : array_like
            The array containing the score evolution,
        '''
        if limit:
            idx = np.argwhere(self.score_evol == self.best_mse)[0][0]

            return self.score_evol[:idx+1]
        
        else:
            return self.score_evol

if __name__ == '__main__':
    sys.path.insert(0, '../../project1/props')
    from preprocess import center, norm_data_zero_one
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
    y_true = norm_data_zero_one(test_func(x))
    y = y_true + np.random.normal(0, 0.1, x.shape)

    X = np.c_[np.ones(n), x, x**2, x**3]
    X = center(X)[:, 1:]   

    grads = ['GD', 'SGD']
    etas = np.logspace(-5, -2, 4)
    moms = [0.0, 0.9]
    epochs = np.logspace(1, 4, 4, dtype=np.int32)
    batch_sizes = np.arange(n/20, n+1, 20, dtype=np.int32)
    lambs = np.logspace(-5, -2, 4)

    olsGD_DF = pd.DataFrame()
    params = product(etas, moms)
    tot = len(etas) * len(moms)

    with alive_bar(tot, title='OLS GD...', length=20) as bar:
        for eta, mom in params:
            reg = RegressionAnalysis(X, y)
            reg.set_params(method='OLS', gradient_descent='GD',
                           optimizer='ADAM')
            reg.set_hyper_params(learning_rate=eta, momentum=mom)
            reg.run()
            mse = reg.best_mse

            olsGD_DF = olsGD_DF.append({'mse': mse, 'eta': eta,
                                        'mom': mom}, ignore_index=True)
            bar()
    
    olsGD_DF.to_csv('ADAM-OLS-GD.csv', index=False)

    olsSGD_DF = pd.DataFrame()
    params = product(etas, moms, epochs, batch_sizes)
    tot *= len(epochs) * len(batch_sizes)

    with alive_bar(tot, title='OLS SDG...', length=20) as bar:
        for eta, mom, epoch, batch in params:
            reg = RegressionAnalysis(X, y)
            reg.set_params(method='OLS', gradient_descent='SGD',
                           optimizer='ADAM', n_epochs=epoch,
                           minibatch_size=batch)
            reg.set_hyper_params(learning_rate=eta, momentum=mom)
            reg.run()
            mse = reg.best_mse

            olsSGD_DF = olsSGD_DF.append({'mse': mse, 'eta': eta,
                                          'mom': mom, 'batch': batch,
                                          'epoch': epoch}, ignore_index=True)
            bar()
    
    olsSGD_DF.to_csv('ADAM-OLS-SGD.csv', index=False)

    ridgeGD_DF = pd.DataFrame()
    params = product(etas, moms, lambs)
    tot = len(etas) * len(moms) * len(lambs)

    with alive_bar(tot, title='Ridge GD...', length=20) as bar:
        for eta, mom, lamb in params:
            reg = RegressionAnalysis(X, y)
            reg.set_params(method='Rdige', gradient_descent='GD',
                           optimizer='ADAM')
            reg.set_hyper_params(learning_rate=eta, momentum=mom,
                                 lamb=lamb)
            reg.run()
            mse = reg.best_mse

            ridgeGD_DF = ridgeGD_DF.append({'mse': mse, 'eta': eta,
                                            'mom': mom, 'lamb': lamb},
                                            ignore_index=True)
            bar()

    ridgeGD_DF.to_csv('ADAM-Ridge-GD.csv', index=False)

    ridgeSGD_DF = pd.DataFrame()
    params = product(etas, moms, lambs, epochs, batch_sizes)
    tot *= len(epochs) * len(batch_sizes)

    with alive_bar(tot, title='Ridge SDG...', length=20) as bar:
        for eta, mom, lamb, epoch, batch in params:
            reg = RegressionAnalysis(X, y)
            reg.set_params(method='Ridge', gradient_descent='SGD',
                           optimizer='ADAM', n_epochs=epoch,
                           minibatch_size=batch)
            reg.set_hyper_params(learning_rate=eta, momentum=mom,
                                 lamb=lamb)
            reg.run()
            mse = reg.best_mse

            ridgeSGD_DF = ridgeSGD_DF.append({'mse': mse, 'eta': eta,
                                              'mom': mom, 'lamb': lamb,
                                              'batch': batch, 
                                              'epoch': epoch}, ignore_index=True)
            bar()
    
    ridgeSGD_DF.to_csv('ADAM-Ridge-SGD.csv', index=False)

    plt.show()