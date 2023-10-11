import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from src import Regression, frankes_function, compare_surface, set_size

np.random.seed(2018)

n = 1000
x, y = rand(2, n)
z = frankes_function(x, y, add_noise=True)

n_x = np.linspace(0, 1, n)
n_y = np.linspace(0, 1, n)

X, Y = np.meshgrid(n_x, n_y)
Z = frankes_function(X, Y, add_noise=True)

reg = Regression(x, y, z, 'franke')
ols_p, ols_beta = reg.OLS(16)
# ols_beta = ols_beta[:36]
# reg.plot_evolution('OLS')
# ridge_lasso_p, ridge_beta, lasso_beta = reg.ridge_and_lasso(-6, -2, ols_p, 100)
# print(reg.OLS_results)
# reg.plot_evolution('ridge')
# reg.plot_evolution('lasso')
# reg.bias_variance_tradeoff(max_degree=16, n_bootstraps=100)
# reg.cross_validation(n_kfolds=10, tradeoff=13)

compare_surface(n_x, 8, Z, reg_model='OLS')

plt.show()