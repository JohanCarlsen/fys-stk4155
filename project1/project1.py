import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from src import Regression, frankes_function, compare_terrain

np.random.seed(2018)

n = 2000
x, y = rand(2, n)
z = frankes_function(x, y)

n_x = np.linspace(0, 1, n)
n_y = np.linspace(0, 1, n)

X, Y = np.meshgrid(n_x, n_y)
Z = frankes_function(X, Y)

reg = Regression(x, y, z, 'franke')
ols_p, ols_beta = reg.OLS(15)
ols_beta = ols_beta[:36]
# reg.plot_evolution('OLS')
# ridge_lasso_p, ridge_beta, lasso_beta = reg.ridge_and_lasso(-4, 2, 5, 1000)
# reg.plot_evolution('ridge')
# reg.plot_evolution('lasso')
# reg.bias_variance_tradeoff(max_degree=15, n_bootstraps=100)
# reg.cross_validation(n_kfolds=10)

compare_terrain(Z, ols_p, ols_beta, n_samples=n, reg_model='OLS')
# compare_terrain(_terrain, ridge_lasso_p, ridge_beta, n_samples=n_samples, reg_model='Ridge')
# compare_terrain(_terrain, ridge_lasso_p, lasso_beta, n_samples=n_samples, reg_model='Lasso')

plt.show()