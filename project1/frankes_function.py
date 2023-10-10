import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from src import Regression, frankes_function, compare_terrain, set_size

np.random.seed(2018)

n = 1000
x, y = rand(2, n)
z = frankes_function(x, y, add_noise=True)

n_x = np.linspace(0, 1, n)
n_y = np.linspace(0, 1, n)

X, Y = np.meshgrid(n_x, n_y)
Z = frankes_function(X, Y, add_noise=True)

reg = Regression(x, y, z, 'franke')
feature = reg.calculate.create_X(X, Y, 13)
beta, ytilde, _ = reg.calculate.ord_least_sq(feature, Z.ravel())
ytilde = ytilde.reshape(n, n)

fig, ax = plt.subplots(figsize=set_size('text'), subplot_kw={'projection': '3d'})
ax.plot_surface(X, Y, ytilde, linewidth=0, cmap='terrain')


# ols_p, ols_beta = reg.OLS(25)
# ols_beta = ols_beta[:36]
# reg.plot_evolution('OLS')
# ridge_lasso_p, ridge_beta, lasso_beta = reg.ridge_and_lasso(-5, 5, ols_p, 1000)
# reg.plot_evolution('ridge', add_zoom=True)
# reg.plot_evolution('lasso')
# reg.bias_variance_tradeoff(max_degree=20, n_bootstraps=100)
# reg.cross_validation(n_kfolds=10, tradeoff=17)

# compare_terrain(Z, ols_p, ols_beta, n_samples=n, reg_model='OLS')
# compare_terrain(_terrain, ridge_lasso_p, ridge_beta, n_samples=n_samples, reg_model='Ridge')
# compare_terrain(_terrain, ridge_lasso_p, lasso_beta, n_samples=n_samples, reg_model='Lasso')

plt.show()