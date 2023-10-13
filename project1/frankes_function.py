import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from src import Regression, frankes_function, compare_surface, set_size
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(2018)

n = 1000
x, y = rand(2, n)
z = frankes_function(x, y, add_noise=True)

reg = Regression(x, y, z, 'franke')
ols_p, ols_beta = reg.OLS(16, identity_test=True)
reg.plot_evolution('OLS')
ridge_lasso_p, ridge_beta, lasso_beta = reg.ridge_and_lasso(-6, -3, ols_p, 75)
print(reg.OLS_results)
reg.plot_evolution('ridge', scale_down=True)
reg.plot_evolution('lasso')
reg.bias_variance_tradeoff(max_degree=16, n_bootstraps=100)
reg.cross_validation(n_kfolds=10, tradeoff=13)

linspace = np.linspace(0, 1, n)
X, Y = np.meshgrid(linspace, linspace)
Z = frankes_function(X, Y, add_noise=True)
                   
compare_surface(linspace, 8, Z, 'Franke OLS', cmap='jet')

# Show the Frank function surface
fig, ax = plt.subplots(figsize=set_size(), subplot_kw={'projection': '3d'})
im = ax.plot_surface(X, Y, Z,
                     linewidth=0,
                     antialiased=False,
                     cmap='jet')

fig.colorbar(im, ax=ax, pad=0.08, shrink=0.75, aspect=15, label='$z$')

ax.view_init(azim=45)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')

fig.savefig('figures/franke-surface.png')
fig.savefig('figures/franke-surface.pdf')

plt.show()