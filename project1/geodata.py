import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from src import Regression, set_size, compare_terrain
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load the terrain
terrain = np.array(imread('n59_e007_1arc_v3.tif').T)

np.random.seed(2023)

tot_points = np.min(terrain.shape)
terrain = terrain[:tot_points, :tot_points]

n_samples = tot_points
data_samples = np.random.randint(0, tot_points, size=(n_samples, 2))

x = data_samples[:, 0]
y = data_samples[:, 1]
xmesh, ymesh = np.meshgrid(x, y)

terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))

reg = Regression(x, y, terrain[x, y], 'geodata')
# ols_p, ols_beta = reg.OLS(21, store_beta=False)
# reg.plot_evolution('OLS')
# ridge_lasso_p, ridge_beta, lasso_beta = reg.ridge_and_lasso(-8, -2, ols_p, 100)
# print(reg.OLS_results)
# reg.plot_evolution('ridge')
# reg.plot_evolution('lasso')
# reg.bias_variance_tradeoff(max_degree=18, n_bootstraps=100)
# reg.cross_validation(n_kfolds=10, tradeoff=15)

N = n_samples
_x = np.linspace(0, 1, N)
_y = np.linspace(0, 1, N)
# X, Y = np.meshgrid(_x, _y)

# feature = reg.calculate.create_X(X, Y, 15)

# beta, ytilde, __ = reg.calculate.ord_least_sq(feature, terrain.ravel())
# ytilde = ytilde.reshape(N, N)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=set_size('text'), subplot_kw={'projection': '3d'})
# ax1.plot_surface(X, Y, ytilde, linewidth=0, antialiased=False, cmap='terrain')
# ax2.plot_surface(X, Y, terrain, linewidth=0, antialiased=False, cmap='terrain')

compare_terrain(_x, 15, terrain, reg_model='OLS')
# compare_terrain(terrain, ols_p, ols_beta, n_samples=n_samples, reg_model='OLS')
# compare_terrain(terrain, ridge_lasso_p, ridge_beta, n_samples=n_samples, reg_model='Ridge')
# compare_terrain(terrain, ridge_lasso_p, lasso_beta, n_samples=n_samples, reg_model='Lasso')


# # Show the terrain
# fig, ax = plt.subplots(figsize=set_size('text'))
# ax.set_title('Terrain over Telemark, Norway')
# im = ax.imshow(terrain, cmap='terrain')
# fig.colorbar(im, label='Normalized elevation [km]', pad=0.02, aspect=30)
# ax.set_xlabel(r'$x$ [arcsec]')
# ax.set_ylabel(r'$y$ [arcsec]')

# fig.tight_layout()
# fig.savefig('figures/geo-data.pdf')
# fig.savefig('figures/geo-data.png')

plt.show()