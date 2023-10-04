import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from src import Regression, set_size, compare_terrain
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load the terrain
terrain1 = np.array(imread('n59_e008_1arc_v3.tif')).T
terrain2 = np.array(imread('n59_e007_1arc_v3.tif')).T
_terrain = np.concatenate(np.array([terrain2, terrain1])).T * 1e-3

np.random.seed(2023)

tot_points = np.min(_terrain.shape)
n_samples = 100 # Should be 1500 for OLS and maybe make another instance with 100 for ridge and lasso and 100 for plotting ols results
data_samples = np.random.randint(0, tot_points, size=(n_samples, 2))

n_x = np.sort(data_samples[:, 0])
n_y = np.sort(data_samples[:, 1])

xmesh, ymesh = np.meshgrid(n_x, n_y)
terrain = _terrain[xmesh, ymesh]

reg = Regression(xmesh.ravel(), ymesh.ravel(), terrain.ravel(), 'geodata')
_, __ = reg.OLS(22, store_beta=False)
reg.plot_evolution('OLS')
ridge_lasso_p, ridge_beta, lasso_beta = reg.ridge(-8, 7, 5, 500)
print(reg.OLS_results)
reg.plot_evolution('ridge')
reg.plot_evolution('lasso')
reg.bias_variance_tradeoff(max_degree=31, n_bootstraps=100)
reg.cross_validation(n_kfolds=10)

n_samples = 1500
data_samples = np.random.randint(0, tot_points, size=(n_samples, 2))

n_x = np.sort(data_samples[:, 0])
n_y = np.sort(data_samples[:, 1])

xmesh, ymesh = np.meshgrid(n_x, n_y)
terrain = _terrain[xmesh, ymesh]

reg = Regression(xmesh.ravel(), ymesh.ravel(), terrain.ravel(), 'geodata')
ols_p, ols_beta = reg.OLS(22, store_beta=False)
compare_terrain(_terrain, ols_p, ols_beta, model_points=500, n_samples=n_samples, reg_model='OLS')
# compare_terrain(_terrain, ridge_lasso_p, ridge_beta, model_points=1000, n_samples=n_samples, reg_model='Ridge')
# compare_terrain(_terrain, ridge_lasso_p, lasso_beta, model_points=1000, n_samples=n_samples, reg_model='Lasso')

plt.show()

# # Show the terrain
# fig, ax = plt.subplots(figsize=set_size())
# ax.set_title('Terrain over Telemark, Norway')
# im = ax.imshow(_terrain, cmap='terrain')
# fig.colorbar(im, label='Elevation [km]', pad=0.02)
# ax.set_xlabel('X [arcsec]')
# ax.set_ylabel('Y [arcsec]')

# fig.tight_layout()
# fig.savefig('figures/geo-data.pdf')
# fig.savefig('figures/geo-data.png')

# plt.show()