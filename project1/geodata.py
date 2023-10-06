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
n_samples = tot_points
data_samples = np.random.randint(0, tot_points, size=(n_samples, 2))

n_x = data_samples[:, 0]
n_y = data_samples[:, 1]

x = (n_x - np.min(n_x)) / (np.max(n_x) - np.min(n_x))
y = (n_y - np.min(n_y)) / (np.max(n_y) - np.min(n_y))

xmesh, ymesh = np.meshgrid(n_x, n_y)
terrain = _terrain[xmesh, ymesh]

reg = Regression(x, y, _terrain[n_x, n_y], 'geodata')
ols_p, ols_beta = reg.OLS(21, store_beta=False)
reg.plot_evolution('OLS')
ridge_lasso_p, ridge_beta, lasso_beta = reg.ridge_and_lasso(-4, -1, 10, 100)
ridge_lasso_p, ridge_beta, lasso_beta = reg.ridge_and_lasso(-7, 7, 6, 200)
print(reg.OLS_results)
reg.plot_evolution('ridge', add_zoom=True)
reg.plot_evolution('lasso')
reg.bias_variance_tradeoff(max_degree=35, n_bootstraps=100) 
reg.cross_validation(n_kfolds=10)

compare_terrain(_terrain, ols_p, ols_beta, n_samples=n_samples, reg_model='OLS')
compare_terrain(_terrain, ridge_lasso_p, ridge_beta, n_samples=n_samples, reg_model='Ridge')
compare_terrain(_terrain, ridge_lasso_p, lasso_beta, n_samples=n_samples, reg_model='Lasso')


# Show the terrain
fig, ax = plt.subplots(figsize=set_size('text'))
ax.set_title('Terrain over Telemark, Norway')
im = ax.imshow(_terrain, cmap='terrain')
fig.colorbar(im, label='Elevation [km]', pad=0.02)
ax.set_xlabel(r'$x$ [arcsec]')
ax.set_ylabel(r'$y$ [arcsec]')

fig.tight_layout()
fig.savefig('figures/geo-data.pdf')
fig.savefig('figures/geo-data.png')

plt.show()