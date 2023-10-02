import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from src import Regression, set_size
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load the terrain
terrain1 = np.array(imread('n59_e008_1arc_v3.tif')).T
terrain2 = np.array(imread('n59_e007_1arc_v3.tif')).T
_terrain = np.concatenate(np.array([terrain2, terrain1])).T / 1e3

np.random.seed(2023)

tot_points = np.min(_terrain.shape)
n_samples = 1000
data_samples = np.random.randint(0, tot_points, size=(n_samples, 2))
x = data_samples[:, 0]
y = data_samples[:, 1]

terrain = _terrain[x, y]

reg = Regression(x, y, terrain.T, 'geodata')
reg.OLS(30, store_beta=False)
reg.plot_evolution('OLS')
reg.ridge(-8, 8, 5, 1000)
print(reg.OLS_results)
reg.plot_evolution('ridge')
reg.plot_evolution('lasso')
reg.bias_variance_tradeoff(max_degree=16, n_bootstraps=100)
reg.cross_validation(n_kfolds=10)

plt.show()


# Show the terrain
fig, ax = plt.subplots(figsize=set_size())
ax.set_title('Terrain over Telemark, Norway')
im = ax.imshow(_terrain, cmap='terrain')
fig.colorbar(im, label='Elevation [km]', pad=0.02)
ax.set_xlabel('X [arcsec]')
ax.set_ylabel('Y [arcsec]')

fig.tight_layout()
fig.savefig('figures/geo-data.pdf')
fig.savefig('figures/geo-data.png')

plt.show()