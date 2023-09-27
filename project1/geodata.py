import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from src import Regression, set_size

# Load the terrain
terrain1 = np.array(imread('n59_e008_1arc_v3.tif')).T
terrain2 = np.array(imread('n59_e007_1arc_v3.tif')).T
terrain = np.concatenate(np.array([terrain2, terrain1])).T / 1e3

n_points = np.min(terrain.shape)
terrain = terrain[:n_points, :n_points]

x = np.arange(n_points)
y = np.arange(n_points)

reg = Regression(x, y, terrain.T, 'geodata')
reg.OLS(30, store_beta=False)
reg.plot_evolution('OLS')
reg.ridge(-5, 1, 3, 50)
reg.plot_evolution('ridge')
reg.plot_evolution('lasso')
reg.bias_variance_tradeoff(max_degree=10, n_bootstraps=50)
reg.cross_validation(n_kfolds=2)

plt.show()


# Show the terrain
fig, ax = plt.subplots(figsize=set_size())
ax.set_title('Terrain over Telemark, Norway')
im = ax.imshow(terrain, cmap='terrain')
fig.colorbar(im, label='Elevation [km]', pad=0.02)
ax.set_xlabel('X [arcsec]')
ax.set_ylabel('Y [arcsec]')

fig.tight_layout()
fig.savefig('figures/geo-data.pdf')
fig.savefig('figures/geo-data.png')

X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, terrain.T, cmap='terrain', linewidth=0, antialiased=False)

plt.show()