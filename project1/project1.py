import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from src import Regression, frankes_function

np.random.seed(2018)

n = 500
x, y = rand(2, n)
z = frankes_function(x, y)

reg = Regression(x, y, z, 'franke')
reg.OLS(8)
reg.plot_evolution('OLS')
reg.ridge(-4, 2, 5, 1000)
reg.plot_evolution('ridge')
reg.plot_evolution('lasso')
reg.bias_variance_tradeoff(max_degree=15, n_bootstraps=100)
reg.cross_validation(n_kfolds=10)
plt.show()