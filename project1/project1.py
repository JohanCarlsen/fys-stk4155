import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from src import Regression, frankes_function

np.random.seed(2018)

n = 500
x, y = rand(2, n)
X, Y = np.meshgrid(x, y)
z = frankes_function(x, y)

reg = Regression(x, y, z)
# reg.OLS(5)
# reg.plot_evolution('OLS', 'franke-test')
# reg.ridge(-4, 4, 5, 1000)
# reg.plot_evolution('ridge', 'franke-test')
# reg.plot_evolution('lasso', 'franke-test')
reg.bias_variance_tradeoff(max_degree=15, n_bootstraps=100)
plt.show()