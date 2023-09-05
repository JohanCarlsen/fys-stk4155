import numpy as np 
from numpy.random import rand
import matplotlib.pyplot as plt 
from src import Regression, frankes_function

np.random.seed(2018)

n = 100
x, y = rand(2, n)
X, Y = np.meshgrid(x, y)
z = frankes_function(X, Y)

reg = Regression(x, y, z)
reg.OLS(5)
reg.plot_evolution('OLS', 'franke-test')
plt.show()