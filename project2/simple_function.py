import sys
sys.path.insert(0, 'src')

import autograd.numpy as np 
from autograd import grad, elementwise_grad
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from alive_progress import alive_bar
import seaborn as sns

from ffnn import NeuralNetwork
from cost_funcs import MeanSquaredError
from preprocess import center, norm_data_zero_one

sns.set_theme()
np.random.seed(2023)

def test_func(x):
    a_0 = 1
    a_1 = 0.09
    a_2 = -0.3
    a_3 = 0.1
    f = a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3
    # f = 2 * np.sin(2 * x) + - 0.5 * np.cos(3 * x) + 0.3 * x**3

    return f

n = 1001
x = np.linspace(-4, 4, n)[:, np.newaxis]
y_true = norm_data_zero_one(test_func(x))
y = y_true + 0.1 * np.random.normal(0, 1, x.shape)

X = x.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = center(X_train)
X_test = center(X_test)
X = center(X)

alpha_start = 5
alpha_stop = 2
alpha_n = int(alpha_start - alpha_stop + 1)

eta_start = 6
eta_stop = 3
eta_n = int(eta_start - eta_stop + 1)

alphas = np.logspace(-alpha_start, -alpha_stop, alpha_n)
etas = np.logspace(-eta_start, -eta_stop, eta_n)
alpha_labels = [r'$10^-$' + f'$^{i}$' for i in range(alpha_start, alpha_stop-1, -1)]
eta_labels = [r'$10^-$' + f'$^{i}$' for i in range(eta_start, eta_stop-1, -1)]

MSEs = np.zeros((len(alphas), len(etas)))
layer_struct = [100]

params = {'input_size': 1, 'hidden_sizes': layer_struct, 'output_size': 1,
          'hidden_activation': 'lrelu', 'output_activation': 'linear',
          'cost_function': 'mse', 'epochs': int(5e2), 'batch_size': 100,
          'solver': 'adam'}

tot = len(alphas) * len(etas)
best_alpha = 0
best_eta = 0
best_mse = np.inf

with alive_bar(tot, length=20, title='Processing...') as bar:
    for i in range(len(alphas)):
        for j in range(len(etas)):
            alpha = alphas[i]
            eta = etas[j]

            NN = NeuralNetwork(eta=eta, alpha=alpha, **params)
            NN.fit(X_train, y_train, X_test, y_test)
            ypred = NN.predict(X_test)
            mse = MeanSquaredError.loss(y_test, ypred)

            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha 
                best_eta = eta

            MSEs[i, j] = mse 
            bar()

fig, ax = plt.subplots()

sns.heatmap(MSEs, annot=True, ax=ax, cmap='viridis',
            cbar_kws={'label': 'MSE'}, xticklabels=eta_labels,
            yticklabels=alpha_labels)

ax.set_title('Test MSEs')
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\alpha$')

NN = NeuralNetwork(eta=best_eta, alpha=best_alpha, **params)
NN.fit(X_train, y_train, X_test, y_test)
ypred = NN.predict(X)
mse = MeanSquaredError.loss(y, ypred)

fig, ax = plt.subplots()
ax.set_title(f'MSE own: {mse:.2e}')
ax.scatter(x, y, color='black', s=0.25, label='Data', alpha=0.75)
ax.plot(x, y_true, color='blue', ls='dashed', label='Target')
ax.plot(x, ypred, color='red', label='FFNN')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()
plt.show()


    

