import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '../project1')
sys.path.insert(0, '../project1/props')

import autograd.numpy as np 
from autograd import grad, elementwise_grad
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt 
from alive_progress import alive_bar
import seaborn as sns

from ffnn import NeuralNetwork
from cost_funcs import MeanSquaredError
from preprocess import center, norm_data_zero_one
from src import set_size

sns.set_theme()
np.random.seed(2023)

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'savefig.bbox': 'tight',
})

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

fig, ax = plt.subplots(figsize=set_size())
ax.scatter(x, y, s=0.01, color='black', label='Data')
ax.plot(x, y_true, label='True')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()

fig.tight_layout()
fig.savefig('figures/pdfs/simplefunc.pdf')
fig.savefig('figures/simplefunc.png')

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
tot = len(alphas) * len(etas)

alpha_labels = [r'$10^-$' + f'$^{i}$' for i in range(alpha_start, alpha_stop-1, -1)]
eta_labels = [r'$10^-$' + f'$^{i}$' for i in range(eta_start, eta_stop-1, -1)]

MSEs = np.zeros((len(alphas), len(etas)))
layer_struct = [100]

fig, axes = plt.subplots(2, 1, figsize=set_size(scale=2), sharex=True)
fig2, ax2 = plt.subplots(figsize=set_size())
kw = {'label': 'MSE', 'pad': 0.02}
axes = axes.flatten()
solvers = ['constant', 'adam']

for ax, solver in zip(axes, solvers):
    params = {'input_size': 1, 'hidden_sizes': layer_struct, 'output_size': 1,
            'hidden_activation': 'lrelu', 'output_activation': 'linear',
            'cost_function': 'mse', 'epochs': int(2.5e2), 'batch_size': 100,
            'solver': solver, 'variable_eta': False}

    best_alpha = 0
    best_eta = 0
    best_mse = np.inf

    with alive_bar(tot, length=20, title='Processing...') as bar:
        for i in range(len(alphas)):
            for j in range(len(etas)):
                alpha = alphas[i]
                eta = etas[j]

                NN = NeuralNetwork(eta=eta, alpha=alpha, **params)
                NN.fit(X_train, y_train, X_test, y_test, verbose=False)
                ypred = NN.predict(X_test)
                mse = MeanSquaredError.loss(y_test, ypred)

                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha 
                    best_eta = eta

                MSEs[i, j] = mse 
                bar()

    if solver == 'constant':
        const_eta = best_eta
        const_alpha = best_alpha

    print(f'\n{solver}')
    print(f'Best eta:   {best_eta}')
    print(f'Best alpha: {best_alpha}')

    sns.heatmap(MSEs, annot=True, ax=ax, cmap='viridis',
                cbar_kws=kw, xticklabels=eta_labels,
                yticklabels=alpha_labels)

    ax.set_title(solver)
    xlabel = r'$\eta$' if solver == 'adam' else ''
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\alpha$')

    NN = NeuralNetwork(eta=best_eta, alpha=best_alpha, **params)
    NN.fit(X_train, y_train, X_test, y_test)
    ypred = NN.predict(X_test)
    mse = MeanSquaredError.loss(y_test, ypred)
    print(f'Final test MSE for {solver}: {mse:.5f}')
    evol = NN.get_score_evolution()
    n_iter = len(evol)
    x = np.arange(1, n_iter+1)
    label = solver
    ax2.plot(x, evol, label=label)

params = {'input_size': 1, 'hidden_sizes': layer_struct, 'output_size': 1,
        'hidden_activation': 'lrelu', 'output_activation': 'linear',
        'cost_function': 'mse', 'epochs': int(2.5e2), 'batch_size': 100,
        'solver': 'constant', 'variable_eta': True}

NN = NeuralNetwork(eta=const_eta, alpha=const_alpha, **params)
NN.fit(X_train, y_train, X_test, y_test, verbose=False)
ypred = NN.predict(X_test)
mse = MeanSquaredError.loss(y_test, ypred)
print('\nvar eta')
print(mse)
evol = NN.get_score_evolution()
n_iter = len(evol)
x = np.arange(1, n_iter+1)

ax2.plot(x, evol, label=r'var. $\eta$ ')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MSE')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()

fig.savefig('figures/pdfs/ann-reg.pdf')
fig.savefig('figures/ann-reg.png')

fig2.savefig('figures/pdfs/ann-reg-conv.pdf')
fig2.savefig('figures/ann-reg-conv.png')

print('\n-----------')
print('SKL results')
print('-----------')
nn = MLPRegressor([100], solver='sgd', alpha=1e-3, batch_size=100,
                  learning_rate_init=1e-3, momentum=0, max_iter=250)

nn.fit(X_train, y_train.ravel())
ypred = nn.predict(X_test)
mse = MeanSquaredError.loss(y_test, ypred)
print('\nconstant')
print(mse)

nn = MLPRegressor([100], learning_rate_init=1e-4, alpha=1e-4,
                  momentum=0, batch_size=100, max_iter=250)

nn.fit(X_train, y_train.ravel())
ypred = nn.predict(X_test)
mse = MeanSquaredError.loss(y_test, ypred)
print('\nadam')
print(mse)

nn = MLPRegressor([100], solver='sgd', alpha=1e-3, batch_size=100,
                  learning_rate='invscaling', learning_rate_init=1e-3,
                  momentum=0, max_iter=250)

nn.fit(X_train, y_train.ravel())
ypred = nn.predict(X_test)
mse = MeanSquaredError.loss(y_test, ypred)
print('\nvar eta')
print(mse)

plt.show()


    

