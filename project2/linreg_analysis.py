import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '../project1')

import autograd.numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from alive_progress import alive_bar
import seaborn as sns
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression as skLR
from linreg import RegressionAnalysis
from preprocess import center, norm_data_zero_one
from src import set_size
from calc import Calculate
MSE = Calculate.mean_sq_err

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

def plot_heatmap(dFrame, title=None, ax=None):
    kw = {'label': 'mse', 'pad': 0.02, 'aspect': 20}
    if ax is None:
        fig, ax = plt.subplots(figsize=set_size('text', scale=1.2))
        ax.set_title(title)
    
    if title in ['RidgeGD', 'RidgeSGD']:
        col = 'lamb'

    else:
        col = 'mom'
    
    ax.set_title(title)

    df = dFrame.pivot_table(index='eta', columns=col, values='mse',
                            aggfunc=np.min)
    
    sns.heatmap(df, annot=True, ax=ax, cmap='viridis', cbar_kws=kw)

olsGD = pd.read_csv('OLS-GD.csv')
olsSGD = pd.read_csv('OLS-SGD.csv')
ridgeGD = pd.read_csv('Ridge-GD.csv')
ridgeSGD = pd.read_csv('Ridge-SGD.csv')

adamOlsGD = pd.read_csv('ADAM-OLS-GD.csv')
adamOlsSGD = pd.read_csv('ADAM-OLS-SGD.csv')
adamRidgeGD = pd.read_csv('ADAM-Ridge-GD.csv')
adamRidgeSGD = pd.read_csv('ADAM-Ridge-SGD.csv')

dfs = [olsGD, ridgeGD, olsSGD, ridgeSGD]
adamDFs = [adamOlsGD, adamRidgeGD, adamOlsSGD, adamRidgeSGD]
names = ['OLSGD', 'RidgeGD', 'OLSSGD', 'RidgeSGD']

# fig, axes = plt.subplots(2, 2, figsize=set_size('text', scale=1.3),
#                          sharex='col', sharey='row')

# fig.suptitle(r'Constant eta')
# for df, name, ax in zip(dfs, names, axes.flatten()):
#     plot_heatmap(df, name, ax)

# fig.savefig('figures/pdfs/heat-const.pdf')
# fig.savefig('figures/heat-const.png')

# fig, axes = plt.subplots(2, 2, figsize=set_size('text', scale=1.3),
#                          sharex='col', sharey='row')

# fig.suptitle('ADAM optimizer')
# for df, name, ax in zip(adamDFs, names, axes.flatten()):
#     plot_heatmap(df, name, ax)

# fig.savefig('figures/pdfs/heat-adam.pdf')
# fig.savefig('figures/heat-adam.png')

# for i, df in enumerate(dfs):
#     arr = np.array(df['mse'])
#     idx = np.argwhere(arr == np.min(arr))[0][0]
#     print(f'\n {names[i]}')
#     print(df.loc[[idx], :])

# for i, adamDFs in enumerate(dfs):
#     arr = np.array(adamDFs['mse'])
#     idx = np.argwhere(arr == np.min(arr))[0][0]
#     print(f'\n {names[i]}')
#     print(adamDFs.loc[[idx], :])

def test_func(x):
    a_0 = 1
    a_1 = 0.09
    a_2 = -0.3
    a_3 = 0.1
    # f = a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3
    f = 2 * np.sin(2 * x) + - 0.5 * np.cos(3 * x) + 0.3 * x**3

    return f

n = 100
x = np.linspace(-4, 4, n)[:, np.newaxis]
y_true = norm_data_zero_one(test_func(x))
y = y_true + np.random.normal(0, 0.1, x.shape)

# fig, ax = plt.subplots(figsize=set_size())
# ax.scatter(x, y, s=1, color='black', label='Data')
# ax.plot(x, y_true, label='True')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.legend()

# fig.tight_layout()
# fig.savefig('figures/pdfs/regfunc.pdf')
# fig.savefig('figures/regfunc.png')

X = np.c_[np.ones(n), x, x**2, x**3]
X = center(X)[:, 1:]

optimizer = 'const'
olsGD_params = {'method': 'OLS', 'gradient_descent': 'GD',
                'optimizer': optimizer, 'max_iter': int(1e4)}
olsGD_hyperparams = {'learning_rate': 0.0001, 'momentum': 0.9}

olsSGD_params = {'method': 'OLS', 'gradient_descent': 'SGD',
                 'optimizer': optimizer, 'max_iter': int(1e4), 'n_epochs': 10,
                 'minibatch_size': 65}
olsSGD_hyperparams = {'learning_rate': 0.01, 'momentum': 0.9}

ridgeGD_params = {'method': 'Ridge', 'gradient_descent': 'GD',
                  'optimizer': optimizer, 'max_iter': int(1e4)}
ridgeGD_hyperparams = {'learning_rate': 0.01, 'momentum': 0.9,
                       'lamb': 0.001}

ridgeSGD_params = {'method': 'Ridge', 'gradient_descent':'SGD',
                   'optimizer': optimizer, 'max_iter': int(1e4), 'n_epochs': 10,
                   'minibatch_size': 65}
ridgeSGD_hyperparams = {'learning_rate': 0.001, 'momentum': 0.9,
                        'lamb': 0.001}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = RegressionAnalysis(X, y)
reg.set_params(**olsGD_params)
reg.set_hyper_params(**olsGD_hyperparams)
reg.run()
olsGD_score = reg.get_score_evol()
olsGD_beta = reg.best_beta

skreg = skLR(fit_intercept=False)
skreg.fit(X_train, y_train)

olsGDpred = X_test @ olsGD_beta
olsSKpred = skreg.predict(X_test)

reg = RegressionAnalysis(X, y)
reg.set_params(**olsSGD_params)
reg.set_hyper_params(**olsSGD_hyperparams)
reg.run()
olsSGD_score = reg.get_score_evol()
olsSGD_beta = reg.best_beta

olsSGDpred = X_test @ olsSGD_beta

reg = RegressionAnalysis(X, y)
reg.set_params(**ridgeGD_params)
reg.set_hyper_params(**ridgeGD_hyperparams)
reg.run()
ridgeGD_score = reg.get_score_evol()
ridgeGD_beta = reg.best_beta

skreg = Ridge(alpha=ridgeGD_hyperparams['lamb'], fit_intercept=False)
skreg.fit(X_train, y_train)

ridgeGDpred = X_test @ ridgeGD_beta
ridgeSKpred = skreg.predict(X_test)

reg = RegressionAnalysis(X, y)
reg.set_params(**ridgeSGD_params)
reg.set_hyper_params(**ridgeSGD_hyperparams)
reg.run()
ridgeSGD_score = reg.get_score_evol()
ridgeSGD_beta = reg.best_beta

ridgeSGDpred = X_test @ ridgeSGD_beta

preds = [[olsGDpred, olsSKpred], [olsSGDpred, olsSKpred],
         [ridgeGDpred, ridgeSKpred], [ridgeSGDpred, ridgeSKpred]]
model = ['OLSGD', 'OLSSGD', 'RidgeGD', 'RidgeSGD']

print(f"{'Model':<13}{'Own':<10}{'Sklearn':<10}")
i = 0
for own, sk in preds:
    mse_own = f'{MSE(y_test, own):.5f}'
    mse_sk = f'{MSE(y_test, sk):.5f}'
    print(f"{model[i]:<13}{mse_own:<10}{mse_sk:<10}")
    i += 1

scores = [olsGD_score, olsSGD_score, ridgeGD_score, ridgeSGD_score]
betas = [olsGD_beta, olsSGD_beta, ridgeGD_beta, ridgeSGD_beta]
names = ['OLSGD', 'OLSSGD', 'RidgeGD', 'RidgeSGD']

fig, ax = plt.subplots(figsize=set_size())
for name, score in zip(names, scores):
    n_its = len(score)
    _x = np.linspace(0, 1, n_its)
    ax.plot(_x, score, label=name + f': {int(n_its)}')

ax.set_xlabel(r'Normalized no. iterations')
ax.set_ylabel(r'MSE')
ax.legend()

if optimizer == 'const':
    ax.set_yscale('log')

fig.savefig('figures/pdfs/' + optimizer + 'linreg-conv.pdf')
fig.savefig('figures/' + optimizer + 'linreg-conv.png')

plt.show()

