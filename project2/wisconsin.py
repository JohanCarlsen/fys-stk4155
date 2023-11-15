import sys
sys.path.insert(0, 'src')

import autograd.numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt 
from alive_progress import alive_bar
import seaborn as sns

from ffnn import NeuralNetwork
from preprocess import center

sns.set_theme()
np.random.seed(2023)

wdbc = load_breast_cancer()

X = wdbc.data 
y = wdbc.target[:, np.newaxis]

_, input_size = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = center(X_train)
X_test = center(X_test)

layer_struct = [100]

params = {'input_size': input_size, 'hidden_sizes': layer_struct,
          'output_size': 1, 'hidden_activation': 'sigmoid',
          'output_activation': 'sigmoid', 'cost_function': 'log',
          'epochs': int(5e2), 'batch_size': 100, 'solver': 'adam'}

start = 7
stop = 2
n = int(start - stop + 1)

etas = np.logspace(-start, -stop, n)
alphas = np.logspace(-start, -stop, n)

alpha_labels = [r'$10^-$' + f'$^{i}$' for i in range(start, stop-1, -1)]
eta_labels = [r'$10^-$' + f'$^{i}$' for i in range(start, stop-1, -1)]

accs = np.zeros((len(etas), len(alphas)))

for i, eta in enumerate(etas):
    for j, alpha in enumerate(alphas):
        NN = NeuralNetwork(**params, eta=eta, alpha=alpha)
        NN.fit(X_train, y_train, X_test, y_test, verbose=False)
        ypred = NN.predict(X_test)
        acc = NN.calculate_score(y_test, ypred)
        accs[i, j] = acc 

best_idx = np.argwhere(accs == np.max(accs))[0]
best_eta = etas[best_idx[0]]
best_alpha = alphas[best_idx[1]]

NN = NeuralNetwork(**params, eta=best_eta, alpha=best_alpha)
NN.fit(X_train, y_train, X_test, y_test)

fig, ax = plt.subplots()

sns.heatmap(accs, annot=True, ax=ax, cmap='viridis',
            xticklabels=alpha_labels, yticklabels=eta_labels,
            cbar_kws={'label': 'Accuracy'})

ax.set_title('Test accuracy')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\eta$')
fig.tight_layout()
plt.show()
