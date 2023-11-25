import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '../project1')
sys.path.insert(0, '../project1/props')

import autograd.numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from alive_progress import alive_bar
import seaborn as sns

from ffnn import NeuralNetwork
from logreg import LogisticRegression
from preprocess import center
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

wdbc = load_breast_cancer()

X = wdbc.data 
y = wdbc.target[:, np.newaxis]

_, input_size = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = center(X_train)
X_test = center(X_test)

layer_struct = [100]
epochs, batch = int(1e4), 100
params = {'input_size': input_size, 'hidden_sizes': layer_struct,
          'output_size': 1, 'hidden_activation': 'sigmoid',
          'output_activation': 'sigmoid', 'cost_function': 'log',
          'epochs': epochs, 'batch_size': batch, 'solver': 'constant'}

start = 6
stop = 1
n = int(start - stop + 1)

etas = np.logspace(-start, -stop, n)
alphas = np.logspace(-start, -stop, n)

alpha_labels = [r'$10^-$' + f'$^{i}$' for i in range(start, stop-1, -1)]
eta_labels = [r'$10^-$' + f'$^{i}$' for i in range(start, stop-1, -1)]

accs = np.zeros((len(etas), len(alphas)))
accs_logreg = np.zeros_like(accs)

for i, eta in enumerate(etas):
    for j, alpha in enumerate(alphas):
        NN = NeuralNetwork(**params, eta=eta, alpha=alpha, variable_eta=False)
        NN.fit(X_train, y_train, X_test, y_test, verbose=False)
        ypred = NN.predict(X_test)
        acc = NN.calculate_score(y_test, ypred)
        accs[i, j] = acc 

        logreg = LogisticRegression(eta=eta, alpha=alpha, n_epochs=epochs,
                                    batch_size=batch, tol=0.5, constant_eta=False)
        
        logreg.fit(X_train, y_train, X_test, y_test, verbose=False)
        logpred = logreg.predict(X_test)
        score = np.sum(logpred == y_test) / y_test.size
        accs_logreg[i, j] = score


best_idx = np.argwhere(accs == np.max(accs))[0]
best_eta = etas[best_idx[0]]
best_alpha = alphas[best_idx[1]]
best_idx_logreg = np.argwhere(accs_logreg == np.max(accs_logreg))[0]
best_eta_logreg = etas[best_idx_logreg[0]]
best_alpha_logreg = etas[best_idx_logreg[1]]

print('\nANN')
print(f'Best eta: {best_eta}')
print(f'Best alpha: {best_alpha}')
print('\nLogReg')
print(f'Best eta: {best_eta_logreg}')
print(f'Best alpha: {best_alpha_logreg}')


kw = {'label': 'Accuracy', 'pad': 0.02}
fig, (ax, ax2) = plt.subplots(2, 1, figsize=set_size(scale=2), sharex=True)

sns.heatmap(accs, annot=True, ax=ax, cmap='viridis',
            xticklabels=alpha_labels, yticklabels=eta_labels,
            cbar_kws=kw)

sns.heatmap(accs_logreg, annot=True, ax=ax2, cmap='viridis',
            xticklabels=alpha_labels, yticklabels=eta_labels,
            cbar_kws=kw)

ax.set_title('ANN')
ax2.set_title('LogReg')
ax2.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\eta$')
ax2.set_ylabel(r'$\eta$')
fig.savefig('figures/pdfs/cancer.pdf')
fig.savefig('figures/cancer.png')

NN = NeuralNetwork(**params, eta=best_eta, alpha=best_alpha, variable_eta=False)
NN.fit(X_train, y_train, X_test, y_test)
y = NN.get_score_evolution(limit=False)

logreg = LogisticRegression(eta=best_eta_logreg, alpha=best_alpha_logreg,
                            n_epochs=epochs, batch_size=batch, tol=0.5,
                            constant_eta=False)

logreg.fit(X_train, y_train, X_test, y_test, verbose=False)
ylog = logreg.score_evol
stop = np.argwhere(ylog == np.max(ylog))[0][0]
ylog = ylog[:stop+1]

sknn = MLPClassifier(layer_struct, activation='logistic', solver='sgd',
                     batch_size=batch, max_iter=epochs, alpha=best_alpha,
                     learning_rate_init=best_eta, momentum=0)

sknn.fit(X_train, y_train.ravel())
skscore = sknn.score(X_test, y_test)
print(f'\nSciKit-Learn acc: {skscore:.4f}\n')

names = ['ANN', 'LogReg']
fig, ax = plt.subplots(figsize=set_size())

for i, evol in enumerate([y, ylog]):
    print(f'{names[i]}: {evol[-1]}')
    print(f'{names[i]}: {np.all(evol == evol[-1])}')
    n_iter = len(evol)
    x = np.linspace(0, 1, n_iter)
    ax.plot(x, evol, label=names[i] + f' {n_iter:.0f}')

    i += 1

ax.set_xlabel('Normalized no. epochs')
ax.set_ylabel('Accuracy')
ax.legend()

fig.savefig('figures/pdfs/classify-conv.pdf')
fig.savefig('figures/classify-conv.png')

plt.show()
