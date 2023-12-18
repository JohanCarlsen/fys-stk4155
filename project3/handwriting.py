import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '../project2/src')
sys.path.insert(0, '../project1')
sys.path.insert(0, '../')

import autograd.numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from alive_progress import alive_bar
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.neural_network import MLPClassifier

from project2.src.ffnn import NeuralNetwork
from project2.src.logreg import LogisticRegression
from project2.src.preprocess import center, to_categorical, from_categorical
from project1.src import set_size
from project3.src.decision_tree import DecisionTree
from project3.src.classification_metrics import Metrics, confusion_matrix

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

train_data = pd.read_csv('pendigits/pendigits_train.csv')
test_data = pd.read_csv('pendigits/pendigits_test.csv')

X_train = np.array(train_data.iloc[:, :-1])
X_test = np.array(test_data.iloc[:, :-1])
y_train = np.array(train_data.iloc[:, -1])
y_test = np.array(test_data.iloc[:, -1])
classes = np.unique(y_train)

X_train = center(X_train)
X_test = center(X_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_size = X_train.shape[1]

layer_struct = [50]
eta_start = 4
eta_stop = 2
eta_n = int(eta_start - eta_stop + 1)
etas = np.logspace(-eta_start, -eta_stop, eta_n)

alpha_start = 4
alpha_stop = 2
alpha_n = int(alpha_start - alpha_stop + 1)
alphas = np.logspace(-alpha_start, -alpha_stop, alpha_n)

alpha_labels = [r'$10^-$' + f'$^{i}$' for i in range(alpha_start, alpha_stop-1, -1)]
eta_labels = [r'$10^-$' + f'$^{i}$' for i in range(eta_start, eta_stop-1, -1)]

params = {'input_size': input_size, 'hidden_sizes': layer_struct,
          'output_size': 10, 'hidden_activation': 'lrelu',
          'output_activation': 'softmax', 'cost_function': 'cross',
          'epochs': int(1e5), 'batch_size': int(1e3),
          'solver': 'constant', 'variable_eta': False}

logparams = {'n_epochs': int(1e5), 'batch_size': int(1e3),
             'multilabel': True}

losses = np.zeros((eta_n, alpha_n))
accs = np.zeros_like(losses)
tot = int(eta_n * alpha_n)

with alive_bar(tot, title='Processing...', length=20) as bar:
    for i, eta in enumerate(etas):
        for j, alpha in enumerate(alphas):
            NN = NeuralNetwork(eta=eta, alpha=alpha, **params)
            NN.fit(X_train, y_train, X_test, y_test, verbose=False)
            ypred = NN.predict(X_test)
            ypred = to_categorical(ypred, n_categories=10)
            losses[i, j] = NN.cost_func.loss(y_test, ypred)
            accs[i, j] = np.average(ypred == y_test)

            bar()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(scale=1.5),
                               sharex=True)

loss_kw = {'label': 'Cross entropy', 'pad': 0.02}
sns.heatmap(losses, annot=True, cmap='viridis', xticklabels=alpha_labels,
            yticklabels=eta_labels, ax=ax1, cbar_kws=loss_kw)

acc_kw = {'label': 'Avg. Accuracy', 'pad': 0.02}
sns.heatmap(accs, annot=True, cmap='viridis', xticklabels=alpha_labels,
            yticklabels=eta_labels, ax=ax2, cbar_kws=acc_kw)

ax1.set_title('Loss')
ax2.set_title('Score')
ax2.set_xlabel(r'$\alpha$')
fig.supylabel(r'$\eta$', fontsize=8)
fig.savefig('figures/pdfs/nn_heat.pdf')
fig.savefig('figures/nn_heat.png')

losses = np.zeros((eta_n, alpha_n))
accs = np.zeros((eta_n, alpha_n))

for i, eta in enumerate(etas):
    for j, alpha in enumerate(alphas):
        logreg = LogisticRegression(eta=eta, alpha=alpha, **logparams)
        logreg.fit(X_train, y_train, X_test, y_test, verbose=False)
        ypred = logreg.predict(X_test)
        ypred = to_categorical(ypred, n_categories=10)
        losses[i, j] = logreg.loss(y_test, ypred)
        accs[i, j] = np.average(ypred == y_test)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(scale=1.5),
                               sharex=True)

loss_kw = {'label': 'Cross entropy', 'pad': 0.02}
sns.heatmap(losses, annot=True, cmap='viridis', xticklabels=alpha_labels,
            yticklabels=eta_labels, ax=ax1, cbar_kws=loss_kw)

acc_kw = {'label': 'Avg. Accuracy', 'pad': 0.02}
sns.heatmap(accs, annot=True, cmap='viridis', xticklabels=alpha_labels,
            yticklabels=eta_labels, ax=ax2, cbar_kws=acc_kw)

ax1.set_title('Loss')
ax2.set_title('Score')
ax2.set_xlabel(r'$\alpha$')
fig.supylabel(r'$\eta$', fontsize=8)
fig.savefig('figures/pdfs/logreg_heat.pdf')
fig.savefig('figures/logreg_heat.png')

log_eta = 1e-4
log_alpha = 1e-2

skLog = SKLogReg(fit_intercept=False, max_iter=200)
skLog.fit(X_train, from_categorical(y_train))
skPred = skLog.predict(X_test)
sk_metrics = Metrics(from_categorical(y_test), skPred, classes)
print('\nSciKit Results:')
sk_metrics.print_metrics()
'''
SciKit Results:

Mean metrics
------------
Accuracy:    0.90106
Precision:   0.90243
Recall:      0.90133
'''

LogReg = LogisticRegression(eta=log_eta, alpha=log_alpha, **logparams)
LogReg.fit(X_train, y_train, X_test, y_test, verbose=False)
logpred = LogReg.predict(X_test)
ytrue = from_categorical(y_test)

log_metrics = Metrics(ytrue, logpred, classes)
print('\nLogPred results:')
log_metrics.print_metrics()
conf = confusion_matrix(ytrue, logpred)
kw = {'label': 'Frequency', 'pad': 0.02, 'aspect': 15}

fig, ax = plt.subplots(figsize=set_size())
ax.set_title('LogReg confusion matrix')
sns.heatmap(conf, annot=True, cmap='viridis', ax=ax, fmt='.0f',
            cbar_kws=kw)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
fig.savefig('figures/pdfs/logreg_confusion.pdf')
fig.savefig('figures/logreg_confusion.png')

logloss = LogReg.loss_evol
logscore = LogReg.score_evol
logepochs = np.arange(1, len(logloss)+1)

fig, ax = plt.subplots(figsize=set_size())
ax2 = ax.twinx()

logLoss, = ax.plot(logepochs, logloss, label='LogReg loss')
logScore, = ax2.plot(logepochs, logscore, color='red', label='LogReg Score')

ax.set_xlabel('Epochs')
ax.set_ylabel('Cross entropy')
ax2.set_ylabel('Avg. accuracy')
ax.legend(handles=[logLoss, logScore], loc='center right')

ax.grid(False)
ax2.grid(False)
fig.savefig('figures/pdfs/loss_score_logreg.pdf')
fig.savefig('figures/loss_score_logreg.png')

nn_eta = 1e-3
nn_alpha = 1e-3

skNN = MLPClassifier(hidden_layer_sizes=(50), solver='sgd',
                     alpha=nn_alpha, learning_rate_init=nn_eta, momentum=0,
                     batch_size=1000, max_iter=2000)

skNN.fit(X_train, from_categorical(y_train))
skNNPred = skNN.predict(X_test)
skNN_metrics = Metrics(ytrue, skNNPred, classes)
print('\nSciKit results:')
skNN_metrics.print_metrics()
'''
SciKit results:

Mean metrics
------------
Accuracy:    0.87275
Precision:   0.87526
Recall:      0.87504
'''

nn = NeuralNetwork(eta=nn_eta, alpha=nn_alpha, **params)
nn.fit(X_train, y_train, X_test, y_test)
ypred = nn.predict(X_test)
nn_conf = confusion_matrix(ytrue, ypred)
nn_metrics = Metrics(ytrue, ypred, classes)

print('\nNN results:')
nn_metrics.print_metrics()

fig, ax = plt.subplots(figsize=set_size())
ax.set_title('NN confusion matrix')
sns.heatmap(nn_conf, annot=True, cmap='viridis', fmt='.0f',
            cbar_kws=kw)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
fig.savefig('figures/pdfs/NN_confusion.pdf')
fig.savefig('figures/NN_confusion.png')

loss = nn.get_loss_evolution(limit=False)
score = nn.get_score_evolution(limit=False)
epochs = np.arange(1, len(loss)+1)

fig, ax = plt.subplots(figsize=set_size())
ax2 = ax.twinx()

_loss, = ax.plot(epochs, loss, label='NN loss')
_score, = ax2.plot(epochs, score, color='red', label='NN score')

ax.set_xlabel('Epochs')
ax.set_ylabel('Cross entropy')
ax2.set_ylabel('Avg. accuracy')
ax.legend(handles=[_loss, _score], loc='center right')

ax.grid(False)
ax2.grid(False)
fig.savefig('figures/pdfs/loss_score_NN.pdf')
fig.savefig('figures/loss_score_NN.png')

plt.show()