import sys
sys.path.insert(0, '../../project2/src')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import autograd.numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from alive_progress import alive_bar
import seaborn as sns

from project2.src.ffnn import NeuralNetwork
from project2.src.logreg import LogisticRegression
from project2.src.preprocess import center, to_categorical
from project1.src import set_size

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

forest = load_iris()
X = forest.data
y = forest.target
input_size = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = center(X_train)
X_test = center(X_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


layer_struct = [15]
params = {'input_size': input_size, 'hidden_sizes': layer_struct,
          'output_size': 3, 'eta': 1e-4, 'alpha': 1e-2,
          'hidden_activation': 'relu', 'output_activation': 'softmax',
          'cost_function': 'cross', 'epochs': int(1e2),
          'batch_size': 50, 'solver': 'constant', 'variable_eta': False}

NN = NeuralNetwork(**params)
NN.fit(X_train, y_train, X_test, y_test)
ypred = NN.predict(X_test)
ypred = to_categorical(ypred)

print(np.average(ypred == y_test))
fig, ax = plt.subplots()
y = NN.get_loss_evolution(limit=False)
y2 = NN.get_score_evolution(limit=False)
x = np.arange(1, len(y)+1)
line, = ax.plot(x, y, label='Loss')
ax2 = ax.twinx()
line2, = ax2.plot(x, y2, color='red', label='Score')
ax.legend(handles=[line, line2])
ax.set_ylabel('Loss')
ax2.set_ylabel('Avg. accuracy')
ax.set_xlabel('Epoch')
ax.grid(False)
ax2.grid(False)

plt.show()
