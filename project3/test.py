import sys
sys.path.insert(0, '../project1')
sys.path.insert(0, '../project2/src')
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

layer_struct = [10, 35, 350]
# layer_struct = [10, 50, 350]
epochs, batch = int(1e4), 100
params = {'input_size': input_size, 'hidden_sizes': layer_struct,
          'output_size': 1, 'hidden_activation': 'lrelu',
          'output_activation': 'sigmoid', 'cost_function': 'log',
          'epochs': epochs, 'batch_size': batch, 'solver': 'constant'}

eta = 1e-4
alpha = 1e-3

NN = NeuralNetwork(**params, eta=eta, alpha=alpha, variable_eta=False)
NN.fit(X_train, y_train, X_test, y_test, verbose=False)

y = NN.get_score_evolution(limit=False)
y2 = NN.get_loss_evolution(limit=False)

ypred = NN.predict(X_test)
acc = NN.calculate_score(y_test, ypred)
print(f'Accuracy: {acc}')

x = np.arange(1, len(y)+1)
fig, ax = plt.subplots()
line, = ax.plot(x, y, label='Accuracy')
ax2 = ax.twinx()
line2, = ax2.plot(x, y2, color='red', label='Loss')
ax.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
ax.set_xlabel('Epochs')
ax.legend(handles=[line, line2])
ax.grid(False)
ax2.grid(False)

plt.show()