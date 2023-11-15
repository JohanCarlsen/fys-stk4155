import sys
sys.path.insert(0, 'src')
import autograd.numpy as np 
from ffnn import NeuralNetwork
from cost_funcs import MeanSquaredError
from preprocess import center, norm_data_zero_one
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_theme()
np.random.seed(2023)

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])

yOR = np.array([0, 1, 1, 1])[:, np.newaxis]
yAND = np.array([0, 0, 0, 1])[:, np.newaxis]
yXOR = np.array([0, 1, 1, 0])[:, np.newaxis]

X = np.column_stack([x1, x2])

params = {'input_size': 2, 'hidden_sizes': [2],
          'output_size': 1, 'hidden_activation': 'sigmoid',
          'output_activation': 'sigmoid', 'cost_function': 'cross',
          'epochs': 100, 'batch_size': 4, 'solver': 'adam'}

etas = np.logspace(-5, 1, 7)
alphas = np.logspace(-5, 1, 7)
accs = np.zeros((len(etas), len(alphas)))

for i, eta in enumerate(etas):
    for j, alpha in enumerate(alphas):
        nn = NeuralNetwork(**params, eta=eta, alpha=alpha)
        nn.fit(X, yOR)
        ypred = nn.predict(X)
        acc = np.sum(ypred == yOR) / len(yOR)
        accs[i, j] = acc 
        print(f'Learning rate: {eta}')
        print(f'Alpha: {alpha}')
        print(f'Accuracy: {acc}\n')

sns.heatmap(accs, annot=True, cmap='viridis')
plt.show()