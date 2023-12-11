import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '../project2/src')
sys.path.insert(0, '../project1')
sys.path.insert(0, '../')

import autograd.numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import seaborn as sns 
import pandas as pd 
from alive_progress import alive_bar

from project2.src.preprocess import center
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

tree = DecisionTree(max_depth=10)
tree.fit(X_train, y_train)
ypred = np.int64(tree.predict(X_test))

metrics = Metrics(y_test, ypred, classes)
metrics.print_metrics()

conf = confusion_matrix(y_test, ypred)

fig, ax = plt.subplots(figsize=set_size())
ax.set_title('Decision tree confusion matrix')
sns.heatmap(conf, annot=True, cmap='viridis', fmt='.0f',
            cbar_kws={'label': 'Frequency', 'pad': 0.02, 'aspect': 15})
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
fig.savefig('figures/pdfs/tree_confusion.pdf')
fig.savefig('figures/tree_confusion.png')

depths = np.arange(5, 26, 5)
classes = np.unique(y_train)
n_classes = len(classes)
n = len(depths)

gini_accs = np.zeros(n)
gini_mean_prec = np.zeros(n)
gini_mean_recalls = np.zeros(n)
gini_precs = np.zeros((n, n_classes))
gini_recalls = np.zeros((n, n_classes))

entropy_accs = np.zeros(n)
entropy_mean_prec = np.zeros(n)
entropy_mean_recalls = np.zeros(n)
entropy_precs = np.zeros((n, n_classes))
entropy_recalls = np.zeros((n, n_classes))

tot = len(depths)
with alive_bar(tot, title='Processing...', length=20) as bar:
    for i, depth in enumerate(depths):
        gini_tree = DecisionTree(max_depth=depth, loss='gini')
        entropy_tree = DecisionTree(max_depth=depth, loss='entropy')

        gini_tree.fit(X_train, y_train)
        entropy_tree.fit(X_train, y_train)

        gini_pred = gini_tree.predict(X_test)
        entropy_pred = entropy_tree.predict(X_test)

        gini_metric = Metrics(y_test, gini_pred, classes)
        gini_accs[i] = gini_metric.accuracy
        gini_precs[i, :] = gini_metric.precision
        gini_recalls[i, :] = gini_metric.recall
        gini_mean_prec[i] = gini_metric.mean_precision
        gini_mean_prec[i] = gini_metric.mean_recall

        entropy_metric = Metrics(y_test, entropy_pred, classes)
        entropy_accs[i] = entropy_metric.accuracy
        entropy_precs[i, :] = entropy_metric.precision
        entropy_recalls[i, :] = entropy_metric.recall
        entropy_mean_prec[i] = entropy_metric.mean_precision
        entropy_mean_recalls[i] = entropy_metric.mean_recall

        bar()

np.save('gini_accuracy', gini_accs)
np.save('gini_mean_precision', gini_mean_prec)
np.save('gini_mean_recall', gini_mean_recalls)
np.save('gini_recalls', gini_recalls)
np.save('gini_precisions', gini_precs)


np.save('entropy_accuracy', entropy_accs)
np.save('entropy_mean_precision', entropy_mean_prec)
np.save('entropy_mean_recall', entropy_mean_recalls)
np.save('entropy_recalls', entropy_recalls)
np.save('entropy_precisions', entropy_precs)

gini_accs = np.load('gini_accuracy.npy')
gini_mean_precs = np.load('gini_mean_precision.npy')
gini_mean_recs = np.load('gini_mean_precision.npy')
gini_precs = np.load('gini_precisions.npy')
gini_recalls = np.load('gini_recalls.npy')

entropy_accs = np.load('entropy_accuracy.npy')
entropy_mean_precs = np.load('entropy_mean_precision.npy')
entropy_mean_recs = np.load('entropy_mean_precision.npy')
entropy_precs = np.load('entropy_precisions.npy')
entropy_recalls = np.load('entropy_recalls.npy')

fig, ax = plt.subplots(figsize=set_size())
ax.plot(depths, gini_accs, label='Gini')
ax.plot(depths, entropy_accs, label='Entropy')
ax.set_xlabel('Max. depth')
ax.set_ylabel('Accuracy')
ax.legend()
fig.savefig('figures/pdfs/tree_accuracy.pdf')
fig.savefig('figures/tree_accuracy.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=set_size('text'))
ax1.set_title('Gini')
ax1.plot(depths, gini_recalls, label=classes)

ax1.plot(depths, gini_mean_recs, lw=2, ls='dashed',
         color='black', label='Mean')

ax1.set_ylabel('Recall')

ax2.set_title('Entropy')
ax2.plot(depths, entropy_recalls, label=classes)

ax2.plot(depths, entropy_mean_recs, lw=2, ls='dashed',
         color='black', label='Mean')

ax2.legend(loc='upper left', bbox_to_anchor=[1, 1])
fig.supxlabel('Max. depth', fontsize=8)
fig.savefig('figures/pdfs/tree_recall.pdf')
fig.savefig('figures/tree_recall.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=set_size('text'))
ax1.set_title('Gini')
ax1.plot(depths, gini_precs, label=classes)

ax1.plot(depths, gini_mean_precs, lw=2, ls='dashed',
         color='black', label='Mean')

ax1.set_ylabel('Precision')

ax2.set_title('Entropy')
ax2.plot(depths, entropy_precs, label=classes)

ax2.plot(depths, entropy_mean_precs, lw=2, ls='dashed',
         color='black', label='Mean')

ax2.legend(loc='upper left', bbox_to_anchor=[1, 1])
fig.supxlabel('Max. depth', fontsize=8)
fig.savefig('figures/pdfs/tree_precision.pdf')
fig.savefig('figures/tree_precision.png')

plt.show()