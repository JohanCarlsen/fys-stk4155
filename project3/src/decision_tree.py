'''
This program is highly based on the DesicionTreeClassifier written 
by Michael Attard.
(https://insidelearningmachines.com/build-a-decision-tree-in-python/)
'''
import numpy as np 
from scipy import stats
from classification_metrics import *
from graphviz import Digraph

class Node(object):
    r'''
    Class representing a tree node. 
    '''
    def __init__(self):
        r'''
        Contructor method.
        '''
        self.split = None 
        self.feature = None
        self.left_node = None 
        self.right_node = None 
        self.leaf_value = None 

    def set_params(self, split, feature):
        r'''
        Set the parameters of the node.

        Parameters
        ----------
        split : float
            Value to make the split on.

        feature : int
            Index of the feature to make the split on.
        '''
        self.split = split 
        self.feature = feature 

    def get_params(self):
        r'''
        Get the parametrs of the node. 
        '''
        return self.split, self.feature
    
    def set_children(self, left_node, right_node):
        r'''
        Create childen nodes to the left and right of the current node.

        Parameters
        ----------
        left_node, right_node : Node
            Childen nodes to the left and right.
        '''
        self.left_node = left_node
        self.right_node = right_node

    def get_left_node(self):
        r'''
        Get the node to the left.
        '''
        return self.left_node
    
    def get_right_node(self):
        r'''
        Get the node to the right.
        '''
        return self.right_node
    
class DecisionTree:
    r'''
    Decision tree for classification.

    Parameters
    ----------
    max_depth : int, default ``None``
        The maximum number of subtrees to grow during training.

    min_split : int, default 2
        Minimum number of samples needed to split a node.

    loss : {'entropy', 'gini'}, default 'entropy'
        Loss function to use. 
    '''
    def __init__(self, max_depth=None, min_split=2, loss='entropy'):
        r'''
        Contructor for the DecisionTree class.

        Parameters
        ----------
        max_depth : int, default ``None``
            The maximum number of subtrees to grow during training.

        min_split : int, default 2
            Minimum number of samples needed to split a node.

        loss : {'entropy', 'gini'}, default 'entropy'
            Loss function to use. 
        '''
        self.tree = None 
        self.max_depth = max_depth
        self.min_split = min_split

        if loss == 'entropy':
            self.loss = self._entropy
        
        elif loss == 'gini':
            self.loss = self._gini

        self.struct = []
        self.leaf_vals = []
    
    def _grow_tree(self, node, node_data, depth_level):
        r'''
        Grow the tree.

        Parameters
        ----------
        node : Node
            Input tree node.

        node_data : ndarray
            Data for the input node.

        depth_level : int 
            Depth level in tree for input node.
        '''
        depth = False
        if self.max_depth is None or self.max_depth >= depth_level + 1:
            depth = True 

        min_split = False
        if self.min_split <= node_data.shape[0]:
            min_split = True 

        n_classes = False 
        if np.unique(node_data[:, -1].shape[0]) != 1:
            n_classes = True 

        if depth and min_split and n_classes:
            impure_node = None 
            feature = None 
            split = None 
            left_data = None 
            right_data = None 

            for i in range(node_data.shape[-1] -1):
                for temp_split in np.unique(node_data[:, i]):
                    inds = node_data[:, i] <= temp_split
                    temp_left = node_data[inds]
                    temp_right = node_data[~inds]
                   
                    if temp_left.size and temp_right.size:
                        impurity = temp_left.shape[0] / node_data.shape[0] \
                                 * self._impurity(temp_left) \
                                 + temp_right.shape[0] / node_data.shape[0] \
                                 * self._impurity(temp_right)
                        
                        if impure_node is None or impurity < impure_node:
                            impure_node = impurity
                            feature = i 
                            split = temp_split 
                            left_data = temp_left 
                            right_data = temp_right

            self.struct.append([depth_level, split, feature])
            node.set_params(split, feature)
            left_node = Node()
            right_node = Node()
            node.set_children(left_node, right_node)

            self._grow_tree(node.get_left_node(), left_data, depth_level+1)
            self._grow_tree(node.get_right_node(), right_data, depth_level+1)
            
        else:
            node.leaf_value = self._leaf_value(node_data)
            self.leaf_vals.append(node.leaf_value)

            return 
        
    def _traverse(self, node, row):
        r'''
        Traverse through the tree.

        Parameters
        ----------
        node : Node
            Current tree node.

        row : ndarray
            Data for the node.

        Returns
        -------
        leaf_value : float
            The value of the node.
        '''
        if node.leaf_value is None:
            split, feature = node.get_params()

            if row[feature] <= split:
                return self._traverse(node.get_left_node(), row)
            
            else: 
                return self._traverse(node.get_right_node(), row)
            
        else:
            return node.leaf_value
        
    def fit(self, X, y):
        r'''
        Train the tree on the input data.

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape `(n_data, n_features)`.

        y : array_like
            Target values of shape `(n_data,)`
        '''
        data = np.c_[X, y]
        self.tree = Node()
        self._grow_tree(self.tree, data, depth_level=1)

        self.struct = np.c_[self.struct, self.leaf_vals[1:]]

    def predict(self, X):
        r'''
        Make a prediction with a trained tree.

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape `(n_data, n_features)`

        Returns
        -------
        array_like : 
            The predictions from traverseing the tree.
        '''
        preds = []
        for i in range(X.shape[0]):
            preds.append(self._traverse(self.tree, X[i, :]))

        return np.array(preds).flatten()
    
    def _gini(self, data):
        r'''
        Gini impurity.

        Parameters
        ----------
        data : ndarray
            Data to calculate the Gini impurity on.

        Returns
        -------
        gini : float 
            The impurity using Gini.
        '''
        gini = 0
        for i in np.unique(data[:, -1]):
            inds = data[:, -1] == i
            p = data[inds].shape[0] / data.shape[0]
            gini += p * (1 - p)

        return gini
    
    def _entropy(self, data):
        r'''
        Entropy impurity.

        Parameters
        ----------
        data : ndarray
            Data to calculate the entropy impurity on.

        Returns
        -------
        gini : float 
            The impurity using entropy.
        '''
        entropy = 0
        for i in np.unique(data[:, -1]):
            inds = data[:, -1] == i
            p = data[inds].shape[0] / data.shape[0]
            entropy -= p * np.log2(p)

        return entropy
    
    def _impurity(self, data):
        r'''
        Return the impurity.

        Parameters
        ----------
        data : ndarray 
            Data to calculate the impurity on.

        Returns
        -------
        float :
            Impurity using the loss function.
        '''
        return self.loss(data)
    
    def _leaf_value(self, data):
        r'''
        Return the modal (most common) value in the input data.

        Parameters
        ----------
        data : ndarray
            Input data.

        Returns
        -------
        int : 
            Mode of input data.
        '''
        return stats.mode(data[:, -1])[0]
    
    def plot_tree(self, save_name, format='pdf'):
        r'''
        Plot and save the structure of the tree.

        Parameters
        ----------
        save_name : str
            Filename to save the figure.

        format : str, default `pdf`
            Format to save the figure.
        '''
        struct = np.sort(np.array(self.struct), axis=0)
        dot = Digraph()

        for i, (d, s, f, v) in enumerate(struct):
            label = f'Feature: {int(f)}\nSplit value: {int(s)}'
            dot.node(f'{i}', label=label)

            if i > 0:
                parent = (i - 1) // 2
                dot.edge(f'{parent}', f'{i}')

            dot.render(save_name, format=format, cleanup=True)
    
if __name__ == '__main__':
    import os 
    import sys 
    sys.path.insert(0, os.path.abspath('../..'))
    sys.path.insert(0, os.path.abspath('../../project1'))
    from project2.src.preprocess import center
    from sklearn.datasets import load_breast_cancer, load_digits
    from sklearn.model_selection import train_test_split
    from project3.src.classification_metrics import Metrics
    np.random.seed(2023)

    data = load_digits()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    DT = DecisionTree(max_depth=5, min_split=2)
    DT.fit(X_train, y_train)
    ypred = DT.predict(X_test)
    metrics = Metrics(y_test, ypred, np.unique(y))
    metrics.print_metrics()
    DT.plot_tree('../figures/pdfs/tree_example')
    DT.plot_tree('../figures/tree_example', format='png')