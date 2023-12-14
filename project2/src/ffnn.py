import autograd.numpy as np 
from autograd import grad
from copy import copy
from cost_funcs import *
from solvers import *
from activations import *
from preprocess import to_categorical

class NeuralNetwork:
    r'''
    Artificial neural network class for regression and classification.

    Parameters
    ----------
    input_size : int
        Nodes in the input layer.
    
    hidden_sizes : list of ints
        Structure of the hidden layers. E.g. ``[10, 20, 10]`` will
        create 3 hidden layers consisting of 10, 20, and 10 hidden
        neurons, respectively.

    output_size : int 
        Nodes in the output layer.

    eta : float
        Learning rate.

    alpha : float 
        L2 regularization parameter.

    hidden_activation : {'sigmoid, relu, lrelu'}
        Activation function for the hidden layers. Note that the weights
        are initialized depending on the activation function as:

            * 'sigmoid' : ``XavierInitializer``.
            * 'relu' : ``HeInitializer``.
            * 'lrelu' : ``LeCunInitializer``.
    
    output_activation : {'sigmoid', 'relu', 'lrelu', 'linear'}
        Activation function for the output layer. If other than `linear`,
        classification is assumed.

    cost_function : {'mse', 'cross', 'log'}
        Cost or loss function. 

            * 'mse' : Mean squared error.
            * 'cross' : Cross entropy.
            * 'log' : Log-loss.

    epochs : int
        Number of iterations to perform.

    batch_size : int 
        Minibatch size to use for stochastic gradient descent. If
        ``batch_size == n_datapoints``, regular gradient descent is used.

    solver : {'adam', 'constant'}, str
        Which optimizer to use. ADAM is the best. 

    random_weights : boolean, optional
        Weather to overwrite the weight initialization. If ``True``, the 
        weights will be initialized from the normal distribution. If
        ``False`` (default), the activation of the hidden layers controls
        this.

    variable_eta : boolearn, optional
        Used with `constant` solver. If ``True`` (default), the learning
        rate will be set by a scheduler.
    '''
    def __init__(self, input_size, hidden_sizes, output_size, eta, alpha,
                 hidden_activation, output_activation, cost_function,
                 epochs, batch_size, solver, random_weights=False,
                 variable_eta=True):
        
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.eta = eta 
        self.alpha = alpha 
        self.epochs = epochs
        self.batch_size = batch_size
        self.solver = solver
        self.variable_eta = variable_eta

        self.is_multilabel = False

        if output_size > 1:
            '''
            Controls the way the predictions are represented. When 
            is_multilabel is true, predictions will be represented as
            one-hot vectors.
            '''
            self.is_multilabel = True

        activs = {'sigmoid': [Sigmoid, XavierInitializer],
                  'relu': [ReLU, HeInitializer],
                  'lrelu': [LeakyReLU, LeCunInitializer]}
        
        out_activs = {'sigmoid': Sigmoid, 'relu': ReLU,
                      'lreu': LeakyReLU, 'linear': Linear,
                      'softmax': Softmax}
        
        costs = {'mse': MeanSquaredError,
                 'cross': CrossEntropy,
                 'log': LogLoss}
        
        self.loss_evol = []
        self.score_evol = []
        
        self.hidden_func, self.weight_func = activs[hidden_activation]
        self.output_func = out_activs[output_activation]
        self.cost_func = costs[cost_function]

        if random_weights:
            self.weight_func = RandomInitializer

        self._create_weights_and_bias()

    def _create_weights_and_bias(self):
        r'''
        Initialize the weights and biases for the layers. 
        '''
        self.weights = []
        self.bias = []

        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i+1]

            weights = self.weight_func.initialize(input_size, output_size)
            bias = np.zeros((1, output_size)) + 1e-2

            self.weights.append(weights)
            self.bias.append(bias)

    def _feed_forward(self, X):
        r'''
        Perform a feed-forward pass through the network.

        Parameters
        ----------
        X : array_like 
            Feature matrix. Must have shape ``(n_datapoints, input_size)``.

        Returns
        -------
        a : array_like
            The activation of the output layer.

            .. math::
                \begin{align}
                    a^L=f(z^L),
                \end{align}

            where :math:`z^L` is the output of the last hidden layer.
        '''
        self.a = [X]
        self.z = [X]

        input_layer = X 
        for i in range(len(self.weights)):
            z = (input_layer @ self.weights[i]) + self.bias[i]

            if i < len(self.weights) - 1:
                a = self.hidden_func.function(z)
            
            else:
                a = self.output_func.function(z)

            self.z.append(z)
            self.a.append(a)
            input_layer = a

        return a
    
    def _backpropagate(self, X, y):
        r'''
        Calculate the back-propagated error from a forward pass through
        the network.

        During training, the weights and biases are updated using their
        respective gradients.

        Parameters
        ----------
        X : array_like
            The feature matrix.

        y : array_like
            Target values. 
        '''
        y_pred = self._feed_forward(X)
        dOut = self.output_func.derivative
        dCost = grad(self.cost_func.loss, 1)
        dHidden = self.hidden_func.derivative

        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                if self.is_multilabel:
                    delta = y_pred - y
                
                else:
                    delta = dOut(self.z[-1]) * dCost(y, y_pred)

            else:
                delta = (delta @ self.weights[i+1].T) * dHidden(self.z[i+1])
            
            w_grad = (self.a[i].T @ delta) + self.weights[i] * self.alpha
            b_grad = np.sum(delta, axis=0)

            w_change = self.w_solver[i].update_change(w_grad)
            b_change = self.b_solver[i].update_change(b_grad)

            self.weights[i] -= w_change
            self.bias[i] -= b_change

    def predict(self, X, tol=0.5):
        r'''
        Predict the output of a given dataset.

        Parameters
        ----------
        X : array_like
            Feature matrix.

        tol : float, optional
            For (hard) classification, if any of the predicted values
            are higher than `tol` (default: 0.5), return 1. Otherwise
            return 0.

        Returns
        -------
        array_like :
            For regression problems, the predicted values are returned. 
            For classification problems, 1 is returned for any values 
            larger than `tol`, and 0 otherwise.
        '''
        y_pred = self._feed_forward(X)

        if self.is_multilabel:
            return np.argmax(y_pred, axis=-1)

        elif self.output_func == Linear:
            return y_pred
        
        else:
            return np.where(y_pred > tol, 1, 0)

    def fit(self, X, y, X_val=None, y_val=None, tol=1e-4, patience=10,
            verbose=True):
        r'''
        Train the network with features and targets.

        Parameters
        ----------
        X : array_like
            Feature matrix.

        y : array_like
            Targets.

        X_val, y_val : array_like, optional
            If provided, the network will use them to test the score
            after each iteration.

        tol : float, optional
            Tolerance of how much the score improves between iterations.

        patience : int, optional
            Controls how many iterations the network will perform with 
            no improvement of the score.

        verbose : boolean, optional
            If ``True`` (default), outputs information during training.
        '''
        
        has_val = False

        if not X_val is None:
            has_val = True

        n = len(X)
        n_batches = int(np.ceil(n / self.batch_size))
        indices = np.arange(n)
        shuffled_inds = np.random.choice(indices, size=n, replace=False)
        rand_inds = np.array_split(shuffled_inds, n_batches)

        self.best_loss = np.inf
        best_weights = None 
        best_bias = None
        counter = 0

        # Learning rate schedule for SGD
        t0 = 1
        t1 = 50
        schedule = lambda t: t0 / (t + t1)

        self.w_solver = []
        self.b_solver = []

        if not self.variable_eta:
            if self.solver == 'adam':
                solver = ADAM(self.eta)            
            
            elif self.solver == 'constant':
                solver = Constant(self.eta)
            
            for i in range(len(self.weights)):
                self.w_solver.append(copy(solver))
                self.b_solver.append(copy(solver))

        for i in range(self.epochs):
            k = 0
            for inds in rand_inds:
                xi = X[inds]
                yi = y[inds]

                if self.variable_eta:
                    t = i * n_batches + k
                    self._learning_rate_scheduler(t)
                    solver = Constant(self.eta)

                    for i in range(len(self.weights)):
                        self.w_solver.append(copy(solver))
                        self.b_solver.append(copy(solver))

                self._backpropagate(xi, yi)

                if has_val:
                    y_pred = self.predict(X_val)

                    if self.is_multilabel:
                        y_pred = to_categorical(y_pred,
                                                n_categories=self.layer_sizes[-1])

                    loss = self.cost_func.loss(y_val, y_pred)

                    if loss < self.best_loss:
                        counter = 0

                        if abs(loss - self.best_loss) <= tol:
                            break

                        self.best_loss = loss
                        best_weights = self.weights
                        best_bias = self.bias 

                    else:
                        counter += 1
                k += 1

            if has_val:
                y_pred = self.predict(X_val)

                if self.is_multilabel:
                    y_pred = to_categorical(y_pred,
                                            n_categories=self.layer_sizes[-1])

                loss = self.cost_func.loss(y_val, y_pred)
                self.loss_evol.append(loss)

                score = self.calculate_score(y_val, y_pred)
                self.score_evol.append(score)
                    

            for w_solver, b_solver in zip(self.w_solver, self.b_solver):
                w_solver.reset()
                b_solver.reset()

            if counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {i} with loss: {self.best_loss}')

                break            

        if self.best_loss < np.inf:
            self.weights = best_weights
            self.bias = best_bias

        else:
            print('Loss did not improve.')

    def calculate_score(self, y_true, y_pred):
        r'''
        Score metrics. R2 score for regression, accuracy
        for classification.

        Parameters
        ----------
        y_true : array_like
            Target values.

        y_pred : array_like
            Predicted values.

        Returns
        -------
        r2score : float
            For regression.

        accuracy : float
            For classification.
        '''
        if self.output_func == Linear:
            ymean = np.mean(y_true)
            total_SOS= np.sum((y_true - ymean)**2)
            residual_SOS = np.sum((y_true - y_pred)**2)
            r2score = 1 - (residual_SOS / total_SOS)

            return r2score
        
        else:
            if self.is_multilabel:
                accuracy = np.average(y_pred == y_true)

            else:
                accuracy = np.sum(y_pred == y_true) / y_true.size

            return accuracy
    
    def get_score_evolution(self, limit=True):
        r'''
        Get the evolution of the score.

        Parameters
        ----------
        limit : boolean, optional
            Wether to slice the array to only contain improving values.

        Returns
        -------
        score_evol : ndarray
            Score array.
        '''
        if limit:
            idx = np.argwhere(self.loss_evol >= self.best_loss)[0][0]

            if not idx is None:
                return self.score_evol[:idx+1]
            
            else:
                return self.score_evol
            
        else:
            return self.score_evol
        
    def get_loss_evolution(self, limit=True):
        r'''
        Get the evolution of the loss.

        Parameters
        ----------
        limit : boolean, optional
            Wether to slice the array to only contain improving values.


        Returns
        -------
        loss_evol : ndarray
            Loss array.
        '''
        if limit:
            idx = np.argwhere(self.loss_evol >= self.best_loss)[0][0]

            if not idx is None:
                return self.loss_evol[:idx+1]
            
            else:
                return self.loss_evol
            
        else:
            return self.loss_evol
        
    def _learning_rate_scheduler(self, t, t0=1, t1=50):
        r'''
        Set a learning rate scheduler to deal with the noise when using
        the SGD method.

        Parameters
        ----------
        t : float
            Time step.

        t0, t1 : int, optional
            Scale parameters. Default values are ``t0=1`` and ``t1=50``.
        '''
        self.eta = t0 / (t + t1)

    def plot_layers(self, node_size=50, layer_spacing=2.5,
                    node_spacing=1.5, lw=0.25, ax=None):
        r'''
        Plot a figure of the layer structure. 

        Parameters
        ----------
        node_size : int, default 50
            Size of the nodes, which are created with ``matplotlib.pyplot.scatter``.

        layer_spacing : float, default 2.5
            Distance between the layers along the x-axis.

        node_spacing : float, defayúlt 1.5
            Vertical distance between the nodes in a layer.

        lw : float, default 0.25
            Width of the lines between the nodes.

        ax : matplotlib.AxesUbplot, optional
            If provided, the figure will be plotted on the given axis.
        '''
        layers = self.layer_sizes
        nodes = max(layers)

        if ax is None:
            fig, ax = plt.subplots()
        
        ax.axis('off')

        layer_x = [i * layer_spacing for i in range(len(layers))]
        y_offset = [node_spacing * (nodes - layer_size) / 2 for layer_size in layers]

        for i, layer_size in enumerate(layers):
            for j in range(layer_size):
                ax.scatter(layer_x[i],
                           j * node_spacing + y_offset[i],
                           s=node_size, ec='black', fc='grey', lw=0.5)

            if i < len(layers) - 1:
                for node1 in range(layers[i]):
                    for node2 in range(layers[i+1]):
                        ax.plot([layer_x[i], layer_x[i+1]],
                                [node1 * node_spacing + y_offset[i],
                                 node2 * node_spacing + y_offset[i+1]],
                                 color='black', lw=lw)
                        
        layer_names = ['Input layer', 'Hidden layers', 'Output layer']
        half = layer_x[-1] / 2
        names_x = [layer_x[0], half, layer_x[-1]]
        for i, layer_name in enumerate(layer_names):
            ax.text(names_x[i], nodes * node_spacing - 0.5,
                    layer_name, ha='center', color='black')
            

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../../')
    sys.path.insert(0, '../../project1/')
    import matplotlib.pyplot as plt 
    import seaborn as sns
    from project1.src import set_size

    sns.set_theme()

    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'savefig.bbox': 'tight',
    })
    
    NN = NeuralNetwork(2, [6, 8, 4], 2, 0.1, 0.1, 'relu', 'linear', 'mse', 10, 10, 'adam')

    fig, ax = plt.subplots(figsize=set_size())
    NN.plot_layers(ax=ax)
    fig.savefig('../figures/pdfs/layer_example.pdf')
    fig.savefig('../figures/layer_example.png')
    plt.show()