import autograd.numpy as np 
from cost_funcs import LogLoss
from autograd import grad 
import matplotlib.pyplot as plt

class LogisticRegression:
    r'''
    Logistic regression class for binary classification.

    The sigmoid function is used on the predicted outcome:

    .. math::
        \begin{align}
            \tilde y &= \mathbf X^T \mathbf\beta\\
            \sigma(\tilde y) &= \frac{1}{1+e^{-\tilde y}}
        \end{align}
    
    The cost function is given by the log-loss:

    .. math::
        \begin{align}
            C(\beta)=-\frac{1}{n}\sum_n\left[y_n\log(\tilde y_n+\delta)+(1-\tilde y_n)\log(1-\tilde y_n+\delta)\right]
        \end{align}

    Parameters
    ----------
    eta : float
        Learning rate (step size).

    alpha : float
        L2 regularization parameter.

    n_epochs : int
        Number of epochs (iterations) to perform. 

    tol : float, optional
        Tolerance of how equal the gradients are between runs
        (default: 10^-4). 
    '''
    def __init__(self, eta, alpha, n_epochs, tol=1e-4):
        self.eta = eta
        self.alpha = alpha
        self.epochs = n_epochs
        self.tol = tol
        self.loss = LogLoss.loss
        self.is_fit = False

    def _sigmoid(self, X, beta):
        r'''
        The sigmoid function.

        .. math::
            \begin{align}
                \tilde y &= \mathbf X^T \bm\beta\\
                \sigma(\tilde y) &= \frac{1}{1+e^{-\tilde y}}
            \end{align}

        Parameters
        ----------
        X : array_like
            The feature matrix.

        beta : array_like
            Beta parameters to be optimized.

        Returns
        -------
        array_like : 
            The sigmoid function.
        '''
        y_tilde = X @ beta 

        return 1 / (1 + np.exp(-y_tilde))
    
    def fit(self, X, y, X_val=None, y_val=None, verbose=True,
            patience=500):
        r'''
        Fit the model to the data.

        Parameters
        ----------
        X : array_like
            The feature matrix.

        y : array_like
            The target values. 

        X_val, y_val : array_like, optional
            Validation sets used to check performance during training.

        verbose : boolean, optional
            If ``True``, outputs information during testing.

        patience : int, optional (default: 500)
            How many epochs to run without improvement before stopping.
        '''
        has_val = False 
        if not X_val is None:
            has_val=True

        self.X = X
        self.beta = np.zeros((self.X.shape[1], 1))
        self.y = y 

        best_acc = 0
        counter = 0

        y_plot = []

        for i in range(self.epochs):
            y_pred = self._sigmoid(self.X, self.beta)
            loss = self.loss(self.y, y_pred)

            if has_val:
                pred = self._sigmoid(X_val, self.beta)
                pred = np.where(pred > self.tol, 1, 0)
                score = np.sum(pred == y_val) / y_val.size 
                y_plot.append(loss)

                print(f'Iteration: {i}')
                print(f'Accuracy: {score}\n')

                if score > best_acc:
                    best_acc = score 
                    counter = 0
                
                else:
                    counter += 1

                if counter >= patience:
                    break

            dBeta = self.X.T @ (y_pred - self.y) / self.y.size
            self.beta -= dBeta

        self.is_fit = True
        x_plot = np.arange(i+1)

        fig, ax = plt.subplots()
        ax.plot(x_plot, y_plot)
        plt.show()

    def predict(self, X):
        r'''
        Predict the outcome of the model on data.

        Parameters
        ----------
        X : array_like
            Feature matrix.

        Raises
        ------
        NotImplementedError
            If the model has not been fitted to data.
        '''
        if not self.is_fit:
            raise NotImplementedError
        
        y_pred = self._sigmoid(X, self.beta)

        return np.where(y_pred > self.tol, 1, 0)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
    from preprocess import center

    np.random.seed(2023)

    wdbc = load_breast_cancer()

    X = wdbc.data 
    y = wdbc.target[:, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = center(X_train)
    X_test = center(X_test)

    logreg = LogisticRegression(1e-3, 1e-3, int(1e4))
    logreg.fit(X_train, y_train, X_test, y_test, verbose=False)
    y_pred = logreg.predict(X_test)