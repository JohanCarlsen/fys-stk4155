import numpy as np 

class Calculate:
    @staticmethod
    def create_X(x, y=None, poly_deg=3, include_ones=True):
        r'''
        Create a design matrix.

        Parameters
        ----------
        x : ndarray
            Points along the x-axis.
        
        y : ndarray
            Points along the y-axis.
        
        poly_deg : int
            Polynomial degree.
        
        Returns
        -------
        X : ndarray
            The created design matrix.
        '''
        N = len(x)
        n = poly_deg


        if y is None:

            X = np.ones((N, n+1))
            for i in range(1, n+1):
                X[:, i] = (x**i).ravel()
                
            # return X

        else:
            if len(x.shape) > 1:
                x = x.ravel()
                y = y.ravel()
            
            l = int((n+1) * (n+2) / 2)
            X = np.ones((N, l))

            for i in range(1, n+1):
                q = int((i * (i+1) / 2))
                for k in range(i+1):
                    X[:, q+k] = (x**(i-k) * y**k)
            
        return X if include_ones else X[:, 1:]

    @staticmethod
    def ord_least_sq(X, y, X_test=None):
        r'''
        Ordinary least square method.

        Parameters
        ----------
        X : ndarray
            Design matrix.
        
        y : ndarray
            Data points to use for OLS.
        
        X_test : ndarray, optional
            Test data to use the model on.
        
        Returns
        -------
        beta : ndarray
            The optimal parameters.
        
        ytilde : ndarray
            The optimal data from the model.
        
        ypred : ndarray
            If ``X_test`` is set, the model will be tested on the test
            data, and ``ypred`` will be returned.
        '''
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y 
        ytilde = X @ beta 

        ypred = None if X_test is None else X_test @ beta 

        return beta, ytilde, ypred
    
    @staticmethod
    def Ridge(X, y, lamb, X_test=None):
        r'''
        Ridge regression,

        Parameters
        ----------
        X, y, X_test : ndarray
            See: :any:`ord_least_sq`
        
        lamb : float
            Regularization parameter.

        Returns
        -------
        beta, ytilde, ypred : ndarray
            See: :any:`ord_least_sq`
        '''
        XTX = X.T @ X + lamb * np.eye(X.shape[1])
        beta = np.linalg.pinv(XTX) @ X.T @ y
        ytilde = X @ beta 

        ypred = None if X_test is None else X_test @ beta 

        return beta, ytilde, ypred

    @staticmethod
    def mean_sq_err(y, ytilde):
        r'''
        Mean squared error.

        Parameters
        ----------
        y : ndarray
            Data points (observed).
        
        ytilde : ndarray
            Data points (model).
        
        Returns
        -------
        float :
            The mean squared error.
        '''
        return np.sum((y - ytilde)**2 / y.size)
    
    @staticmethod
    def R2_score(y, ytilde):
        r'''
        R2 score.

        Parameters
        ----------
        y, ytilde : ndarray
            See: :any:`mean_sq_err`
        
        Returns
        -------
        r2score : float
            The R2 score.
        '''
        ymean = np.mean(y)
        total_SOS= np.sum((y - ymean)**2)
        residual_SOS = np.sum((y - ytilde)**2)
        r2score = 1 - (residual_SOS / total_SOS)

        return r2score
