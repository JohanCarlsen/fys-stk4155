import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.utils import resample 

plt.rcParams.update({
    'lines.linewidth': 1,
    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.fancybox': False
})

def set_size(width='col', scale=1.0, subplot=(1, 1)):
    r'''
    Set the size of a figure where the height/wifdth equals the golden ratio.

    Parameters
    ----------
    width : {'col', 'text'} or float, optional 
        Width of the figure in pt. Possible values:

            * 'col' (default): 255.46837
            * 'text' : 528.93675

    scale : float, default: 1.0
        How to scale the height of the figure, ie. ``figsize=(width, height * scale)``

    subplot : tuple, default: (1, 1)
        How to scale the figure size based on the number of subplots:
        ``figsize=(width, height * subplot[0] / subplot[1])``

    Returns
    -------
    tuple
        Dimension of the figure, ie. ``(width, height * scale * subplot[0] / subplot[1])``

    Notes
    -----
        The built-in values for `width` and `height` are the column width and text width
        in REVTeX document class. To obtain the appropriate values for your document, 
        run the commands ``\the\columnwidth`` and ``\the\textwidth`` in your dobument body.
    '''
    widths = {
        'col': 255.46837,
        'text': 528.93675
    }

    if width in widths:
        width_pt = widths[width]
    
    else:
        width_pt = width
    
    ratio = (5**0.5 - 1) / 2
    fig_width = width_pt / 72.27
    fig_height = fig_width * ratio * scale * subplot[0] / subplot[1]
    fig_dims = (fig_width, fig_height)

    return fig_dims

class Regression:
    r'''
    Regression class for Ordinary Least Square (:any:`OLS`), Ridge (:any:`ridge`),
    and Lasso (:any:`lasso`) regression. 

    Attributes
    ----------
    x_data, y_data : array_like
        Data along the x and y axes.

    z_data : array_like, default: None
        If the function describing the data is 2D, the z_data parameter describe
        the data along the z axis.
    '''

    def __init__(self, x_data, y_data, z_data=None):

        if z_data is None:
            self.X = x_data
            self.y_data = y_data
            self.dim = 1
        else:
            self.X = np.column_stack((x_data, y_data))
            self.y_data = z_data 
            self.dim = 2

        self.X_train_unscaled, self.X_test_unscaled, self.y_train, self.y_test = train_test_split(self.X, self.y_data, test_size=0.2)
        
        scaler = StandardScaler()
        scaler.fit(self.X_train_unscaled)
        self.X_train = scaler.transform(self.X_train_unscaled)
        self.X_test = scaler.transform(self.X_test_unscaled)

    def OLS(self, n_poly, identity_test=False):
        r'''
        Ordinary least square regression method.

        .. math::
            \begin{align}
            \tilde\beta &= (X^TX)^{-1}X^Ty \\
            \tilde y &= X\tilde\beta
            \end{align}
        
        with a user-defined polynomial degree.

        Parameters
        ----------
        n_poly : int
            Polynomial degree.

        identity_test : bool, default: False
            If ``True``, the method performs a test to see if the implementation is correct, 
            ie. if the design matrix is the identity matrix, the mean square error should be 0.
        '''
        max_polys = n_poly * self.dim**2 + 1

        if identity_test:
            X = np.eye(len(self.X))
            beta = np.linalg.inv(X.T @ X) @ X.T @ self.y_data
            y_tilde = X @ beta 
            mse = MSE(self.y_data, y_tilde)

            print(f'\nMSE of OLS, using the identity matrix as the design matrix: {mse}\n')

        self.poly_degs = np.arange(1, n_poly + 1)
        self.mse_ols_train = np.zeros(n_poly)
        self.mse_ols_test = np.zeros(n_poly)
        self.r2_ols_train = np.zeros(n_poly)
        self.r2_ols_test = np.zeros(n_poly)
        self.beta_ols = np.zeros((n_poly, max_polys))

        for i in range(len(self.poly_degs)):
            model = Pipeline([('scaler', StandardScaler()), \
                            ('poly', PolynomialFeatures(degree=self.poly_degs[i])), \
                            ('linear', LinearRegression())])

            clf = model.fit(self.X_train, self.y_train)
            beta = model.named_steps['linear'].coef_
            beta = beta[0] if len(beta) == 1 else beta
            zeros = np.zeros(max_polys - len(beta))
            beta = np.append(beta, zeros)
            self.beta_ols[i, :] = beta

            y_tilde = clf.predict(self.X_train)
            y_predict = clf.predict(self.X_test)
            
            mse_train = MSE(self.y_train, y_tilde)
            mse_test = MSE(self.y_test, y_predict)
            r2_train = R2(self.y_train, y_tilde)
            r2_test = R2(self.y_test, y_predict)

            self.mse_ols_train[i] = mse_train
            self.mse_ols_test[i] = mse_test
            self.r2_ols_train[i] = r2_train
            self.r2_ols_test[i] = r2_test
    
    def ridge(self, lambda_min, lambda_max, poly_deg, n_lambda):
        r'''
        Perform Ridge regression on the provided data.

        .. math::
            \begin{align}
            \hat\beta&=(X^TX+\lambda\mathbb{I})^{-1}X^T y \\
            \tilde y&=X\hat\beta
            \end{align}
        
        Parameters
        ----------

        lambda_min, lambda_max : float
            Lowest/highest value for :math:`\log_{10}\lambda`.
        
        poly_deg : int
            Order of the polynomial to fit. 
        
        n_lambda : int
            Number of :math:`\lambda`-elements to compute.
        '''
        y_train, y_test = self.y_train, self.y_test
        poly = PolynomialFeatures(degree=poly_deg)
        X_train = poly.fit_transform(self.X_train, y_train)
        X_test = poly.fit_transform(self.X_test, y_test)

        self.lambdas = np.logspace(lambda_min, lambda_max, n_lambda)
        self.mse_ridge_train = np.zeros(n_lambda)
        self.mse_ridge_test = np.zeros(n_lambda)
        self.r2_ridge_train = np.zeros(n_lambda)
        self.r2_ridge_test = np.zeros(n_lambda)

        self.mse_lasso_train = np.zeros(n_lambda)
        self.mse_lasso_test = np.zeros(n_lambda)
        self.r2_lasso_train = np.zeros(n_lambda)
        self.r2_lasso_test = np.zeros(n_lambda)

        beta = []
        beta_lasso = []

        for i in range(n_lambda):
            lambda_i = self.lambdas[i]
            beta_tilde = np.linalg.inv(X_train.T @ X_train + lambda_i * np.eye(X_train.shape[1])) @ X_train.T @ y_train
            beta.append(beta_tilde)

            y_tilde = X_train @ beta_tilde
            y_predict = X_test @ beta_tilde

            mse_train = MSE(y_train, y_tilde)
            mse_test = MSE(y_test, y_predict)
            r2_train = R2(y_train, y_tilde)
            r2_test = R2(y_test, y_predict)

            self.mse_ridge_train[i] = mse_train
            self.mse_ridge_test[i] = mse_test
            self.r2_ridge_train[i] = r2_train
            self.r2_ridge_test[i] = r2_test

            lasso_reg = Lasso(lambda_i)
            lasso_reg.fit(X_train, y_train)

            y_tilde_lasso = lasso_reg.predict(X_train)
            y_predict_lasso = lasso_reg.predict(X_test)

            beta_lasso.append(lasso_reg.coef_)

            mse_train_lasso = MSE(y_train, y_tilde_lasso)
            mse_test_lasso = MSE(y_test, y_predict_lasso)
            r2_train_lasso = R2(y_train, y_tilde_lasso)
            r2_test_lasso = R2(y_test, y_predict_lasso)

            self.mse_lasso_train[i] = mse_train_lasso
            self.mse_lasso_test[i] = mse_test_lasso
            self.r2_lasso_train[i] = r2_train_lasso
            self.r2_lasso_test[i] = r2_test_lasso
        
        self.beta_ridge = np.array(beta)
        self.beta_lasso = np.array(beta_lasso)

    def plot_evolution(self, model, figname, add_lasso=True):
        r'''
        Plot the evolution of the Mean Squared Error (MSE), R2 score, and the 
        values for the optimal parameters.

        Parameters
        ----------
        model : {'OLS', 'ridge', 'lasso'}
            Which model to use.

            * 'OLS': The curves will be as functions of the polynomial degree.
            * 'ridge' or 'lasso': The curves will be as functions of :math:`\lambda`.
        
        figname : str
            For saving the figure.

        add_lasso : bool, default: True
            Option to add the `lasso` model to the figure for `ridge`.
            If so, the figure will show only the MSE and R2 score for 
            both models. 
        '''
        if model == 'OLS':
            mse_train, mse_test = self.mse_ols_train, self.mse_ols_test
            r2_train, r2_test = self.r2_ols_train, self.r2_ols_test
            beta = self.beta_ols; x = self.poly_degs
            x_label = 'Polynomial degree'
        
        elif model == 'ridge' and add_lasso:
            mse_train, mse_test = self.mse_ridge_train, self.mse_ridge_test
            r2_train, r2_test = self.r2_ridge_train, self.r2_ridge_test
            beta = self.beta_ridge; x = np.log10(self.lambdas)

            mse_train_lasso, mse_test_lasso = self.mse_lasso_train, self.mse_lasso_test
            r2_train_lasso, r2_test_lasso = self.r2_lasso_train, self.r2_lasso_test
            beta_lasso = self.beta_lasso; x = np.log10(self.lambdas)

            x_label = r'$\log_{10}\,\lambda$'

            grid_spec = dict(hspace=0)
            fig, ax = plt.subplots(2, 1, figsize=set_size(subplot=(2, 1)), gridspec_kw=grid_spec)

            ax[0].plot(x, mse_test, color='red', label='Ridge test')
            ax[0].plot(x, mse_test_lasso, ls='dashed', color='red', label='Lasso test')
            ax[0].plot(x, mse_train, color='black', label='Ridge train')
            ax[0].plot(x, mse_train_lasso, ls='dashed', color='black', label='Lasso train')
            ax[0].set_ylabel('MSE')
            ax[0].xaxis.set_tick_params(which='both', top=True, labeltop=True, bottom=False, labelbottom=False)

            ax[1].plot(x, r2_test, color='red')
            ax[1].plot(x, r2_test_lasso, ls='dashed', color='red')
            ax[1].plot(x, r2_train, color='black')
            ax[1].plot(x, r2_train_lasso, ls='dashed', color='black')
            ax[1].set_ylabel('R2 score')
            ax[1].set_xlabel(x_label)

            fig.legend(loc='center right')
            fig.tight_layout()
            fig.savefig('figures/' + model + '_lasso_' + figname + '.pdf', bbox_inches='tight')
            fig.savefig('figures/' + model + '_lasso_' + figname + '.png', bbox_inches='tight')

            return
        
        elif model == 'ridge' and not add_lasso:
            mse_train, mse_test = self.mse_ridge_train, self.mse_ridge_test
            r2_train, r2_test = self.r2_ridge_train, self.r2_ridge_test
            beta = self.beta_ridge; x = np.log10(self.lambdas)

            x_label = r'$\log_{10}\,\lambda$'
        
        elif model == 'lasso':
            mse_train, mse_test = self.mse_lasso_train, self.mse_lasso_test
            r2_train, r2_test = self.r2_lasso_train, self.r2_lasso_test
            beta = self.beta_lasso; x = np.log10(self.lambdas)
            x_label = r'$\log_{10}\,\lambda$'
        
        grid_spec = dict(hspace=0, height_ratios=[1, 1, 0, 2])
        fig, ax = plt.subplots(4, 1, figsize=set_size(subplot=(2, 1)), gridspec_kw=grid_spec)

        line1, = ax[0].plot(x, mse_test, color='red', label='Test')
        line2, = ax[0].plot(x, mse_train, color='black', label='Train')
        ax[0].set_ylabel('MSE')
        ax[0].xaxis.set_tick_params(which='both', top=True, labeltop=True, bottom=True, labelbottom=False)

        ax[1].plot(x, r2_test, color='red', label='Test')
        ax[1].plot(x, r2_train, color='black', label='Train')
        ax[1].set_ylabel('R2 score')
        ax[1].xaxis.set_tick_params(top=True, labeltop=False, bottom=True, labelbottom=False)

        ax[2].set_visible(False)

        if model == 'OLS':

            for i in range(len(beta[:, 0])):
                ax[3].plot(x, beta[:, i], label=r'$\beta$' + f'$_{i+1}$')
            
            if not figname == 'test':
                ax[3].legend(ncol=2)
        
        elif model =='ridge' or model == 'lasso':

            for i in range(len(beta[0, :])):
                ax[3].plot(x, beta[:, i])

        ax[3].set_ylabel(r'$\beta_i$ value')
        ax[3].xaxis.set_tick_params(top=True, labeltop=False, bottom=True, labelbottom=True)
        ax[3].set_xlabel(x_label)

        fig.legend(handles=[line1, line2], bbox_to_anchor=(1, 0.75))
        fig.tight_layout()
        fig.savefig('figures/' + model + '_' + figname + '.pdf', bbox_inches='tight')
        fig.savefig('figures/' + model + '_' + figname + '.png', bbox_inches='tight')
    
    def bias_variance_tradeoff(self, max_degree, n_bootstraps, data_dim=2):
        r'''
        Perform OLS on the data and plot the evolution of the
        bias and variance as functions of the polynomial degree.

        Parameters
        ----------
        max_degree : int
            Maximum polynomial degree to evalute the OLS for.
        
        n_bootstraps : int 
            The number of bootstrap resamples to perform.
        
        data_dim : int, default: 2
            Dimension of the data the model is evaluated on.
        '''
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        if data_dim == 1:
            max_degree += 1
            degree = np.arange(max_degree)
            error_y_test = y_test
            keepdims = True
            filename = 'test-bias-var-trade'
            scale = 'linear'
        
        elif data_dim == 2:
            degree = np.arange(1, max_degree+1)
            error_y_test = y_test[:, np.newaxis]
            keepdims = False
            filename = 'bias-var-trade'
            scale = 'log'

        error = np.zeros(max_degree)
        bias = np.zeros_like(error)
        variance = np.zeros_like(error)

        for deg in degree:
            model = make_pipeline(PolynomialFeatures(degree=deg),
                                  LinearRegression(fit_intercept=False))

            y_pred = np.empty((y_test.shape[0], n_bootstraps))

            for i in range(n_bootstraps):
                x_, y_ = resample(X_train, y_train)
                y_pred[:, i] = model.fit(x_, y_).predict(X_test).ravel()

            deg_idx = deg - 1 if data_dim == 2 else deg

            error[deg_idx] = np.mean(np.mean((error_y_test - y_pred)**2, axis=1, keepdims=keepdims))
            bias[deg_idx] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=keepdims))**2)
            variance[deg_idx] = np.mean(np.var(y_pred, axis=1, keepdims=keepdims))

        fig, ax = plt.subplots(figsize=set_size())
        ax.plot(degree, error, label='Error', color='red')
        ax.plot(degree, bias, label='Bias', color='green')
        ax.plot(degree, variance, label='Variance', color='blue')
        ax.set_xlabel('Polynomial degree')
        ax.set_yscale(scale)
        ax.legend()

        fig.tight_layout()
        fig.savefig('figures/' + filename + '.png', bbox_inches='tight')
        fig.savefig('figures/' + filename + '.pdf', bbox_inches='tight')

    def cross_validation(self, n_kfolds):
        r'''
        Calculate and plot the results from a k-fold cross validation.

        Parameters
        ----------
        n_kfolds : int 
            The number of k-folds to concider.
        '''
        X, y = self.X, self.y_data 
        n_data = X.shape[0]
        indices = np.arange(n_data)
        shuffled_inds = np.random.choice(indices, replace=False, size=n_data)
        kfolds = np.array_split(shuffled_inds, n_kfolds)
        KFold_sklearn = KFold(n_splits=n_kfolds)

        n_poly = 15
        n_lambda = 500
        degrees = np.arange(1, n_poly+1)
        lambdas = np.logspace(-4, 4, n_lambda)

        scores_OLS = np.zeros((n_poly, n_kfolds))
        scores_Ridge = np.zeros((n_lambda, n_kfolds))
        scores_Lasso = np.zeros_like(scores_Ridge)

        est_MSE_OLS_sklearn = np.zeros(n_poly)
        est_MSE_Ridge_sklearn = np.zeros(n_lambda)
        est_MSE_Lasso_sklearn = np.zeros_like(est_MSE_Ridge_sklearn)

        for k in range(n_kfolds):
            inds = kfolds[k]
            boolean = np.zeros_like(indices, dtype=bool)
            boolean[inds] = True 
            train_inds = ~boolean 
            test_inds = boolean

            x_train = X[train_inds]
            y_train = y[train_inds]
            x_test = X[test_inds]
            y_test = y[test_inds]

            for deg in degrees:
                i = deg - 1

                poly = PolynomialFeatures(degree=deg)
                X_train = poly.fit_transform(x_train)
                X_test = poly.fit_transform(x_test)

                # OLS
                beta_OLS = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
                y_pred_OLS = X_test @ beta_OLS 
                scores_OLS[i, k] = np.sum((y_pred_OLS - y_test)**2) / np.size(y_pred_OLS)

                # Sklearn's OLS results
                X_sklearn = poly.fit_transform(X)
                lin = LinearRegression(fit_intercept=False)
                MSE_OLS_sklearn = cross_val_score(lin, X_sklearn, y, scoring='neg_mean_squared_error', cv=KFold_sklearn)
                est_MSE_OLS_sklearn[i] = np.mean(-MSE_OLS_sklearn)


            # Ridge and Lasso
            poly = PolynomialFeatures(degree=5)
            X_train = poly.fit_transform(x_train)
            X_test = poly.fit_transform(x_test)
            X_sklearn = poly.fit_transform(X)

            for l in range(len(lambdas)):
                beta_Ridge = np.linalg.inv(X_train.T @ X_train + lambdas[l] * np.eye(X_train.shape[1])) @ X_train.T @ y_train
                y_pred_Ridge = X_test @ beta_Ridge
                scores_Ridge[l, k] = np.sum((y_pred_Ridge - y_test)**2) / np.size(y_pred_Ridge)

                # Sklearn's Ridge results
                ridge = Ridge(alpha=lambdas[l])
                MSE_Ridge_sklearn = cross_val_score(ridge, X_sklearn, y, scoring='neg_mean_squared_error', cv=KFold_sklearn)
                est_MSE_Ridge_sklearn[l] = np.mean(-MSE_Ridge_sklearn)

                # Sklearn' Lasso results
                lasso = Lasso(lambdas[l])
                y_pred_Lasso = lasso.fit(X_train, y_train).predict(X_test)
                scores_Lasso[l, k] = np.sum((y_pred_Lasso - y_test)**2) / np.size(y_pred_Lasso)

                MSE_Lasso_sklearn = cross_val_score(lasso, X_sklearn, y, scoring='neg_mean_squared_error', cv=KFold_sklearn)
                est_MSE_Lasso_sklearn[l] = np.mean(-MSE_Lasso_sklearn)

        
        est_MSE_OLS = np.mean(scores_OLS, axis=1)
        est_MSE_Ridge = np.mean(scores_Ridge, axis=1)
        est_MSE_Lasso = np.mean(scores_Lasso, axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(subplot=(2, 1)))

        ax1.set_title(f'{n_kfolds} KFolds')
        ax1.plot(degrees, est_MSE_OLS, color='k', label=r'OLS')
        ax1.plot(degrees, est_MSE_OLS_sklearn, 'k--')
        ax1.set_xlabel('Polynomial degree')
        ax1.set_yscale('log')
        ax1.legend()

        ax2.plot(np.log10(lambdas), est_MSE_Ridge, color='r', label=r'Ridge')
        ax2.plot(np.log10(lambdas), est_MSE_Ridge_sklearn, 'r--')
        ax2.plot(np.log10(lambdas), est_MSE_Lasso, color='b', label=r'Lasso')
        ax2.plot(np.log10(lambdas), est_MSE_Lasso_sklearn, 'b--')
        ax2.set_xlabel(r'$\log_{10}\lambda$')
        ax2.legend()

        fig.supylabel('MSE', fontsize=8)
        fig.tight_layout()
        fig.savefig(f'figures/{n_kfolds}-KFolds-cross-val.png', bbox_inches='tight')
        fig.savefig(f'figures/{n_kfolds}-KFolds-cross-val.pdf', bbox_inches='tight')
        

def frankes_function(x, y, add_noise=True):
    r'''
    Franke's function.

    Parameters
    ----------
    x, y : array_like
        The `x` and `y` values to evaluate the function on.

    add_noise : bool, default: True
        Weather to add Gaussian noise :math:`\epsilon\sim\mathcal N(+,\sigma^2)`
        to the data.

    Returns
    -------
    array_like
        The resulting Franke's function, with or without the noise. 
    '''
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)

    res = term1 + term2 + term3 + term4

    if add_noise:
        return res + np.random.normal(0, 1.0, x.shape)

    else:
        return res
    
if __name__ == '__main__':
    np.random.seed(2018)
    n = 40
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x - 2)**2) + np.random.normal(0, 0.1, x.shape)

    fit = Regression(x, y)
    fit.OLS(20, identity_test=True)
    fit.plot_evolution('OLS', 'test')
    fit.bias_variance_tradeoff(max_degree=13, n_bootstraps=100, data_dim=1)
    plt.show()

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(projection="3d")

    # x = np.arange(0, 1, 0.05)
    # y = np.arange(0, 1, 0.05)
    # X, Y = np.meshgrid(x,y)
    # z = frankes_function(X, Y, add_noise=False)

    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)

    # # Customize the z axis.
    # ax.set_zlim(-0.10, 1.40)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()