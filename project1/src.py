import sys 
import os
sys.path.append('./props')
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, gridspec
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.utils import resample 
from scipy.linalg import svd
from time import perf_counter_ns
from numba import njit
from calc import Calculate

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
    'legend.fancybox': False,
    'legend.frameon': False,
    'savefig.bbox': 'tight',
    'axes.formatter.use_mathtext': True,
    'axes.formatter.useoffset': False,
    'axes.formatter.limits': [-2, 4]
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
        run the commands ``\the\columnwidth`` and ``\the\textwidth`` in your document body.
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
    Regression class for Ordinary Least Square (:any:`OLS`), Ridge,
    and Lasso (:any:`ridge_and_lasso`) regression. 

    Attributes
    ----------
    x_data, y_data : array_like
        Data along the x and y axes.

    z_data : array_like, default: None
        If the function describing the data is 2D, the z_data parameter describe
        the data along the z axis.
    
    name : str, optional
        The name that will be the title of certain figures. If not set during
        initialization, the figname has to be set  when calling :any:`plot_evolution`
    '''

    def __init__(self, x_data, y_data, z_data=None, name=None):
        self.figname = name

        # Setting the Calculate class as a property of Regression
        self.calculate = Calculate()

        # Likely to be removed as data is 2D
        if z_data is None:
            self.X = x_data
            self.y_data = y_data
            self.dim = 1
            tot_data_points = len(y_data)
        
        else:
            self.X = np.column_stack((x_data, y_data))
            self.y_data = z_data 
            self.dim = 2
            tot_data_points = len(z_data)
        
        print(f'\nLoaded with {tot_data_points} data points.')

        # Split the data into train and test, and scale by subtracting
        # the mean and divide by the standard deviation.
        self.X_train_unscaled, self.X_test_unscaled, self.y_train, self.y_test = train_test_split(self.X, self.y_data, test_size=0.2)

        scaler = StandardScaler()
        scaler.fit(self.X_train_unscaled)
        self.X_train = scaler.transform(self.X_train_unscaled)
        self.X_test = scaler.transform(self.X_test_unscaled)

    def OLS(self, n_poly, identity_test=False, store_beta=True):
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
        
        store_beta : bool, default: True
            If the input data is large, there is a chance that the beta values will be large 
            arrays. This parameter ensures the betas are not stored.
        
        Returns
        -------
        opt_deg : int
            The degree where the MSE of the test data is lowest.
        
        beta_opt : ndarray
            The optimal parameters for the lowest MSE.
        '''
        max_polys = int((n_poly + 1) * (n_poly + 2) / 2)
        self.store_beta = store_beta

        # Should return 0.
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

        beta_opt = None 
        min_mse = np.inf

        t1 = perf_counter_ns()
        print('\nOLS REGRESSION')
        print('--------------')

        for i in range(len(self.poly_degs)):
            X_train = self.calculate.create_X(self.X_train[:, 0],
                                              self.X_train[:, 1],
                                              self.poly_degs[i])

            X_test = self.calculate.create_X(self.X_test[:, 0],
                                             self.X_test[:, 1],
                                             self.poly_degs[i])
            
            beta, y_tilde, y_predict = self.calculate.ord_least_sq(X_train,
                                                                   X_test=X_test,
                                                                   y=self.y_train)
            
            mse_train = self.calculate.mean_sq_err(self.y_train, y_tilde)
            mse_test = self.calculate.mean_sq_err(self.y_test, y_predict)
            r2_train = self.calculate.R2_score(self.y_train, y_tilde)
            r2_test = self.calculate.R2_score(self.y_test, y_predict)

            if mse_test < min_mse:
                beta_opt = beta
                min_mse = mse_test

            if store_beta:
                beta = beta[0] if len(beta) == 1 else beta
                zeros = np.zeros(max_polys - len(beta))
                beta = np.append(beta, zeros)
                self.beta_ols[i, :] = beta

            self.mse_ols_train[i] = mse_train
            self.mse_ols_test[i] = mse_test
            self.r2_ols_train[i] = r2_train
            self.r2_ols_test[i] = r2_test

        t2 = perf_counter_ns()
        time = (t2 - t1) * 1e-9

        if time%60 > 1:
            print(f'Completed in {time//60:.0f} min {time%60:.0f} sec.')

        min_ols = np.min(self.mse_ols_test)
        best_r2 = np.max(self.r2_ols_test)

        self.opt_deg = self.poly_degs[self.mse_ols_test == min_ols][0]
        opt_deg_r2 = self.poly_degs[self.r2_ols_test == best_r2][0]

        OLS_results = '\nOLS results\n-----------\n'
        OLS_results += f'Highest R2 score: {best_r2:.3f} at degree: {opt_deg_r2}\n'
        OLS_results += f'Lowest MSE: {min_ols:11.3f} at degree: {self.opt_deg}\n'

        print(OLS_results)

        self.OLS_results = OLS_results

        return self.opt_deg, beta_opt
    
    def ridge_and_lasso(self, lambda_min, lambda_max, poly_deg, n_lambda):
        r'''
        Perform Ridge and Lasso regression on the provided data.

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
        
        Returns
        -------
        ridge_lasso_poly_deg : int
            The polynomial degree used for the regression.
        
        beta_opt_ridge : ndarray
            The optimal parameter for the Ridge regression.
        
        beta_opt_lasso : ndarray
            The optimal parameters for the Lasso regression.
        '''
        self.ridge_lasso_poly_deg = poly_deg
        y_train, y_test = self.y_train, self.y_test
        X_train = self.calculate.create_X(self.X_train[:, 0],
                                          self.X_train[:, 1],
                                          poly_deg)

        X_test = self.calculate.create_X(self.X_test[:, 0],
                                          self.X_test[:, 1],
                                          poly_deg)

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

        print('\nRIDGE AND LASSO REGRESSION')
        print('--------------------------')
        print('Finished:')
        t1 = perf_counter_ns()
        
        min_mse_lasso = np.inf
        min_mse_ridge = np.inf

        beta_opt_lasso = None 
        beta_opt_ridge = None

        tol = 1e-2 if poly_deg > 6 else 1e-4

        for i in range(n_lambda):
            lamb = self.lambdas[i]
            beta_ridge, y_tilde, y_predict = self.calculate.Ridge(X_train,
                                                                  X_test=X_test,
                                                                  y=self.y_train,
                                                                  lamb=lamb)

            mse_train = self.calculate.mean_sq_err(y_train, y_tilde)
            mse_test = self.calculate.mean_sq_err(y_test, y_predict)
            r2_train = self.calculate.R2_score(y_train, y_tilde)
            r2_test = self.calculate.R2_score(y_test, y_predict)

            self.mse_ridge_train[i] = mse_train
            self.mse_ridge_test[i] = mse_test
            self.r2_ridge_train[i] = r2_train
            self.r2_ridge_test[i] = r2_test

            if mse_test < min_mse_ridge:
                beta_opt_ridge = beta_ridge
                min_mse_ridge = mse_test

            lasso_reg = Lasso(lamb, fit_intercept=False, tol=tol, max_iter=int(1e5))
            lasso_reg.fit(X_train, y_train)

            y_tilde_lasso = lasso_reg.predict(X_train)
            y_predict_lasso = lasso_reg.predict(X_test)

            beta_lasso.append(lasso_reg.coef_)

            mse_train_lasso = self.calculate.mean_sq_err(y_train, y_tilde_lasso)
            mse_test_lasso = self.calculate.mean_sq_err(y_test, y_predict_lasso)
            r2_train_lasso = self.calculate.R2_score(y_train, y_tilde_lasso)
            r2_test_lasso = self.calculate.R2_score(y_test, y_predict_lasso)

            if mse_test_lasso < min_mse_lasso:
                beta_opt_lasso = lasso_reg.coef_
                min_mse_lasso = mse_test_lasso

            self.mse_lasso_train[i] = mse_train_lasso
            self.mse_lasso_test[i] = mse_test_lasso
            self.r2_lasso_train[i] = r2_train_lasso
            self.r2_lasso_test[i] = r2_test_lasso

            print(f'{(i+1) / n_lambda * 100:5.1f} %')
        
        t2 = perf_counter_ns()
        time = (t2 - t1) * 1e-9
        print(f'Completed in {time//60:.0f} min {time%60:.0f} sec.')

        self.beta_ridge = np.array(beta)
        self.beta_lasso = np.array(beta_lasso)

        min_ridge = np.min(self.mse_ridge_test)
        min_lasso = np.min(self.mse_lasso_test)
        abs_diff = abs(min_ridge - min_lasso)
        best_r2_ridge = np.max(self.r2_ridge_test)
        best_r2_lasso = np.max(self.r2_lasso_test)
        
        Ridge_results = '\nRidge results\n-------------\n'
        Ridge_results += f'Polynomial degree: {poly_deg}\n'
        Ridge_results += f'Highest R2 score: {best_r2_ridge:.5f} at lambda: {self.lambdas[self.r2_ridge_test == best_r2_ridge][0]:.3e}\n'
        Ridge_results += f'Lowest MSE: {min_ridge:13.5f} at lambda: {self.lambdas[self.mse_ridge_test == min_ridge][0]:.3e}\n'

        Lasso_results = '\nLasso results\n-------------\n'
        Lasso_results += f'Polynomial degree: {poly_deg}\n'
        Lasso_results += f'Highest R2 score: {best_r2_lasso:.5f} at lambda: {self.lambdas[self.r2_lasso_test == best_r2_lasso][0]:.3e}\n'
        Lasso_results += f'Lowest MSE: {min_lasso:13.5f} at lambda: {self.lambdas[self.mse_lasso_test == min_lasso][0]:.3e}\n'

        print(Ridge_results)
        print(Lasso_results)
        print(f'Absolute difference between Ridge and Lasso MSE: {abs_diff:.3e}')

        self.Ridge_results = Ridge_results
        self.Lasso_results = Lasso_results

        return self.ridge_lasso_poly_deg, beta_opt_ridge, beta_opt_lasso

    def plot_evolution(self, model, figname=None, add_lasso=True, add_zoom=False, scale_down=False):
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
        
        add_zoom : bool, default: False
            If ``True``, the lowest MSE value and highest R2 score will
            be shown as zoomed in axis.
        
        scale_down : bool, default: False
            The figure may be quite high relative to its width, so the
            setting this option to ``True`` will scale the height by 
            75 percent.
        '''
        if self.figname is None: 
            self.figname = figname

        if model == 'OLS':
            mse_train, mse_test = self.mse_ols_train, self.mse_ols_test
            r2_train, r2_test = self.r2_ols_train, self.r2_ols_test
            beta = self.beta_ols; x = self.poly_degs
            x_label = 'Polynomial degree'
            title = 'OLS'
        
        elif model == 'ridge' and add_lasso:
            mse_train, mse_test = self.mse_ridge_train, self.mse_ridge_test
            r2_train, r2_test = self.r2_ridge_train, self.r2_ridge_test
            beta = self.beta_ridge; x = np.log10(self.lambdas)

            mse_train_lasso, mse_test_lasso = self.mse_lasso_train, self.mse_lasso_test
            r2_train_lasso, r2_test_lasso = self.r2_lasso_train, self.r2_lasso_test
            beta_lasso = self.beta_lasso

            scale = 0.75 if scale_down else 1.0

            x_label = r'$\log_{10}\,\lambda$'
            fig, ax = plt.subplots(2, 1, figsize=set_size(subplot=(2, 1),scale=scale), sharex=True)
            ax[0].set_title(f'Polynomial degree: {self.ridge_lasso_poly_deg}')

            ax[0].plot(x, mse_test, 'r', label='Ridge test')
            ax[0].plot(x, mse_train, 'r--', label='Ridge train')
            ax[0].plot(x, mse_test_lasso, 'b', label='Lasso test')
            ax[0].plot(x, mse_train_lasso, 'b--', label='Lasso train')
            ax[0].set_ylabel('MSE')
            ax[0].legend(ncol=2, loc='upper left')

            ax[1].plot(x, r2_test, 'r', label='Ridge test')
            ax[1].plot(x, r2_train, 'r--', label='Ridge train')
            ax[1].plot(x, r2_test_lasso, 'b', label='Lasso test')
            ax[1].plot(x, r2_train_lasso, 'b--', label='Lasso train')
            ax[1].set_ylabel('R2 score')
            ax[1].set_xlabel(x_label)
            ax[1].legend(ncol=2, loc='lower left')

            has_min_mse = np.any(mse_test < mse_test[0])
            has_min_mse_lasso = np.any(mse_test_lasso < mse_test_lasso[0])
            has_max_r2 = np.any(r2_test > r2_test[0])
            has_max_r2_lasso = np.any(r2_test_lasso > r2_test_lasso[0])

            mse_cond = [has_min_mse, has_min_mse_lasso]
            r2_cond = [has_max_r2, has_max_r2_lasso]

            if add_zoom:
                x_diff = 0.5

                if np.any(mse_cond):
                    if np.min(mse_test_lasso) < np.min(mse_test):
                        y_center = np.min(mse_test_lasso)
                        best_model = mse_test_lasso
                    
                    else: 
                        y_center = np.min(mse_test)
                        best_model = mse_test

                    x_center_idx = np.argwhere(best_model == y_center)[0]
                    x_center = x[x_center_idx]

                    y_diff = 0.00005

                    x01 = x_center - x_diff
                    x02 = x_center + x_diff
                    y01 = y_center - y_diff
                    y02 = y_center + y_diff

                    axins0 = ax[0].inset_axes(
                        [0.25, 0.45, 0.25, 0.3],
                        # [x[0], 0.055, 4.5, 0.02],
                        # transform=ax[0].transData,
                        xlim=(x01, x02),
                        ylim=(y01, y02)
                    )

                    axins0.plot(x, mse_test, 'r')
                    axins0.plot(x, mse_test_lasso, 'b')
                    _, corners = ax[0].indicate_inset_zoom(axins0, ec='k')
                    corners[0].set_visible(True)
                    corners[1].set_visible(False)
                    corners[2].set_visible(True)
                    corners[3].set_visible(False)

                if np.any(r2_cond):
                    if np.max(r2_test_lasso) > np.max(r2_test):
                        y_center = np.max(r2_test_lasso)
                        best_model = r2_test_lasso
                    
                    else: 
                        y_center = np.max(r2_test)
                        best_model = r2_test

                    x_center_idx = np.argwhere(best_model == y_center)[0]
                    x_center = x[x_center_idx]

                    y_diff = 0.0005

                    x11 = x_center - x_diff
                    x12 = x_center + x_diff
                    y11 = y_center - y_diff
                    y12 = y_center + y_diff

                    axins1 = ax[1].inset_axes(
                        [0.25, 0.3, 0.25, 0.3],
                        # [x[0], 0.32, 4.5, 0.175],
                        # transform=ax[1].transData,
                        xlim=(x11, x12),
                        ylim=(y11, y12)
                    )

                    axins1.plot(x, r2_test, 'r')
                    axins1.plot(x, r2_test_lasso, 'b')
                    _, corners1 = ax[1].indicate_inset_zoom(axins1, ec='k')
                    corners1[0].set_visible(False)
                    corners1[1].set_visible(True)
                    corners1[2].set_visible(False)
                    corners1[3].set_visible(True)

                if not np.any([mse_cond, r2_cond]):
                    print('\nRequested zoom but the best score is the first value.')

            fig.tight_layout()
            fig.savefig('figures/' + model + '_lasso_' + self.figname + '.pdf')
            fig.savefig('figures/' + model + '_lasso_' + self.figname + '.png')

            return
        
        elif model == 'ridge' and not add_lasso:
            mse_train, mse_test = self.mse_ridge_train, self.mse_ridge_test
            r2_train, r2_test = self.r2_ridge_train, self.r2_ridge_test
            beta = self.beta_ridge; x = np.log10(self.lambdas)
            title = f'Ridge, polynomial degree: {self.ridge_lasso_poly_deg}'

            x_label = r'$\log_{10}\,\lambda$'
        
        elif model == 'lasso':
            mse_train, mse_test = self.mse_lasso_train, self.mse_lasso_test
            r2_train, r2_test = self.r2_lasso_train, self.r2_lasso_test
            beta = self.beta_lasso; x = np.log10(self.lambdas)
            x_label = r'$\log_{10}\,\lambda$'
            title = f'Lasso, polynomial degree: {self.ridge_lasso_poly_deg}'
        
        if model == 'OLS' and not self.store_beta:
            fig, ax = plt.subplots(2, 1, figsize=set_size(subplot=(2, 1)), sharex=True)

            ax[0].plot(x, mse_test, 'r', label='Test')
            ax[0].plot(x, mse_train, 'r--', label='Train')
            ax[0].set_ylabel('MSE')
            ax[0].legend()

            ax[1].plot(x, r2_test, 'b', label='Test')
            ax[1].plot(x, r2_train, 'b--', label='Train')
            ax[1].set_ylabel('R2 Score')
            ax[1].set_xlabel(x_label)
            ax[1].legend()

            fig.tight_layout()
            fig.savefig('figures/' + model + '_' + self.figname + '.pdf')
            fig.savefig('figures/' + model + '_' + self.figname + '.png')

        else:
            grid_spec = dict(hspace=0.2, height_ratios=[0.5, 0.5, 0, 1])
            fig, ax = plt.subplots(4, 1, figsize=set_size( subplot=(2, 1), scale=0.75), gridspec_kw=grid_spec)

            line1, = ax[0].plot(x, mse_test, color='red', label='Test')
            line2, = ax[0].plot(x, mse_train, color='black', label='Train')
            ax[0].set_ylabel('MSE')
            ax[0].xaxis.set_tick_params(which='both', top=False, labeltop=False, bottom=True, labelbottom=False)

            ax[1].plot(x, r2_test, color='red', label='Test')
            ax[1].plot(x, r2_train, color='black', label='Train')
            ax[1].set_ylabel('R2 score')
            ax[1].xaxis.set_tick_params(top=False, labeltop=False, bottom=True, labelbottom=False)

            ax[2].set_visible(False)

            if model == 'OLS':

                for i in range(len(beta[:, 0])):
                    label = r'$\beta$' + f'$_{i+1}$'
                    ax[3].scatter(x, beta[:, i], s=2, label=label)
                
                if not figname == 'test' and beta.shape[0] < 5:
                    ax[3].legend(ncol=2)
            
            elif model =='ridge' or model == 'lasso':

                for i in range(len(beta[0, :])):
                    label = r'$\beta$' + f'$_{i+1}$'
                    ax[3].plot(x, beta[:, i], label=label)

            ax[3].set_ylabel(r'$\beta_i$ value')
            ax[3].xaxis.set_tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
            ax[3].set_xlabel(x_label)

            fig.legend(handles=[line1, line2], bbox_to_anchor=(0.5, 0.81), loc='center')
            fig.savefig('figures/' + model + '_' + self.figname + '.pdf')
            fig.savefig('figures/' + model + '_' + self.figname + '.png')
    
    def bias_variance_tradeoff(self, max_degree, n_bootstraps, data_dim=2, best_deg_ols=None):
        r'''
        Compute OLS with a user defined number of bootstraps on the data
        and plot the evolution of the bias and variance as functions of
        the polynomial degree.

        Parameters
        ----------
        max_degree : int
            Maximum polynomial degree to evalute the OLS for.
        
        n_bootstraps : int 
            The number of bootstrap resamples to perform.
        
        data_dim : int, default: 2
            Dimension of the data the model is evaluated on.
        
        beta_deg_ols : int or float, optional
            The optimal degree as computed by the :any:`OLS` method.
            If not given, the degree is assumed to be computed from 
            the same code run as the present one.
        
        Warning
        -------
        As the data traffic can be very large, the number of samples 
        used to perform the regression should not be too big, as this 
        can lead to the program being killed before completing. 
        '''
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        
        ols_p = self.opt_deg if best_deg_ols is None else best_deg_ols

        if data_dim == 1:
            max_degree += 1
            degree = np.arange(max_degree)
            error_y_test = y_test
            keepdims = True
            filename = 'test-bias-var-trade'
            scale = 'linear'
        
        elif data_dim == 2 and not self.figname == 'geodata':
            degree = np.arange(1, max_degree+1)
            error_y_test = y_test[:, np.newaxis]
            keepdims = False
            filename = self.figname + '-bias-var-trade'
            scale = 'linear'
        
        else:
            degree = np.arange(1, max_degree+1)
            y_test = y_test.flatten()
            error_y_test = y_test[:, np.newaxis]
            keepdims = False 
            filename = self.figname + '-bias-var-trade'
            scale = 'linear'
        
        error = np.zeros(max_degree)
        bias = np.zeros_like(error)
        variance = np.zeros_like(error)

        bias_var_dist = np.inf
        tradeoff = 0

        j = 0
        print('\nBIAS VARIANCE TRADEOFF')
        print('----------------------')
        print('Finished:')
        t1 = perf_counter_ns()

        for deg in degree:
            model = make_pipeline(PolynomialFeatures(degree=deg),
                                  LinearRegression(fit_intercept=False))

            idx = y_test.shape[0] if not self.figname == 'geodata' else len(y_test.flatten())
            y_pred = np.empty((idx, n_bootstraps))

            for i in range(n_bootstraps):
                x_, y_ = resample(X_train, y_train)
                y_pred[:, i] = model.fit(x_, y_).predict(X_test).ravel()

            error[j] = np.mean(np.mean((error_y_test - y_pred)**2, axis=1, keepdims=keepdims))
            bias[j] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=keepdims))**2)
            variance[j] = np.mean(np.var(y_pred, axis=1, keepdims=keepdims))

            dist = (bias[j] - variance[j])**2

            if dist < bias_var_dist:
                tradeoff = deg 
                bias_var_dist = dist

            j += 1

            print(f'{deg / max_degree * 100:5.1f} %')

        t2 = perf_counter_ns()
        time = (t2 - t1) * 1e-9
        print(f'Completed in {time//60} min {time%60:.0f} sec.')
        print(f'Trade-off happens at degree: {tradeoff}.')
        self.tradeoff = tradeoff

        fig, ax = plt.subplots(figsize=set_size())
        ax.set_title(f'{n_bootstraps} bootstraps')
        ax.plot(degree, error, 'r', label='Error')
        ax.plot(degree, bias, 'g--', label='Bias')
        ax.plot(degree, variance, 'b', label='Variance')

        ymin, ymax = ax.get_ylim()

        ax.plot([tradeoff, tradeoff], [ymin, ymax], 'k', lw=0.75, alpha=0.75, ls='dashdot', label='Trade-off')
        ax.plot([ols_p, ols_p], [ymin, ymax], 'k', lw=0.75, alpha=0.75, ls='dashed', label='OLS')
        ax.set_xlabel('Polynomial degree')
        ax.set_yscale(scale)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.arange(np.ceil(max(degree)+1))[::2])
        ax.legend(ncol=2)

        fig.tight_layout()
        fig.savefig('figures/' + filename + '.png')
        fig.savefig('figures/' + filename + '.pdf')

    def cross_validation(self, n_kfolds, best_deg_ols=None, tradeoff=None, add_zoom=False):
        r'''
        Calculate and plot the results from a k-fold cross validation.

        Parameters
        ----------
        n_kfolds : int 
            The number of k-folds to concider.
        
        best_deg_ols : int or float, optional
            See :any:`bias_variance_tradeoff`
        
        tradeoff : int or float, optional
            The degree where the bias-variance trade-off happens. If 
            not set, the method assumes the degree has been computed 
            during the present run.
        
        add_zoom : bool, default: False
            If ``True`` , a zoomed in axis will display the location
            where the best model, i.e. eihter Lasso or Ridge has its 
            lowest MSE. 
        '''
        X, y = self.X, self.y_data
        X = StandardScaler().fit(X).transform(X)

        ols_p = self.opt_deg if best_deg_ols is None else best_deg_ols
        trade = self.tradeoff if tradeoff is None else tradeoff
        
        n_data = X.shape[0]
        indices = np.arange(n_data)
        shuffled_inds = np.random.choice(indices, replace=False, size=n_data)
        kfolds = np.array_split(shuffled_inds, n_kfolds)
        KFold_sklearn = KFold(n_splits=n_kfolds, shuffle=True, random_state=2023)

        if self.figname == 'geodata':
            n_poly = 21
            n_lambda = 75
            lambdas = np.logspace(-3.5, -1.5, n_lambda)
        
        else:
            n_poly = 16
            n_lambda = 75
            lambdas = np.logspace(-6, 3, n_lambda)

        poly_deg = tradeoff
        degrees = np.arange(1, n_poly+1)

        scores_OLS = np.zeros((n_poly, n_kfolds))
        scores_Ridge = np.zeros((n_lambda, n_kfolds))
        scores_Lasso = np.zeros_like(scores_Ridge)

        est_MSE_OLS_sklearn = np.zeros(n_poly)
        est_MSE_Ridge_sklearn = np.zeros(n_lambda)
        est_MSE_Lasso_sklearn = np.zeros_like(est_MSE_Ridge_sklearn)

        @njit
        def _compute_OLS(X_train, X_test, y_train, y_test):
            r'''
            Helper function to compute the OLS with the no-python 
            decorator.

            Parameters
            ----------
            X_train : ndarray
                Training part of the design matrix.
            
            X_test : ndarray
                Testing part of the design matrix.
            
            y_train : ndarray
                Train data.

            y_test : ndarray
                Test data.
            
            Returns
            -------
            float :
                The MSE.
            '''
            beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
            y_pred = X_test @ beta

            return np.sum((y_pred - y_test)**2) / y_pred.size
        
        @njit
        def _compute_Ridge(X_train, X_test, y_train, y_test, lamb):
            r'''
            Helper function to compute the Ridge with the no-python 
            decorator.

            Parameters
            ----------
            X_train : ndarray
                Training part of the design matrix.
            
            X_test : ndarray
                Testing part of the design matrix.
            
            y_train : ndarray
                Train data.

            y_test : ndarray
                Test data.
            
            lamb : float
                The regularization parameter.
            
            Returns
            -------
            float :
                The MSE.
            '''
            _singular = X_train.T @ X_train + lamb * np.eye(X_train.shape[1])
            U, s, VT = np.linalg.svd(_singular)
            D = np.zeros((len(U), len(VT)))
            np.fill_diagonal(D, s)
            _unsingular = U @ D @ VT 
            inv = np.linalg.pinv(_unsingular)

            beta = inv @ X_train.T @ y_train
            y_pred = X_test @ beta

            return np.sum((y_pred - y_test)**2) / y_pred.size

        print('\nCROSS VALIDATION')
        print('----------------')
        print('Finished:')
        t1 = perf_counter_ns()
        tol = 0.1 if poly_deg > 4 else 1e-4

        count = 0
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

                X_train = self.calculate.create_X(x_train[:, 0], x_train[:, 1], deg)
                X_test = self.calculate.create_X(x_test[:, 0], x_test[:, 1], deg)

                # OLS
                scores_OLS[i, k] = _compute_OLS(X_train, X_test, y_train, y_test)

                # Sklearn's OLS results
                X_sklearn = PolynomialFeatures(degree=deg).fit_transform(X)
                lin = LinearRegression(fit_intercept=True)
                MSE_OLS_sklearn = cross_val_score(lin, X_sklearn, y, scoring='neg_mean_squared_error', cv=KFold_sklearn)
                est_MSE_OLS_sklearn[i] = np.mean(-MSE_OLS_sklearn)

            # Ridge and Lasso
            poly = PolynomialFeatures(degree=poly_deg)
            X_train = self.calculate.create_X(x_train[:, 0], x_train[:, 1], poly_deg)
            X_test = self.calculate.create_X(x_test[:, 0], x_test[:, 1], poly_deg)
            X_sklearn = poly.fit_transform(X)

            for l, lamb in enumerate(lambdas):
                scores_Ridge[l, k] = _compute_Ridge(X_train, X_test, y_train, y_test, lamb)

                # Sklearn's Ridge results
                ridge = Ridge(alpha=lamb, fit_intercept=False)
                MSE_Ridge_sklearn = cross_val_score(ridge, X_sklearn, y, scoring='neg_mean_squared_error', cv=KFold_sklearn)
                est_MSE_Ridge_sklearn[l] = np.mean(-MSE_Ridge_sklearn)

                # Sklearn' Lasso results
                lasso = Lasso(lamb, fit_intercept=False, tol=tol, max_iter=int(1e5))
                y_pred_Lasso = lasso.fit(X_train, y_train).predict(X_test)
                scores_Lasso[l, k] = np.sum((y_pred_Lasso - y_test)**2) / np.size(y_pred_Lasso)
                MSE_Lasso_sklearn = cross_val_score(lasso, X_sklearn, y, scoring='neg_mean_squared_error', cv=KFold_sklearn)
                est_MSE_Lasso_sklearn[l] = np.mean(-MSE_Lasso_sklearn)
            
                print(f'{(count+1) / (len(lambdas) * n_kfolds) * 100:5.1f} %')
                count += 1

        t2 = perf_counter_ns()
        time = (t2 - t1) * 1e-9
        print(f'Completed in {time//60} min {time%60:.0f} sec.')
        
        est_MSE_OLS = np.mean(scores_OLS, axis=1)
        est_MSE_Ridge = np.mean(scores_Ridge, axis=1)
        est_MSE_Lasso = np.mean(scores_Lasso, axis=1)

        best_MSE_OLS = np.min(est_MSE_OLS)
        best_MSE_Ridge = np.min(est_MSE_Ridge)
        best_MSE_Lasso = np.min(est_MSE_Lasso)
        best_deg = degrees[np.where(est_MSE_OLS == best_MSE_OLS)[0]][0]
        best_lamb_Ridge = lambdas[np.where(est_MSE_Ridge == best_MSE_Ridge)[0]][0]
        best_lamb_Lasso = lambdas[np.where(est_MSE_Lasso == best_MSE_Lasso)[0]][0]

        print(f'OLS:{best_MSE_OLS:10.3f} at degree {best_deg}')
        print(f'Ridge:{best_MSE_Ridge:8.3f} at lambda {best_lamb_Ridge:.3g}')
        print(f'Lasso:{best_MSE_Lasso:8.3f} at lambda {best_lamb_Lasso:.3g}\n')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(subplot=(2, 1)))

        ax1.set_title(f'{n_kfolds} KFolds')
        ax1.plot(degrees, est_MSE_OLS, color='k', label=r'OLS')
        ax1.plot(degrees, est_MSE_OLS_sklearn, 'k--')
        ax1.set_xlabel('Polynomial degree')
        ax1.legend(loc='lower left')

        lasso, = ax2.plot(np.log10(lambdas), est_MSE_Lasso, lw=0.5, color='b', label=r'Lasso')
        ax2.plot(np.log10(lambdas), est_MSE_Lasso_sklearn, 'b--')
        ridge, = ax2.plot(np.log10(lambdas), est_MSE_Ridge, lw=0.5, color='r', label=r'Ridge')
        ax2.plot(np.log10(lambdas), est_MSE_Ridge_sklearn, 'r--')
        ax2.set_xlabel(r'$\log_{10}\lambda$')
        ax2.legend()

        if best_MSE_Ridge < best_MSE_Lasso:
            y_center = best_MSE_Ridge
            best_model = est_MSE_Ridge
            line = ridge
        
        else:
            y_center = best_MSE_Lasso
            best_model = est_MSE_Lasso
            line = lasso

        x_center_idx = np.argwhere(best_model == y_center)[0]
        x_center = np.log10(lambdas[x_center_idx])
        y_diff = 2e-4

        x1 = x_center - 0.7
        x2 = x_center + 0.7
        y1 = y_center - y_diff
        y2 = y_center + y_diff

        axins = ax2.inset_axes(
                        [0.2, 0.3, 0.25, 0.3],
                        # [x[0], 0.055, 4.5, 0.02],
                        # transform=ax2.transData,
                        xlim=(x1, x2),
                        ylim=(y1, y2)
        )

        axins.plot(np.log10(lambdas), best_model, color=line.get_color())
        axins.plot(np.log10(lambdas), est_MSE_Lasso_sklearn, 'b--')
        axins.plot(np.log10(lambdas), est_MSE_Ridge_sklearn, 'r--')

        _, corners = ax2.indicate_inset_zoom(axins, ec='k')
        corners[0].set_visible(False)
        corners[1].set_visible(True)
        corners[2].set_visible(True)
        corners[3].set_visible(False)


        fig.supylabel('MSE', fontsize=8)
        fig.tight_layout()
        fig.savefig('figures/' + self.figname + f'-{n_kfolds}-KFolds-cross-val.png')
        fig.savefig('figures/' + self.figname + f'-{n_kfolds}-KFolds-cross-val.pdf')
        
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
        return res + 0.1 * np.random.normal(0, 1.0, x.shape)

    else:
        return res
    
def compare_surface(linspace, poly_deg, surface, reg_model, azim_view_init=45, cmap='terrain'):
    r'''
    Create a side-by-side comparison figure of model and surface data.

    The figure will display both the maps and the surfaces.

    Parameters
    ----------
    linspace : ndarray
        A ``np.linspace`` instance that will be used to create the mesh
        grid.
    
    poly_deg : int
        Polynomial degree to use for the regression.
    
    surface : array_like
        The true surface.
    
    reg_model : {'OLS', 'Ridge'}, str
        Regression model to be used. Available are OLS and Ridge.
    
   azim_view_init : int, default: 45
        The azimuthal angle to view the 3D surface from. 
    
    cmap : str, default: 'terrain'
        Colormap to use when plotting.
    '''
    mods = {'OLS': Calculate.ord_least_sq,
            'Ridge': Calculate.Ridge
           }

    path = 'figures/'

    if len(reg_model.split(' ')) > 1:
        figname, modname = reg_model.split(' ')
        name = modname
        mod = mods[modname]
        path += figname
    
    else: 
        name = reg_model
        mod = mods[reg_model]

    name += f'-compare-terrain-{poly_deg}'
    path += name
    N = len(linspace)

    x = linspace 
    y = linspace 
    X, Y = np.meshgrid(x, y)

    feature = Calculate.create_X(X, Y, poly_deg)
    beta, ytilde, _ = mod(feature, surface.ravel())
    ytilde = ytilde.reshape(N, N)

    fig = plt.figure(figsize=set_size('text', subplot=(2, 1), scale=0.65))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot_surface(X, Y, ytilde,
        linewidth=0,
        antialiased=False,
        cmap=cmap
    )

    ax3.view_init(azim=azim_view_init)

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot_surface(
        X, Y, surface,
        linewidth=0,
        antialiased=False,
        cmap=cmap,
    )

    ax4.view_init(azim=azim_view_init)

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)

    title = reg_model + f', polynomial degree {poly_deg}'
    fig.suptitle(title, fontsize=10)

    im1 = ax1.imshow(ytilde, cmap=cmap)
    im2 = ax2.imshow(surface, cmap=cmap)
    ims = [im1, im2]

    label1 = ''
    label2 = 'Elevation [km]'
    labels = [label1, label2]

    ax1.set_title('Model')
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$y$')    
    ax3.set_zlabel(r'$z$')
    ax2.set_title('Terrain data')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax4.set_xlabel(r'$x$')
    ax4.set_ylabel(r'$y$')
    ax4.set_zlabel(r'$z$')

    axes = [ax1, ax2, ax3, ax4]
    fig.colorbar(im2, pad=0.05, aspect=25, ax=axes[:2], label='Normalized elevation')
    fig.colorbar(im2, pad=0.05, aspect=25, ax=axes[2:], label='Normalized elevation')

    fig.savefig(path + '.png')
    fig.savefig(path + '.pdf')
