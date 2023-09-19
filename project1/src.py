import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge

plt.rcParams.update({
    'lines.linewidth': 1,
    'xtick.direction': 'inout',
    'ytick.direction': 'inout'
})

class Regression:

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
        y_train, y_test = self.y_train, self.y_test
        poly = PolynomialFeatures(degree=poly_deg)
        X_train = poly.fit_transform(self.X_train, y_train)
        X_test = poly.fit_transform(self.X_test, y_test)

        self.lambdas = np.logspace(lambda_min, lambda_max, n_lambda)
        self.mse_ridge_train = np.zeros(n_lambda)
        self.mse_ridge_test = np.zeros(n_lambda)
        self.r2_ridge_train = np.zeros(n_lambda)
        self.r2_ridge_test = np.zeros(n_lambda)

        beta = []

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
        
        self.beta_ridge = np.array(beta)

    def plot_evolution(self, model, figname):

        if model == 'OLS':
            mse_train, mse_test = self.mse_ols_train, self.mse_ols_test
            r2_train, r2_test = self.r2_ols_train, self.r2_ols_test
            beta = self.beta_ols; x = self.poly_degs
            x_label = 'Polynomial degree'
        
        elif model == 'ridge':
            mse_train, mse_test = self.mse_ridge_train, self.mse_ridge_test
            r2_train, r2_test = self.r2_ridge_train, self.r2_ridge_test
            beta = self.beta_ridge; x = self.lambdas
            x_label = r'$\lambda$'
        
        grid_spec = dict(hspace=0, height_ratios=[1, 1, 0, 2])
        fig, ax = plt.subplots(4, 1, figsize=(10, 7.5), gridspec_kw=grid_spec)

        ax[0].plot(x, mse_test, lw=1, color='red', label='MSE test')
        ax[0].plot(x, mse_train, lw=1, color='black', label='MSE train')
        ax[0].set_ylabel('MSE')
        ax[0].legend()
        ax[0].xaxis.set_tick_params(which='both', top=True, labeltop=True, bottom=True, labelbottom=False)

        ax[1].plot(x, r2_test, lw=1, color='red', label='R2 test')
        ax[1].plot(x, r2_train, lw=1, color='black', label='R2 train')
        ax[1].set_ylabel('R2 score')
        ax[1].legend()
        ax[1].xaxis.set_tick_params(top=True, labeltop=False, bottom=True, labelbottom=False)

        ax[2].set_visible(False)

        if model == 'OLS':

            for i in range(len(beta[:, 0])):
                ax[3].plot(x, beta[:, i], lw=1, label=r'$\beta$' + f'$_{i+1}$')
            
            ax[3].legend()
        
        elif model =='ridge':

            for i in range(len(beta[0, :])):
                ax[3].plot(x, beta[:, i])
            
            ax[0].set_xscale('log')
            ax[1].set_xscale('log')
            ax[3].set_xscale('log')

        ax[3].set_ylabel(r'$\beta_i$ value')
        ax[3].xaxis.set_tick_params(top=True, labeltop=False, bottom=True, labelbottom=True)
        ax[3].set_xlabel(x_label)

        fig.savefig('figures/' + model + '_' + figname + '.pdf', bbox_inches='tight')
        fig.savefig('figures/' + model + '_' + figname + '.png', bbox_inches='tight')

def frankes_function(x, y, add_noise=True):
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
    n = 100
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x - 2)**2) + np.random.normal(0, 0.1, x.shape)

    fit = Regression(x, y)
    fit.OLS(35, identity_test=True)
    fit.plot_evolution('OLS', 'test')
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