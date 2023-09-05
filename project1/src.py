import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_data, test_size=0.2)

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
            beta = model.named_steps['linear'].coef_[0]
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
        
    def plot_evolution(self, model, figname):
        x = self.poly_degs

        if model == 'OLS':
            mse_train, mse_test = self.mse_ols_train, self.mse_ols_test
            r2_train, r2_test = self.r2_ols_train, self.r2_ols_test
            beta = self.beta_ols
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
        fig.suptitle(f'Data points: {len(self.X)}')

        ax[0].plot(x, mse_test, lw=1, color='red', label='MSE test')
        ax[0].plot(x, mse_train, lw=1, color='black', label='MSE train')
        ax[0].set_ylabel('MSE')
        ax[0].legend()

        ax[1].plot(x, r2_test, lw=1, color='blue', label='R2 test')
        ax[1].plot(x, r2_train, lw=1, color='green', label='R2 train')
        ax[1].set_ylabel('R2 score')
        ax[1].legend()

        fig.supxlabel('Polynomial degree')
        fig.subplots_adjust(hspace=0)
        fig.savefig('figures/' + model + '_' + figname + '.png', bbox_inches='tight')
        fig.savefig('figures/' + model + '_' + figname + '.pdf', bbox_inches='tight')
        
        fig, ax = plt.subplots(figsize=(10, 5))

        for i in range(len(self.poly_degs)):
            ax.plot(x, beta[:, i], lw=1, label=r'$\beta$' + f'$_{self.poly_degs[i]}$')
        
        ax.set_xlabel('Polynomial degree')
        ax.set_ylabel(r'$\beta_i$ value')
        ax.legend()

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