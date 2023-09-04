import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

class Regression:

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.2)

    def OLS(self, n_poly, identity_test=False):

        if identity_test:
            X = np.eye(len(self.x_data))
            beta = np.linalg.inv(X.T @ X) @ X.T @ self.y_data
            y_tilde = X @ beta 
            mse = MSE(self.y_data, y_tilde)

            print(f'\nMSE of OLS, using the identity matrix as the design matrix: {mse}\n')

        self.poly_degs = np.arange(1, n_poly + 1)
        self.mse_ols_train = np.zeros(n_poly)
        self.mse_ols_test = np.zeros(n_poly)
        self.r2_ols_train = np.zeros(n_poly)
        self.r2_ols_test = np.zeros(n_poly)
        self.beta_ols = np.zeros((n_poly, n_poly+1))

        for i in range(len(self.poly_degs)):
            model = Pipeline([('poly', PolynomialFeatures(degree=self.poly_degs[i])), \
                            ('linear', LinearRegression())])
            
            clf = model.fit(self.X_train, self.y_train)
            beta = model.named_steps['linear'].coef_[0]
            self.beta_ols[i, :i+2] = beta

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
        fig.suptitle(f'Data points: {len(self.x_data)}')

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
        
        coeffs = np.arange(len(beta[:, 0]))
        
        fig, ax = plt.subplots(figsize=(10, 7.5))

        for i in range(4):
            ax.plot(beta[:, i], coeffs, lw=1, label=r'$\beta$' + f'_{i}')
        
        ax.legend()

if __name__ == '__main__':
    np.random.seed(2018)
    n = 100
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x - 2)**2) + np.random.normal(0, 0.1, x.shape)

    fit = Regression(x, y)
    fit.OLS(35)
    fit.plot_evolution('OLS', 'test')
    plt.show()
