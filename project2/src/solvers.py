import autograd.numpy as np

class Solvers:
    r'''
    Parent class of the solvers. Should not be used directly.
    '''
    def __init__(self, eta):
        self.eta = eta 
    
    def update_change(self, gradient):
        raise NotImplementedError
    
    def reset(self):
        pass 

class Constant(Solvers):
    r'''
    Constant learning rate.

    Attributes
    ----------
    eta : float
        Learning rate.
    '''
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        r'''
        Update the change using the gradient.

        Parameters
        ----------
        gradient : array_like
            Gradient used to update the change.

        Returns
        -------
        array_like :
            The update :math:`\eta\nabla`.
        '''
        return self.eta * gradient 
    
    def reset(self):
        r'''
        Not used for this solver.
        '''
        pass 

class ADAM(Solvers):
    r'''
    The ADAM solver optimizer.

    Attributes
    ----------
    eta : float
        Learning rate.

    rho1, rho2 : float, optional
        Parameters used to calculate the momentum. ``rho1`` (default=0.9)
        is used for the first moment, and ``rho2`` (default=0.999) is
        used for the second moment.
    '''
    def __init__(self, eta, rho1=0.9, rho2=0.999):
        super().__init__(eta)
        self.rho1 = rho1 
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        r'''
        Update the change using the gradient.

        Parameters
        ----------
        gradient : array_like
            Gradient used to update the change.

        Returns
        -------
        array_like :
            The update.
        '''
        delta = 1e-8

        self.moment = self.rho1 * self.moment \
                    + (1 - self.rho1) * gradient 
        
        self.second = self.rho2 * self.second \
                    + (1 - self.rho2) * gradient * gradient 
        
        moment_corrected = self.moment / (1 - self.rho1**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))
    
    def reset(self):
        r'''
        Resets the solver.
        '''
        self.n_epochs += 1
        self.moment = 0 
        self.second = 0