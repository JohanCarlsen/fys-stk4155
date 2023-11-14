import autograd.numpy as np

class Solvers:
    def __init__(self, eta):
        self.eta = eta 
    
    def update_change(self, gradient):
        raise NotImplementedError
    
    def reset(self):
        pass 

class Constant(Solvers):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient 
    
    def reset(self):
        pass 

class ADAM(Solvers):

    def __init__(self, eta, rho1=0.9, rho2=0.999):
        super().__init__(eta)
        self.rho1 = rho1 
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8

        self.moment = self.rho1 * self.moment \
                    + (1 - self.rho1) * gradient 
        
        self.second = self.rho2 * self.second \
                    + (1 - self.rho2) * gradient * gradient 
        
        moment_corrected = self.moment / (1 - self.rho1**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))
    
    def reset(self):
        self.n_epochs += 1
        self.moment = 0 
        self.second = 0