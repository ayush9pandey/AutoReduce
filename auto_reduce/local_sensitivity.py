from system import System
import numpy as np
from ode import ODE
from scipy.integrate import solve_ivp

class SSM(System):
    '''
    Class that computes local sensitivity analysis coefficients for the given Model using a numerical 
    approximation method discussed in doi: https://doi.org/10.1016/0021-9991(76)90007-3
    Uses numerical method to find sensitivity analysis matrix (SSM).
    Both the Jacobian matrix and the Z matrix are estimated using 4th
    order central difference as given in the paper.
    '''
    def __init__(self, x, f, params = None, C = None, g = None, h = None, 
                params_values = [], x_init = [], timepoints = None):
        super().__init__(x, f, params, C, g, h, params_values, x_init)
        if timepoints is None:
            timepoints = []
        else:
            self.timepoints = timepoints
        return

    def compute_Zj(self, x, j):
        '''
        Compute Z_j, i.e. df/dp_j at a particular timepoint k for the parameter p_j. 
        Returns a vector of size n x 1. 
        '''
        # initialize Z
        Z = np.zeros(self.n)    
        P_holder = self.params_values
        # For each state
        for i in range(self.n):
            P = P_holder
            F = np.zeros( (4,1) ) # For 4th order difference
            h = P[j]*0.01 # Small parameter for this parameter
            # Gets O(4) central difference on dfi/dpj
            if h != 0:
                P[j] = P_holder[j] + 2*h
                f = self.evaluate(self.f, x, P)
                F[0] = f[i]
                P[j] = P_holder[j] + h
                f = self.evaluate(self.f, x, P)
                F[1] = f[i]
                P[j] = P_holder[j] - h
                f = self.evaluate(self.f, x, P)
                F[2] = f[i]
                P[j] = P_holder[j] - 2*h
                f = self.evaluate(self.f, x, P)
                F[3] = f[i]
                #Store approx. dfi/dpj into Z
                Z[i] = (-F[0] + 8*F[1] - 8*F[2] + F[3])/(12*h)   
        return Z

    def compute_J(self, x):
        '''
        Compute the Jacobian J = df/dx at a timepoint k.
        Returns a matrix of size n x n.
        '''
        # initialize J
        J = np.zeros( (self.n, self.n) )   
        P = self.params_values 
        # store x
        X = x 
        for i in range(self.n):
            for j in range(self.n):
                F = np.zeros( (4,1) )
                h = X[j]*0.01
                # Gets O(4) central difference on dfi/dxj
                if h != 0:
                    x = X
                    x[j] = X[j] + 2*h
                    f = self.evaluate(self.f, x, P)
                    F[0] = f[i]
                    x[j] = X[j] + h
                    f = self.evaluate(self.f, x, P)
                    F[1] = f[i]
                    x[j] = X[j] - h
                    f = self.evaluate(self.f, x, P)
                    F[2] = f[i]
                    x[j] = X[j] - 2*h
                    f = self.evaluate(self.f, x, P)
                    F[3] = f[i]
                    #Store approx. dfi/dxj into J
                    J[i,j]= (-F[0] + 8*F[1] - 8*F[2] + F[3])/(12*h)   
        return J

    def compute_SSM(self, normalize = False):
        '''
        Returns the sensitivity coefficients S_j for each parameter p_j. 
        The sensitivity coefficients are written in a sensitivity matrix SSM of size len(timepoints) x len(params) x n
        If normalize argument is true, the coefficients are normalized by the nominal value of each paramneter.
        '''
        def sens_func(t, x, J, Z):
            # forms ODE to solve for sensitivity coefficient S
            dsdt = J@x + Z
            return dsdt
        P = self.params_values
        S0 = np.zeros(self.n) # Initial value for S_i  
        SSM = np.zeros( (len(self.timepoints), len(P), self.n) )
        # solve for all x's in timeframe set by timepoints
        sol = ODE(self.x, self.f, params = self.params, params_values = self.params_values,
                C = self.C, g = self.g, h = self.h, x_init = self.x_init, timepoints = self.timepoints).solve_system()
        xs = sol.y
        xs = np.reshape(xs,(len(self.timepoints), self.n))
        self.xs = xs
        # Solve for SSM at each time point 
        for k in range(len(self.timepoints)): 
            timepoints = self.timepoints[0:k+1]
            if len(timepoints) == 1:
                continue
            t_span = (timepoints[0], timepoints[-1])
            # get the jacobian matrix
            J = self.compute_J(xs[k,:])
            #Solve for S = dx/dp for all x and all P (or theta, the parameters) at time point k
            for j in range(len(P)): 
                # get the pmatrix
                Zj = self.compute_Zj(xs[k,:], j)
                # solve for S
                sens_func_ode = lambda t, x : sens_func(t, x, J, Zj)
                sol = solve_ivp(sens_func_ode, t_span, S0, t_eval = timepoints)
                S = sol.y
                S = np.reshape(S, (len(timepoints), self.n))
                SSM[k,j,:] = S[k,:]
        self.SSM = SSM
        if normalize:
            SSM = self.normalize_SSM() #Identifiablity was estimated using an normalized SSM
        return SSM

    def normalize_SSM(self):
        '''
        Returns normalized sensitivity coefficients. 
        Multiplies each sensitivity coefficient with the corresponding parameter p_j
        Divides the result by the corresponding state to obtain the normalized coefficient that is returned.
        '''
        SSM_normalized = np.zeros(np.shape(self.SSM))
        for j in range(len(self.params_values)):
            for i in range(self.n):
                SSM_normalized[:,j,i] = np.divide(self.SSM[:,j,i]*self.params_values[j], self.xs[:,i]) 
        self.SSM_normalized = SSM_normalized
        return SSM_normalized

