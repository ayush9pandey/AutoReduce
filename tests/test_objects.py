
from autoreduce import load_ODE_model
from autoreduce import System
import numpy as np # type: ignore


# ## System object attributes


# Create symbolic objects
x, f, P = load_ODE_model(2, 2)
f[0] = -x[0]**2 + P[0]*x[1]
f[1] = -P[1]*x[1]
print('The states : {0}'.format(x))
print('The dynamics f : {0}'.format(f))
print('The dynamics P : {0}'.format(P))
C = np.array([[0,1]]).tolist()


# System class
sys = System(x, f, params = P, C = C)


# ## Solve the System using the ODE subclass 


# Solve ODE from System
from autoreduce.ode import ODE
timepoints = np.linspace(0,20,100)
sys.params_values = [2, 4]
sys.x_init = [0,10]
sys_ode = ODE(sys.x, sys.f, params = sys.params,
              params_values = sys.params_values,
              C = sys.C, x_init = sys.x_init,
              timepoints = timepoints)
solution = sys_ode.solve_system()


# ## Local sensitivity analysis tools for System using the SSM subclass


# Solve for sensitivity analysis from System
from autoreduce.local_sensitivity import SSM
timepoints = np.linspace(0,20,10)
sys.params_values = [2, 4]
sys.x_init = [0,10]
sys_ssm = SSM(sys.x, sys.f, params = sys.params,
              params_values = sys.params_values,
              C = sys.C, x_init = sys.x_init,
              timepoints = timepoints)
solution = sys_ssm.compute_SSM()

J = sys_ssm.compute_J([2,1])
Z = sys_ssm.compute_Zj([2,1], 1)
print('J = ',J)
print('Z = ', Z)
print('SSM = ',solution)

