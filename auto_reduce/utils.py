from auto_reduce import ode
from auto_reduce import local_sensitivity
from auto_reduce import model_reduction 

def get_ODE(system_obj, timepoints, **kwargs):
    '''
    For the given timepoints, create an ODE class object for this System object.
    '''
    ode_obj = ode.ODE(system_obj.x, system_obj.f, params = system_obj.params,
                params_values = system_obj.params_values, x_init = system_obj.x_init, 
                timepoints = timepoints, **kwargs)
    return ode_obj

def solve_ODE_SSM(system_obj, timepoints_ode, timepoints_ssm, **kwargs):
    '''
    For the given timepoints, returns the full solution (states, sensitivity coefficients, outputs)
    '''
    ode = system_obj.get_ODE(timepoints_ode)
    ssm = system_obj.get_SSM(timepoints_ssm)
    x_sol = ode.solve_system().y
    y = system_obj.C@x_sol
    Ss = ssm.compute_SSM()
    return x_sol, y, Ss

def get_SSM(system_obj, timepoints, **kwargs):
    '''
    For the given timepoints, create an SSM class object for this System object.
    '''
    ssm_obj = local_sensitivity.SSM(system_obj.x, system_obj.f, params = system_obj.params,
                params_values = system_obj.params_values, x_init = system_obj.x_init, 
                timepoints = timepoints, **kwargs)
    return ssm_obj

def reduce(system_obj, timepoints_ode, timepoints_ssm, **kwargs):
    red_obj = model_reduction.Reduce(system_obj.x, system_obj.f, C = system_obj.C, 
                params = system_obj.params, g = system_obj.g, h = system_obj.h,
                params_values = system_obj.params_values, x_init = system_obj.x_init, 
                timepoints_ode = timepoints_ode, timepoints_ssm = timepoints_ssm, **kwargs)
    return red_obj