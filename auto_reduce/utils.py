from auto_reduce import ode
from auto_reduce import local_sensitivity
from auto_reduce import model_reduction 

def get_ODE(system_obj, timepoints, **kwargs):
    '''
    For the given timepoints, create an ODE class object for this System object.
    '''
    ode_obj = ode.ODE(system_obj.x, system_obj.f, C = system_obj.C, g = system_obj.g, h = system_obj.h, u = system_obj.u,
                params = system_obj.params, params_values = system_obj.params_values, 
                x_init = system_obj.x_init, timepoints = timepoints, **kwargs)
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
    ssm_obj = local_sensitivity.SSM(system_obj.x, system_obj.f, g = system_obj.g,
                C = system_obj.C, h = system_obj.h, u = system_obj.u, params = system_obj.params,
                params_values = system_obj.params_values, x_init = system_obj.x_init, 
                timepoints = timepoints, **kwargs)
    return ssm_obj

def reduce(system_obj, timepoints_ode, timepoints_ssm, **kwargs):
    red_obj = model_reduction.Reduce(system_obj.x, system_obj.f, C = system_obj.C, 
                params = system_obj.params, g = system_obj.g, h = system_obj.h, u = system_obj.u,
                params_values = system_obj.params_values, x_init = system_obj.x_init, 
                timepoints_ode = timepoints_ode, timepoints_ssm = timepoints_ssm, **kwargs)
    return red_obj

def reduce_utils(reduce_obj, **kwargs):
    reduce_utils_obj = model_reduction.ReduceUtils(reduce_obj.x, reduce_obj.f, C = reduce_obj.C, 
                params = reduce_obj.params, g = reduce_obj.g, h = reduce_obj.h, u = reduce_obj.u,
                params_values = reduce_obj.params_values, x_init = reduce_obj.x_init, 
                timepoints_ode = reduce_obj.timepoints_ode, timepoints_ssm = reduce_obj.timepoints_ssm,
                error_tol=reduce_obj.error_tol, nstates_tol=reduce_obj.nstates_tol)
    return reduce_utils_obj

def get_ode_solutions(system_obj, timepoints, **kwargs):
    x_sol = get_ODE(system_obj, timepoints, **kwargs).solve_system().T
    return x_sol


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()