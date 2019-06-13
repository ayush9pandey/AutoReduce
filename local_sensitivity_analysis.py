###############################
#### Code contributed by : Samuel E. Clamons.
#### Used with permission.
###############################
import numdifftools as nd
import numpy as np
import scipy.integrate
import copy

def sensitivity_to_parameter(ode_sol, ode_jac, ode, params, n_vars, p, t_min,
                             t_max):
    '''
    Calculates the response of each derivative (defined by ode) to changes
    in a single parameter (the jth one) over time.

    params:
        ode_sol - An OdeSolution object holding a continuously-interpolated
                    solution for ode.
        ode_jac - Jacobian of the ode, as calculated by numdifftools.Jacobian.
        ode - The ODE for the system, of the form ode(t, x, params)
        params - A list of parameters to feed to ode.
        p - The index of the parameter to calculate sensitivities to.
        t_min - Starting time.
        t_max - Ending time.
    returns: An OdeSolution object representing the (continously-interpolated)
                sensitivity of each variable to the specified parameter over
                time.
    '''
    # Build a scipy-integratable derivative-of-sensitivity function.
    def dS_dt(t, s):
        xs = ode_sol(t)
        # Wrapper to let numdifftools calculate df/dp.
        def ode_as_parameter_call(param):
            call_params = copy.deepcopy(params)
            call_params[p] = param
            return ode(t, xs, call_params)

        df_dp = lambda xs: nd.Jacobian(ode_as_parameter_call)(xs).transpose()[:,0]

        return df_dp(params[p]) + np.matmul(ode_jac(xs), s)

    sol = scipy.integrate.solve_ivp(dS_dt, (t_min, t_max), np.zeros(n_vars),
                                    dense_output = True)
    return sol.sol


def solve_extended_ode(ode, params, t_min, t_max, init, method = "RK45"):
    '''
    Augments an ODE system (as a scipy-integratable function) into an ODE
    representing the original ODE plus sensitivities, then solves them all.

    The key equation here is, for a system 
    
    dx/dt = f(x, p, t),

    dS_j/dt = f_j + J*S_j

    where S_j is the vector of sensitivities of xs to parameter j, f_j is the
    vector of derivatives df/dp_j, and J is the Jacobian of f w.r.t. xs.

    params:
        ode - An ode to solve, with a signature ode(t, xs, parameters).
        params - a vector of parameters around which to calculate sensitivity.
        t_min - Starting time.
        t_max - Ending time.
        init - Initial conditions for the ode.
        method - ODE solving method, passed directly to
                    scipy.integrate.solve_ivp.

    Returns: (x_sols, sensitivities)
        x_sols - An OdeSolution object with the solution to the original ODE.
                    Shape of a solution is (n_variables, n_times)
        sensitivities - An array of OdeSolution objects of  size (n_params)
                        holding sensitivities of each variable to each
                        parameter over time, as a continuous interpolation.
                        Shape of a solution is (n_variables, n_times)
    '''
    n_variables = len(init)
    n_params    = len(params)

    # Solve ODE.
    ode_func = lambda t, xs: ode(t, xs, params)
    ode_jac  = nd.Jacobian(lambda x: ode_func(0, x))
    sol = scipy.integrate.solve_ivp(ode_func, (t_min, t_max), init,
                                    method = method, dense_output = True,
                                    jac = lambda t, x: ode_jac(x))

    if sol.status != 0:
        raise RuntimeException("In solve_extended_ode, solve_ivp failed with "
                               " error message: " + sol.message)

    sensitivities = [None] * n_params
    for p in range(n_params):
        print("\rSolving sensitivity for parameter %d/%d       " \
              % (p+1, n_params))
        sensitivities[p] = sensitivity_to_parameter(sol.sol, ode_jac, ode,
                                                        params, n_variables, p,
                                                        t_min, t_max)

    return(sol.sol, sensitivities)