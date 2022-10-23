from libsbml import *
import sys
import numpy as np
from sympy import Symbol,sympify
from .model_reduction import Reduce

def load_ODE_model(n_states, n_params = 0):
    return ode_to_sympy(n_states, n_params)

def ode_to_sympy(odesize, n_params = 0):
    '''
    Returns Sympy object for the given ODE function
    '''
    from sympy import symbols 
    f = []
    x = []
    P = []
    for i in range(odesize):
        str_var = 'x' + str(i)
        str_f = 'f' + str(i)
        vars()[str_f] = symbols('f%d'%i)
        vars()[str_var] = symbols('x%d'%i)
        f.append(vars()[str_f])
        x.append(vars()[str_var])
    for k in range(n_params):
        str_P = 'P' + str(k)
        vars()[str_P] = symbols('P' + '%d'%k)
        P.append(vars()[str_P])
    return x, f, P


def sympy_to_sbml(model):
    sbml_doc = None
    return sbml_doc


#### SBML to ODE #####

# This file reads an SBML file using libSBML, 
#
# - expands all function definitions
# - expands all initial assignments
# - converts local parameter to global ones
# - then it goes ahead and write the ODE system
#   for use with scipy.integrate
# - it emits a function called simulateModel
#   that takes three parameters: t0, tend and numpoints
#   with that the model can be simulated as needed
# - finally the emitted function is called and the result plotted 
# - it is also written out into a file called generated.py 
#
 
 


def load_sbml(filename, **kwargs):
    '''A function that takes in an SBML file and returns x,f,P,params_values.
    x is a list of species written as Sympy objects
    f is a list of functions written as Sympy objects
    P is a list of parameters written as Sympy objects
    params_values is a list of parameter values, in the same order as P
    x_init is a list of initial conditions, in the same order as x

    Returns: A reducible Reduce(System) object
    '''

    # Get the sbml file, check for errors, and perform conversions
    doc = readSBMLFromFile(filename)
    if doc.getNumErrors(LIBSBML_SEV_FATAL):
        print('Encountered serious errors while reading file')
        print(doc.getErrorLog().toString())
        sys.exit(1)
    doc.getErrorLog().clearLog()
    # Convert local params to global params
    props = ConversionProperties()
    props.addOption("promoteLocalParameters", True)
    if doc.convert(props) != LIBSBML_OPERATION_SUCCESS: 
        print('The document could not be converted')
        print(doc.getErrorLog().toString())
    # Expand initial assignments
    props = ConversionProperties()
    props.addOption("expandInitialAssignments", True)
    if doc.convert(props) != LIBSBML_OPERATION_SUCCESS: 
        print('The document could not be converted')
        print(doc.getErrorLog().toString())
    # Expand functions definitions
    props = ConversionProperties()
    props.addOption("expandFunctionDefinitions", True)
    if doc.convert(props) != LIBSBML_OPERATION_SUCCESS: 
        print('The document could not be converted')
        print(doc.getErrorLog().toString())
    # Get model and define important lists, dictionaries
    mod = doc.getModel()
    x = []
    x_init = []
    P = []
    params_values = []
    reactions = {}
    # Append species symbol to 'x' and append initial amount/concentration to x_init
    # x[i] corresponds to x_init[i]
    for i in range(mod.getNumSpecies()):
        species = mod.getSpecies(i)
        x.append(Symbol(species.getId()))
        if species.isSetInitialConcentration():
            x_init.append(species.getInitialConcentration())
        elif species.isSetInitialAmount():
            x_init.append(species.getInitialAmount())
        else:
            x_init.append(0)
    # Append parameter symbol to 'P' and parameter values to 'params_values'
    for i in range(mod.getNumParameters()):
        params = mod.getParameter(i)
        params_values.append(params.getValue())
        P.append(Symbol(params.getId()))
    # Get kinetic formula for each reaction, store in dictionary 'reactions'
    for i in range(mod.getNumReactions()):
        reaction = mod.getReaction(i)
        kinetics = reaction.getKineticLaw()
        reactions[reaction.getId()] = sympify(kinetics.getFormula())
    # Define f
    f = [sympify(0)] * len(x)
    # Loop to define functions in 'f'
    for i in range(mod.getNumReactions()):
        reaction = mod.getReaction(i)
        # subtract reactant kinetic formula
        for j in range(reaction.getNumReactants()):
            ref = reaction.getReactant(j)
            species = sympify(mod.getSpecies(ref.getSpecies()).getId())
            curr_index = x.index(species)
            # Check stoichiometry
            if ref.getStoichiometry() == 1.0: 
                f[curr_index] += -reactions[reaction.getId()]
            else:
                f[curr_index] += -reactions[reaction.getId()]*ref.getStoichiometry()
        # add product kinetic formula
        for j in range(reaction.getNumProducts()):
            ref = reaction.getProduct(j)
            species = sympify(mod.getSpecies(ref.getSpecies()).getId())
            curr_index = x.index(species)
            # Check stoichiometry
            if ref.getStoichiometry() == 1.0:
                f[curr_index] += +reactions[reaction.getId()]
            else:
                f[curr_index] += +reactions[reaction.getId()]*ref.getStoichiometry()
    if 'outputs' in kwargs:
        outputs = kwargs['outputs']
        if type(outputs) is not list:
            outputs = [outputs]
        C = np.zeros( (len(outputs), len(x)))
        output_count = 0
        for output in outputs:
            index_output = x.index(sympify(output))
            C[output_count, index_output] = 1
            output_count += 1
    else:
        C = None
    sys = Reduce(x, f, params = P, params_values = params_values, x_init = x_init, C = C, **kwargs)
    return sys
