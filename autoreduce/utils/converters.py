"""Model import and symbolic conversion utilities."""

from pathlib import Path

import numpy as np  # type: ignore
from libsbml import (
    LIBSBML_OPERATION_SUCCESS,
    LIBSBML_SEV_FATAL,
    ConversionProperties,
    readSBMLFromFile,
)
from sympy import Integer, Symbol, parse_expr  # type: ignore

from autoreduce.system.system import System


def _species_symbol_map(model, rename_species=None):
    """Map SBML species identifiers to SymPy symbols."""
    rename_species = {} if rename_species is None else dict(rename_species)
    species_ids = [species.getId() for species in model.getListOfSpecies()]
    unknown_species = sorted(set(rename_species) - set(species_ids))
    if unknown_species:
        unknown_names = ", ".join(str(species) for species in unknown_species)
        available_names = ", ".join(sorted(species_ids))
        raise ValueError(
            "Species rename keys did not match SBML species: "
            f"{unknown_names}. Available species are: {available_names}."
        )
    mapping = {}
    used_names = set()
    for species in model.getListOfSpecies():
        species_id = species.getId()
        new_name = str(rename_species.get(species_id, species_id))
        if new_name in used_names:
            raise ValueError(
                f"Species rename for {species_id!r} creates duplicate "
                f"symbol {new_name!r}."
            )
        used_names.add(new_name)
        mapping[species_id] = Symbol(new_name)
    return mapping


def load_ode_model(n_states, n_params=0, outputs=None):
    """Directly load an ODE skeleton with SymPy symbols."""
    x, f, P = ode_to_sympy(n_states, n_params)
    if outputs:
        output_names = [outputs] if isinstance(outputs, str) else list(outputs)
        states = [str(state) for state in x]
        missing_outputs = [
            output for output in output_names if output not in states
        ]
        if missing_outputs:
            raise ValueError(
                "Outputs did not match any state in the ODE model: "
                f"{missing_outputs}. Available states are: {states}."
            )
    return x, f, P


def ode_to_sympy(odesize, n_params=0):
    """
    Returns Sympy object for the given ODE function
    """
    from sympy import symbols  # type: ignore

    f = []
    x = []
    P = []
    for i in range(odesize):
        f.append(symbols("f%d" % i))
        x.append(symbols("x%d" % i))
    for k in range(n_params):
        P.append(symbols("P" + "%d" % k))
    return x, f, P


def sympy_to_sbml(model):
    """Return an SBML document for a SymPy-backed model.

    This conversion path is not implemented yet.
    """
    raise NotImplementedError("SymPy-to-SBML conversion is not implemented.")


# SBML to ODE #
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


def load_sbml(filename, outputs=None, rename_species=None, **kwargs):
    """Load an SBML file as a System object.

    Parameters
    ----------
    filename
        Path to the SBML file.
    outputs
        Optional SBML species identifier or list of SBML species identifiers
        to use as linear outputs. A row is added to ``C`` for each output.
    rename_species
        Optional mapping from SBML species identifiers to shorter symbol names
        used in the returned ``System``.

    The returned ``System`` has species in ``x``, dynamics in ``f``,
    parameter values in ``params_dict``, and initial conditions in ``x_init``.

    Returns: A System object
    """

    # Get the sbml file, check for errors, and perform conversions
    sbml_path = Path(filename)
    if not sbml_path.is_file():
        raise FileNotFoundError(f"SBML file not found: {filename}")
    filename = str(sbml_path)

    doc = readSBMLFromFile(filename)
    if doc.getNumErrors(LIBSBML_SEV_FATAL):
        raise ValueError(
            "Encountered serious errors while reading SBML file "
            f"{filename}:\n{doc.getErrorLog().toString()}"
        )
    doc.getErrorLog().clearLog()
    # Convert local params to global params
    props = ConversionProperties()
    props.addOption("promoteLocalParameters", True)
    if doc.convert(props) != LIBSBML_OPERATION_SUCCESS:
        print("The document could not be converted")
        print(doc.getErrorLog().toString())
    # Expand initial assignments
    props = ConversionProperties()
    props.addOption("expandInitialAssignments", True)
    if doc.convert(props) != LIBSBML_OPERATION_SUCCESS:
        print("The document could not be converted")
        print(doc.getErrorLog().toString())
    # Expand functions definitions
    props = ConversionProperties()
    props.addOption("expandFunctionDefinitions", True)
    if doc.convert(props) != LIBSBML_OPERATION_SUCCESS:
        print("The document could not be converted")
        print(doc.getErrorLog().toString())
    # Get model and define important lists, dictionaries
    mod = doc.getModel()
    if mod is None:
        raise ValueError(f"No SBML model found in file: {filename}")

    species_symbols = _species_symbol_map(mod, rename_species=rename_species)
    x = []
    x_init = []
    P = []
    params_values = []
    reactions = {}
    # Append species symbol to 'x' and append initial
    # amount/concentration to x_init
    # x[i] corresponds to x_init[i]
    for i in range(mod.getNumSpecies()):
        species = mod.getSpecies(i)
        x.append(species_symbols[species.getId()])
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
        formula = kinetics.getFormula()
        # Create a mapping of species/parameter IDs to their symbols
        symbol_map = dict(species_symbols)
        for param in mod.getListOfParameters():
            symbol_map[param.getId()] = Symbol(param.getId())
        # Parse the formula using sympy's parse_expr with local_dict
        reactions[reaction.getId()] = parse_expr(
            formula, local_dict=symbol_map
        )
    # Define f
    f = [Integer(0)] * len(x)
    # Loop to define functions in 'f'
    for i in range(mod.getNumReactions()):
        reaction = mod.getReaction(i)
        # subtract reactant kinetic formula
        for j in range(reaction.getNumReactants()):
            ref = reaction.getReactant(j)
            species = species_symbols[ref.getSpecies()]
            curr_index = x.index(species)
            if ref.getStoichiometry() == 1.0:
                f[curr_index] += -reactions[reaction.getId()]
            else:
                f[curr_index] += (
                    -reactions[reaction.getId()] * ref.getStoichiometry()
                )
        # add product kinetic formula
        for j in range(reaction.getNumProducts()):
            ref = reaction.getProduct(j)
            species = species_symbols[ref.getSpecies()]
            curr_index = x.index(species)
            if ref.getStoichiometry() == 1.0:
                f[curr_index] += +reactions[reaction.getId()]
            else:
                f[curr_index] += (
                    +reactions[reaction.getId()] * ref.getStoichiometry()
                )
    if outputs is None or outputs == []:
        C = None
    else:
        output_names = [outputs] if isinstance(outputs, str) else list(outputs)
        species_ids = [species.getId() for species in mod.getListOfSpecies()]
        missing_outputs = [
            output for output in output_names if output not in species_ids
        ]
        if missing_outputs:
            raise ValueError(
                "Outputs did not match any species in the SBML model: "
                f"{missing_outputs}. Available species are: {species_ids}."
            )
        C = np.zeros((len(output_names), len(x)))
        for output_count, output in enumerate(output_names):
            index_output = x.index(species_symbols[output])
            C[output_count, index_output] = 1
            print(f"Your output {output!r} is now set using the system.C matrix!")
    sys = System(
        x,
        f,
        params_dict=dict(zip(P, params_values)),
        x_init=x_init,
        C=C,
        **kwargs,
    )
    return sys
