import libsbml
import numpy as np


def create_sbml_model(compartment_id="default", time_units='second',
                      extent_units='mole', substance_units='mole',
                      length_units='metre', area_units='square_metre',
                      volume_units='litre', volume=1e-6,
                      model_id=None, **kwargs):
    """Creates an SBML Level 3 Version 2 model
    with some fixed standard settings.
    Refer to python-libsbml for more information on SBML API.
    :param compartment_id:
    :param time_units:
    :param extent_units:
    :param substance_units:
    :param length_units:
    :param area_units:
    :param volume_units:
    :param volume:
    :param model_id:
    :return:  the SBMLDocument and the Model object as a tuple
    """
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel()
    if model_id is None:
        model_id = 'autoreduce_'+str(np.random.randint(1, 1e6))
    model.setId(model_id)
    model.setName(model_id)
    # Define units for area
    unitdef = model.createUnitDefinition()
    unitdef.setId('square_metre')
    unit = unitdef.createUnit()
    unit.setKind(libsbml.UNIT_KIND_METRE)
    unit.setExponent(2)
    unit.setScale(0)
    unit.setMultiplier(1)

    # Set up required units and containers
    model.setTimeUnits(time_units)  # set model-wide time units
    model.setExtentUnits(extent_units)  # set model units of extent
    model.setSubstanceUnits(substance_units)  # set model substance units
    model.setLengthUnits(length_units)  # area units
    model.setAreaUnits(area_units)  # area units
    model.setVolumeUnits(volume_units)  # default volume unit
    compartment = model.createCompartment()
    compartment.setId(compartment_id)
    compartment.setVolume(1)
    compartment.setConstant(True)
    return document, model


def add_species(model, species, initial_condition=0):
    """Add species to SBML model

    Args:
        model (libsbml.Model): SBML model to add the species to
        species (str): Species string to add to model
        initial_condition (float, optional):Initial 
        condition of species. Defaults to 0.
    Returns:
        libsbml.Model: SBML Model object
    """
    default_compartment = model.getCompartment(0)
    sbml_species = model.createSpecies()
    sbml_species.setName(species)
    sbml_species.setId(species)
    sbml_species.setCompartment(default_compartment.getId())
    sbml_species.setConstant(False)
    sbml_species.setBoundaryCondition(False)
    sbml_species.setHasOnlySubstanceUnits(False)
    sbml_species.setSubstanceUnits('mole')
    sbml_species.setInitialConcentration(initial_condition)
    return model


def add_reaction(model: libsbml.Model, species: str,
                 kinetic_law: str, reaction_id: str, all_species: list):
    """Add reaction to given model

    Args:
        model (libsbml.Model): SBML model object
        species (str): Species name
        kinetic_law (str): Kinetic law for the species in the ODE
                           with sympy Symbols
        reaction_id (str): Reaction identifier string
        all_species (list): List of all species in System

    Returns:
        libsbml.Model : SBML Model object
    """
    sbml_reaction = model.createReaction()
    sbml_reaction.setId(reaction_id)
    sbml_reaction.setName(reaction_id)
    sbml_reaction.setReversible(False)
    # Create product
    product = sbml_reaction.createProduct()
    product.setSpecies(species)
    product.setConstant(False)
    # Create modifiers
    modifier_species = []
    for element in kinetic_law.free_symbols:
        if str(element) in all_species and str(element) != species:
            modifier_species.append(str(element))
    for modifier_id in modifier_species:
        modifier = sbml_reaction.createModifier()
        modifier.setSpecies(modifier_id)
    # Create kinetic law
    ratelaw = sbml_reaction.createKineticLaw()
    str_kinetic_law = str(kinetic_law)
    str_kinetic_law = str_kinetic_law.replace("**", "^")
    math_ast = libsbml.parseL3Formula(str_kinetic_law)
    flag = ratelaw.setMath(math_ast)
    if not flag == libsbml.LIBSBML_OPERATION_SUCCESS or math_ast is None:
        raise ValueError("Could not write the rate law for"
                         "reaction to SBML. Check the ODE"
                         "functions of species {0}.".format(species))
    return model


def add_parameters(model: libsbml.Model, all_parameters: list,
                   all_values: list):
    """Adds global parameters to the SBML Model

    Args:
        model (libsbml.Model): SBML Model to add global parameters to
        all_parameters (List): List of all parameter name symbols
        all_values (List): List of corresponding parameter values

    Returns:
        libsbml.Model: Updated SBML model object
    """
    for name, value in zip(all_parameters, all_values,):
        if model.getParameter(name) is None:
            param = model.createParameter()
            param.setId(name)
            param.setConstant(True)
            param.setValue(value)
        else:
            param = model.getParameter(name)
    return model
