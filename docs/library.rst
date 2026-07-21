**********************
The AutoReduce Library
**********************

This chapter documents the public modules, classes, and functions that make up
AutoReduce.

Systems
=======

.. automodule:: autoreduce.system.system

.. autosummary::
   :toctree: generated/
   :nosignatures:

   System

.. automodule:: autoreduce.system.control

.. autosummary::
   :toctree: generated/
   :nosignatures:

   from_nonlinear_io_system

.. automodule:: autoreduce.system.pydmd

.. autosummary::
   :toctree: generated/
   :nosignatures:

   from_linear_operator
   from_dmd_model

Solvers
=======

.. automodule:: autoreduce.solvers.ode

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ODE

.. automodule:: autoreduce.solvers.ssm

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SSM

Reductions
==========

.. automodule:: autoreduce.reductions.core

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Reduce
   ReduceUtils
   get_error_metric
   get_robustness_metric
   create_system

.. automodule:: autoreduce.reductions.timescale

.. autosummary::
   :toctree: generated/
   :nosignatures:

   solve_timescale_separation
   explore_all_QSS_models
   reduce_with_input

.. automodule:: autoreduce.reductions.utils

.. autosummary::
   :toctree: generated/
   :nosignatures:

   sympy_variables_exist
   sympy_solve_and_substitute
   sympy_get_steady_state_solutions

.. automodule:: autoreduce.reductions.conservation

.. autosummary::
   :toctree: generated/
   :nosignatures:

   find_conserved_sets
   setup_conservation_laws
   apply_conservation_laws
   solve_conservation_laws

.. automodule:: autoreduce.reductions.abundance

.. autosummary::
   :toctree: generated/
   :nosignatures:

   solve_approximations

.. automodule:: autoreduce.reductions.projection.dmd

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DMDReduction
   fit_dmd
   fit_dmdc

Utilities
=========

.. automodule:: autoreduce.utils.converters

.. autosummary::
   :toctree: generated/
   :nosignatures:

   load_ODE_model
   ode_to_sympy
   sympy_to_sbml
   load_sbml

.. automodule:: autoreduce.solvers.utils

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_ODE
   solve_ODE_SSM
   get_SSM
   solve_ode
   solve_ssm
   solve_sensitivity
   get_ode_solutions

.. automodule:: autoreduce.utils.sbml

.. autosummary::
   :toctree: generated/
   :nosignatures:

   create_sbml_model
   add_species
   add_reaction
   add_parameters
