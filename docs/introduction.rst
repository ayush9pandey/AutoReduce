Introduction
============

AutoReduce can be used to obtain smaller dynamical models from higher
dimensional models. We consider a generalized nonlinear system description as

.. math::

   \dot{x} = f(x, \theta, u), \qquad y = h(x, \theta, u),

where ``x`` is the state vector, ``theta`` is the parameter vector, ``u`` is an
optional input, and ``y`` is the measured or designed output.

Reduction Methods
-----------------

AutoReduce currently provides:

- Time-scale separation for quasi-steady-state approximation (QSSA).
- Conservation-law reduction for invariant total quantities.
- Local sensitivity analysis for ranking parameter effects
  and quantifying robustness of reduced models.

Model Sources
-------------

Models can be constructed directly with SymPy expressions. AutoReduce also
provides compatibility modules for common scientific modeling workflows:

- SBML import and export through python-libsbml.
- BioCRNpyler integration through SBML files.
- python-control `NonlinearIOSystem`.
