########################################################
AutoReduce: An Automated Model Reduction Toolbox
########################################################

AutoReduce is a tool for obtaining reduced-order models of
nonlinear dynamical systems using symbolic computation in Python.
It is designed for workflows where the reduced model should remain
interpretable. That is, users provide symbolic system
equations, select states or constraints that define the reduction problem,
and obtain reduced dynamics that can be inspected, simulated, and exported.

More specifically, the package supports model reduction by
time-scale separation, conservation laws, abundance assumptions,
local sensitivity analysis, and (coming soon!) projection-based methods.
Models can be imported in multiple formats:
`SymPy <https://www.sympy.org/>`_,
`SBML <http://sbml.org/>`_,
`BioCRNpyler <https://github.com/buildacell/biocrnpyler>`_,
`python-control's NonlinearIOSystem <https://python-control.readthedocs.io/>`_,
and PyDMD. The reduced models are returned as symbolic dynamics that
can be exported through supported compatibility routes.

.. rubric:: Main features

- Quasi-steady-state approximation (QSSA) through time-scale separation.
- Numerical simulation of full and reduced systems.
- Quantification of error between full and reduced models.
- Conservation-law reduction for systems with invariant total quantities.
- Local sensitivity analysis for parameter-dependent systems to
  rank (and choose) parameter effects.
- Robustness computation for reduced models that are closest to the full model
  even under perturbations of the parameters.

.. rubric:: Background

AutoReduce was originally developed for automated construction of
phenomenological models from more detailed biological circuit descriptions.
The original workflow combines time-scale separation, conservation laws, and
species abundance assumptions to produce smaller models that retain
the input-output relationships needed for design analysis [PandeyMurray2020]_.

The robustness tools in AutoReduce are connected to structured model
reduction guarantees for dynamical systems, with biomolecular examples
developed in [PandeyMurray2023]_.

Compatibility notes:

- SymPy is used for symbolic ODE definitions.
- SciPy is used for numerical ODE simulation.
- python-libsbml supports SBML import and export.
- BioCRNpyler models can be used through SBML-based workflows.
- python-control `NonlinearIOSystem` models are supported for model imports.
- PyDMD-based DMD and DMDc reductions are supported by the ``dmd`` extra.

.. rubric:: Related research

The optional PyDMD and python-control integrations build on [Demo2018]_ and
[Fuller2021]_, respectively.

.. [PandeyMurray2020] Ayush Pandey and Richard M. Murray. "Model Reduction
   Tools For Phenomenological Modeling of Input-Controlled Biological
   Circuits." bioRxiv, 2020. https://doi.org/10.1101/2020.02.15.950840

.. [PandeyMurray2023] Ayush Pandey and Richard M. Murray. "Robustness
   guarantees for structured model reduction of dynamical systems with
   applications to biomolecular models." *International Journal of Robust and
   Nonlinear Control*, 33(9):5058-5086, 2023.
   https://doi.org/10.1002/rnc.6013

.. [Demo2018] Nicola Demo, Marco Tezzele, and Gianluigi Rozza. "PyDMD:
   Python Dynamic Mode Decomposition." *Journal of Open Source Software*,
   3(22):530, 2018. https://doi.org/10.21105/joss.00530

.. [Fuller2021] Sawyer Fuller, Ben Greiner, Jason Moore, Richard Murray,
   Rene van Paassen, and Rory Yorke. "The Python Control Systems Library
   (python-control)." In *2021 60th IEEE Conference on Decision and Control
   (CDC)*, 4875-4881, 2021. https://doi.org/10.1109/CDC45484.2021.9683368

.. toctree::
   :caption: User Guide
   :maxdepth: 1
   :numbered: 2

   introduction
   installation
   usage
   systems
   solvers
   reductions
   examples

.. toctree::
   :caption: Reference Manual
   :maxdepth: 1

   library
   develop
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
