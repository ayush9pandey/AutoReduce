.. currentmodule:: autoreduce

*******
Systems
*******

AutoReduce represents models with `system.system.System`. A system stores
symbolic states, symbolic dynamics, parameters, initial conditions, and either
a linear output matrix or a nonlinear output function.

Core System
===========

Use `System` when model equations are already available as SymPy expressions.

.. code-block:: python

   from sympy import Symbol

   from autoreduce import System

   x = Symbol("x")
   k = Symbol("k")
   system = System([x], [-k * x], params_dict={k: 1.0}, x_init=[2.0])

Parameters can be read and updated through ``params_dict``:

.. code-block:: python

   system.get_param(k)
   system.set_param(k, 0.5)
   system.set_param_dict({k: 2.0})

python-control Adapter
======================

The python-control adapter is implemented in `autoreduce.system.control` and
requires the `control` extra.

.. code-block:: bash

   pip install "autoreduce[control]"

The adapter converts a `control.NonlinearIOSystem` into an AutoReduce
`System` by evaluating the python-control update and output functions with
symbolic states, inputs, and parameters.

PyDMD Adapter
=============

The PyDMD adapter is implemented in `autoreduce.system.pydmd`. It converts a
DMD operator matrix, or a fitted PyDMD model when PyDMD is installed, into a
linear symbolic `System`.
