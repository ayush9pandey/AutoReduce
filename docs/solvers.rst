.. currentmodule:: autoreduce

*******
Solvers
*******

AutoReduce provides numerical solvers in `autoreduce.solvers`.

ODE Solver
==========

`solvers.ode.ODE` evaluates a symbolic `System` and solves the resulting ODE
with SciPy.
For most workflows, call `solve_ode(system, timepoints)` directly.

Sensitivity Solver
==================

`solvers.ssm.SSM` computes local sensitivity coefficients and Jacobian
approximations for symbolic systems.
For most workflows, call `solve_sensitivity(system, timepoints)` directly.
