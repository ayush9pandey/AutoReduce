.. currentmodule:: autoreduce

**********
Reductions
**********

AutoReduce implements reduction algorithms in `autoreduce.reductions`.

Time-Scale Separation
=====================

Use `solve_timescale_separation` for QSSA-style reductions from a plain
`System`. `reductions.timescale.Reduce` remains available for advanced
search workflows.

Conservation Laws
=================

Use `solve_conservation_laws` to apply conservation-law reductions from a
plain `System`. When conserved sets are detected automatically, the
`conservation_search_depth` argument controls how many ODE terms are summed
while looking for cancellations.

Projection Methods
==================

Projection-based DMD and DMDc wrappers are implemented in
`autoreduce.reductions.projection.dmd`. They require the `dmd` extra.

.. code-block:: bash

   pip install "autoreduce[dmd]"

Abundance-Based Methods
=======================

The `autoreduce.reductions.abundance` module marks the package location for
abundance-based reduction methods. The implementation currently raises
`NotImplementedError` until the algorithm is added.
