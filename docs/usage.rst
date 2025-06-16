Usage
=====

Basic Usage
----------

Here's a simple example of how to use autoReduce:

.. code-block:: python

    from autoreduce import System
    from autoreduce.utils import get_reducible

    # Create a system
    x = [Symbol('x1'), Symbol('x2')]
    f = [-x[0] + x[1], -x[1]]
    system = System(x, f)

    # Get reducible system
    reducible_system = get_reducible(system)

    # Get reduced model
    reduced_system, collapsed_system = reducible_system.solve_timescale_separation([x[0]])

For more examples, see the :doc:`examples` section.
