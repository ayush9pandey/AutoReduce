# AutoReduce: An Automated Model Reduction Toolbox

Python toolbox to obtain reduced model expressions using time-scale
separation, conservation laws, sensitivity analysis, and projection-based
interfaces to established reduction libraries.

[![Build](https://github.com/ayush9pandey/autoreduce/actions/workflows/build.yml/badge.svg)](https://github.com/ayush9pandey/autoreduce/actions/workflows/build.yml)
[![Lint](https://github.com/ayush9pandey/autoreduce/actions/workflows/lint.yml/badge.svg)](https://github.com/ayush9pandey/autoreduce/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/ayush9pandey/AutoReduce/graph/badge.svg?token=h9diSMrJbj)](https://codecov.io/gh/ayush9pandey/AutoReduce)
[![PyPI version](https://badge.fury.io/py/autoreduce.svg)](https://badge.fury.io/py/autoreduce)
[![Documentation Status](https://readthedocs.org/projects/autoreduce/badge/?version=latest)](https://autoreduce.readthedocs.io/en/latest/?badge=latest)

## Overview

AutoReduce is a Python package for automated model reduction of nonlinear
dynamical systems. It provides tools for:

- Automated model reduction using QSSA (Quasi-Steady State Approximation)
- Conservation-law based reductions
- Local sensitivity analysis
- SBML import/export through python-libsbml
- Integration with [BioCRNPyler](https://biocrnpyler.readthedocs.io/) for synthetic biology models
- Optional python-control and PyDMD interfaces

See the [bioRxiv paper](https://doi.org/10.1101/2020.02.15.950840)
and [International Journal of Robust and Nonlinear Control paper](https://doi.org/10.1002/rnc.6013)
for the model-reduction background.

## Quick Start

```python
import numpy as np
from sympy import Symbol

from autoreduce import System, solve_timescale_separation

S = Symbol("S")
C = Symbol("C")
P = Symbol("P")
k1 = Symbol("k1")
k2 = Symbol("k2")
k3 = Symbol("k3")
E_total = Symbol("E_total")

E = E_total - C
x = [S, C, P]
f = [
    -k1 * E * S + k2 * C,
    k1 * E * S - (k2 + k3) * C,
    k3 * C,
]

system = System(
    x,
    f,
    params=[k1, k2, k3, E_total],
    params_values=[1.0, 0.5, 0.25, 1.0],
    x_init=[10.0, 0.0, 0.0],
    C=np.array([[1, 0, 0], [0, 0, 1]]),
)

reduced_system, collapsed_system = solve_timescale_separation(
    system,
    [S, P],
    fast_states=[C],
)
```

For more examples, check out the [documentation](https://autoreduce.readthedocs.io/en/latest/examples.html).

## Installation

Supported Python versions are 3.9 - 3.12. AutoReduce currently requires
NumPy < 2 to keep compatibility with BioCRNpyler.

Install the latest version of AutoReduce:

```bash
pip install autoreduce
```

Install with all optional dependencies:

```bash
pip install autoreduce[all]
```

Install optional integrations explicitly:

```bash
pip install "autoreduce[bio]"
pip install "autoreduce[control]"
pip install "autoreduce[dmd]"
```

For development installation:

```bash
git clone https://github.com/ayush9pandey/autoreduce.git
cd autoreduce
pip install -e ".[dev]"
```

## Documentation

Full documentation is available at [autoreduce.readthedocs.io](https://autoreduce.readthedocs.io/).

## Contributing

We welcome contributions. Developer notes and release instructions are in the
[documentation](https://autoreduce.readthedocs.io/en/latest/develop.html).

## Contact

For questions, feedback, or suggestions, please contact:
- Ayush Pandey (ayushpandey at ucmerced dot edu)
- [GitHub Issues](https://github.com/ayush9pandey/autoreduce/issues)

## License

Released under the BSD 3-Clause License (see `LICENSE`)

Copyright (c) 2025, Ayush Pandey. All rights reserved.
