from os import listdir, getcwd, pardir, environ
from os.path import join, abspath
import asyncio
import sys

import pytest

# Set environment variable to use platformdirs
environ["JUPYTER_PLATFORM_DIRS"] = "1"

# Set event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

nb_not_installed = False
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
except ModuleNotFoundError:
    nb_not_installed = True


# Helper function to load and run notebooks with a given name at a given path
def run_notebook(filename, path):
    with open(filename) as nb_file:
        ep = ExecutePreprocessor()
        nb = nbformat.read(nb_file, nbformat.NO_CONVERT)
        try:
            ep.preprocess(nb, {"metadata": {"path": path}})
        except CellExecutionError:
            msg = f"\nError executing the notebook {join(path, filename)}\n"
            print(msg)
            raise


# Create a list of all notebooks to run
# ADD NEW EXAMPLE FOLDERS HERE IF NEEDED
cwd = getcwd()
paths = [join(cwd, "examples")]
nb_names = []
for p in paths:
    nb_names += [join(p, f) for f in listdir(p) if f[-6:] == ".ipynb"]


# create an iterative test to make sure each notebook runs without any errors.
@pytest.mark.parametrize("nb", nb_names)
@pytest.mark.skipif(
    nb_not_installed, reason="requires jupyter to be installed"
)
def test_jupyter_notebooks(nb):
    path = abspath(join(nb, pardir))
    run_notebook(nb, path)
