import asyncio
import os
import sys
from pathlib import Path

import pytest

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
os.environ["JUPYTER_ALLOW_INSECURE_WRITES"] = "1"
os.environ["PYTHONPATH"] = (
    str(Path.cwd())
    + os.pathsep
    + os.environ.get("PYTHONPATH", "")
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

nb_not_installed = False
try:
    import nbformat
    from nbconvert.preprocessors import (
        CellExecutionError,
        ExecutePreprocessor,
    )
except ModuleNotFoundError:
    nb_not_installed = True


def run_notebook(filename: Path):
    """Execute a notebook in its own directory."""
    with filename.open(encoding="utf-8") as nb_file:
        notebook = nbformat.read(nb_file, nbformat.NO_CONVERT)
    processor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        processor.preprocess(
            notebook,
            {"metadata": {"path": str(filename.parent)}},
        )
    except CellExecutionError:
        msg = f"\nError executing the notebook {filename}\n"
        print(msg)
        raise


ALL_NOTEBOOKS = sorted(Path("examples").rglob("*.ipynb"))
EXECUTED_NOTEBOOKS = [
    Path("examples/ecological/Viral spread.ipynb"),
]


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS)
@pytest.mark.skipif(
    nb_not_installed,
    reason="requires nbformat and nbconvert to be installed",
)
def test_notebook_is_valid_json(notebook):
    """Read every example notebook with nbformat."""
    with notebook.open(encoding="utf-8") as handle:
        nbformat.read(handle, nbformat.NO_CONVERT)


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook", EXECUTED_NOTEBOOKS)
@pytest.mark.skipif(
    nb_not_installed,
    reason="requires nbformat and nbconvert to be installed",
)
def test_smoke_jupyter_notebooks(notebook):
    """Execute lightweight smoke notebooks without errors."""
    run_notebook(notebook)
