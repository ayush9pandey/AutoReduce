"""Generate autosummary fragments for the AutoReduce reference manual."""

import ast
from pathlib import Path


AUTOSUMMARY = """
.. autosummary::
   :toctree: generated/
   :nosignatures:
"""


def get_public_objects(py_path):
    """Return public top-level classes and functions in a Python file."""
    with py_path.open(encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=str(py_path))
    public_objects = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if not node.name.startswith("_"):
                public_objects.append(node.name)
    return public_objects


def write_section(handle, title, module_name, objects):
    """Write one module section to an RST file handle."""
    handle.write(f"{title}\n")
    handle.write("-" * len(title) + "\n\n")
    handle.write(f".. automodule:: {module_name}\n\n")
    if objects:
        handle.write(AUTOSUMMARY)
        handle.write("\n")
        for obj in objects:
            handle.write(f"   {module_name}.{obj}\n")
        handle.write("\n")


def generate_module_rst(package_dir, docs_dir, package_name, output_name):
    """Generate an RST fragment for all modules under a package directory."""
    output_path = docs_dir / output_name
    with output_path.open("w", encoding="utf-8") as handle:
        for py_path in sorted(package_dir.rglob("*.py")):
            if py_path.name == "__init__.py":
                continue
            relative = py_path.relative_to(package_dir.parent)
            module_name = ".".join(
                [package_name] + list(relative.with_suffix("").parts[1:])
            )
            title = module_name.replace("_", " ")
            objects = get_public_objects(py_path)
            write_section(handle, title, module_name, objects)


if __name__ == "__main__":
    docs = Path(__file__).resolve().parent
    root = docs.parent
    package = root / "autoreduce"
    generate_module_rst(package, docs, "autoreduce", "_autogen_library.rst")
