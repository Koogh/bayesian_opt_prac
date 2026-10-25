#!/usr/bin/env python
"""Assemble the tutorial notebook from cells_part1/2."""
import sys
from pathlib import Path
import nbformat as nbf

BUILD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BUILD_DIR.parent
sys.path.insert(0, str(BUILD_DIR))
from cells_part1 import CELLS as C1  # noqa: E402
from cells_part2 import CELLS as C2  # noqa: E402

cells = C1 + C2
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python (llm)", "language": "python", "name": "llm"},
    "language_info": {"name": "python"},
}
for kind, src in cells:
    src = src.strip("\n") + "\n"
    if kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

out = PROJECT_DIR / "bayesian_optimization_tutorial.ipynb"
nbf.write(nb, out)
n_md = sum(1 for k, _ in cells if k == "md")
print(f"wrote {out}: {len(cells)} cells ({n_md} markdown, {len(cells) - n_md} code)")
