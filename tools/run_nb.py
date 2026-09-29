"""Execute a notebook in place, top to bottom, in a fresh kernel.

`allow_errors=False` is the point: the assignment requires a notebook that runs
top to bottom, so any cell that raises must fail the build rather than land in
the repo with a traceback in its output.
"""
import sys, nbformat
from nbclient import NotebookClient

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                        resources={"metadata": {"path": "."}}, allow_errors=False)
client.execute()
nbformat.write(nb, path)
n_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.outputs)
print(f"{path}: executed {len(nb.cells)} cells, {n_out} produced output")
