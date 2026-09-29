"""Extract every cell's stdout from the executed notebook into a plain-text log.

The log is committed so the README's numbers can be checked without opening the
notebook or rerunning anything.
"""
import sys, nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
out = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code" or not cell.get("outputs"):
        continue
    text = "".join(
        o.get("text", "") for o in cell.outputs if o.output_type == "stream"
    )
    if text.strip():
        out.append(f"{'=' * 78}\n=== cell {i}\n{'=' * 78}\n{text}")

with open(sys.argv[2], "w") as f:
    f.write("\n".join(out))
print(f"{sys.argv[2]}: {sum(len(o.splitlines()) for o in out):,} lines "
      f"from {len(out)} cells")
