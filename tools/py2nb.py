"""Convert a `# %%` cell script into an .ipynb.

`# %% [markdown]` starts a markdown cell (its body is a triple-quoted string or
`#`-prefixed lines); `# %%` starts a code cell. Kept deliberately small — the
notebook is the deliverable, this is just the pipe that builds it.
"""
import sys, pathlib, nbformat


def split_cells(src: str):
    cells, kind, buf = [], "code", []
    for line in src.splitlines():
        if line.startswith("# %%"):
            if buf:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if buf:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, b) for k, b in cells if b.strip()]


def demarkdown(body: str) -> str:
    body = body.strip()
    if body.startswith('"""'):
        return body.strip('"').strip("\n")
    return "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())


def main(src_path, out_path):
    nb = nbformat.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    })
    for kind, body in split_cells(pathlib.Path(src_path).read_text()):
        nb.cells.append(
            nbformat.v4.new_markdown_cell(demarkdown(body)) if kind == "markdown"
            else nbformat.v4.new_code_cell(body)
        )
    nbformat.write(nb, out_path)
    print(f"{out_path}: {len(nb.cells)} cells")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
