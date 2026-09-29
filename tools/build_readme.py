"""Render README.md from README.tmpl.md, filling {{path.to.key:fmt}} from results.json.

S11 runs a single model/dataset for all five items, so there's just one results file —
unlike S10's two-model split, no namespace prefix is needed: {{item3.peak_ratio_no_warmup}}
reads straight from results.json. A placeholder that does not resolve is an error, not a
silently-empty span: no number in the write-up is typed by hand, every one is looked up
from a file the notebook actually wrote on its last run.
"""
import json, pathlib, re, sys

results = json.loads(pathlib.Path("results.json").read_text())

tmpl = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "README.tmpl.md").read_text()

PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_.\[\]]+)(?::([^}]+))?\}\}")
missing = []


def lookup(path):
    node = results
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def render(m):
    path, fmt = m.group(1), m.group(2)
    try:
        val = lookup(path)
    except (KeyError, IndexError, ValueError, TypeError):
        missing.append(f"{path}  (no such key in results.json)")
        return f"<<MISSING {path}>>"
    if not fmt:
        return str(val)
    try:
        return format(val, fmt)
    except (ValueError, TypeError) as e:
        missing.append(f"{path}  (bad format {fmt!r} for {type(val).__name__}: {e})")
        return f"<<BADFMT {path}>>"


out = PLACEHOLDER.sub(render, tmpl)
if missing:
    print("unresolved placeholders:", *sorted(set(missing)), sep="\n  ")
    sys.exit(1)

pathlib.Path("README.md").write_text(out)
print(f"README.md: {len(out.splitlines())} lines, "
      f"{len(PLACEHOLDER.findall(tmpl))} values filled from results.json")
