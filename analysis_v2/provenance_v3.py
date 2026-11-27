"""followup-v3 A9: gather the provenance of every trace set on wulab2 into a small, committable folder.

  python -m analysis_v2.provenance_v3 --data data --logs logs --out out/v3/provenance

For every trace folder under --data (recursively, any folder holding *.npz + manifests):
  manifests/<folder>.jsonl   one line per trace manifest, with the generated token list replaced by its length
                             and sha256 (the full list is in the trace folder on wulab2)
  status/<folder>.json       the collector's _status.json, if present
  logs/                      every collection / replay log under --logs (copied as is)
  identity.txt               bit-identity checks of traces collected with extra read-outs against the plain
                             traces of the same documents (ranked, scores, generated tokens; re-run in followup-v3
                             because the original check output was not kept)
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

# (folder with read-outs, plain folder): the read-outs must not change the selection
PAIRS = [("dev_attn", "dev"), ("dev_digest", "dev"), ("8b_dev_digest", "8b_dev")]


def strip(man):
    g = man.pop("generated", None)
    if g is not None:
        man["generated_len"] = len(g)
        man["generated_sha256"] = hashlib.sha256(json.dumps(g).encode()).hexdigest()
    return man


def trace_dirs(root):
    return sorted({p.parent for p in root.rglob("*.npz")})


def identity(data, a, b):
    lines, bad, n = [], 0, 0
    for pa in sorted((data / a).glob("*.npz")):
        pb = data / b / pa.name
        if not pb.exists():
            lines.append(f"  {pa.name}: no plain trace")
            bad += 1
            continue
        n += 1
        with np.load(pa) as za, np.load(pb) as zb:
            same_r = np.array_equal(za["ranked"], zb["ranked"])
            same_s = np.array_equal(za["scores"], zb["scores"], equal_nan=True)
        ma = json.loads(pa.with_suffix(".json").read_text())
        mb = json.loads(pb.with_suffix(".json").read_text())
        same_g = ma.get("generated") == mb.get("generated")
        if not (same_r and same_s and same_g):
            bad += 1
            lines.append(f"  {pa.name}: ranked {same_r} scores {same_s} generated {same_g}")
    return [f"{a} vs {b}: {n} documents compared, {bad} differ"] + lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (a.out / "manifests").mkdir(parents=True, exist_ok=True)
    (a.out / "status").mkdir(parents=True, exist_ok=True)
    index = []
    for d in trace_dirs(a.data):
        name = str(d.relative_to(a.data)).replace("/", "__")
        mans = []
        for js in sorted(d.glob("*.json")):
            if js.name.startswith("_"):
                continue
            mans.append(strip(json.loads(js.read_text())))
        with open(a.out / "manifests" / f"{name}.jsonl", "w") as f:
            for m in mans:
                f.write(json.dumps(m, sort_keys=True) + "\n")
        if (d / "_status.json").exists():
            shutil.copy(d / "_status.json", a.out / "status" / f"{name}.json")
        stops = {}
        for m in mans:
            stops[m.get("stop")] = stops.get(m.get("stop"), 0) + 1
        index.append(dict(folder=str(d), traces=len(list(d.glob("*.npz"))), manifests=len(mans), stops=stops))
    (a.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    if a.logs.exists():
        shutil.copytree(a.logs, a.out / "logs", dirs_exist_ok=True, ignore=shutil.ignore_patterns("v3"))
    lines = []
    for x, y in PAIRS:
        if (a.data / x).exists() and (a.data / y).exists():
            lines += identity(a.data, x, y)
    (a.out / "identity.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
