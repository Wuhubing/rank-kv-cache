"""Cut each selected request's trace where its first answer ends.

Decode step s attends with query token generated[s] and produces generated[s + 1]. If
the first generated token containing --stop (default a newline) is generated[k], the
answer is generated[0..k], produced by steps 0..k-1, so steps k and later are dropped.
This removes MultiFieldQA's repeated "Answer: ..." tail without re-running the model.
Requests of other tasks are copied unchanged. Needs transformers for the tokenizer.
"""
import argparse
import gzip
import json
from pathlib import Path

from replay import open_trace


def cut_step(generated, decode, stop):
    """Number of decode steps to keep, or None when no token contains stop."""
    for k, token in enumerate(generated):
        if stop in decode([token]):
            return k
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path,
                        help="directory with trace.jsonl (or a v2 trace.jsonl.gz) and manifests/")
    parser.add_argument("--tasks", nargs="+", default=["multifieldqa_en"])
    parser.add_argument("--stop", default="\n")
    parser.add_argument("--out", type=Path, help="default: <run_dir>/trace_truncated.jsonl")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    manifests = [json.loads(p.read_text()) for p in sorted((args.run_dir / "manifests").glob("*.json"))]
    tokenizer = AutoTokenizer.from_pretrained(manifests[0]["model"],
                                              revision=manifests[0]["model_revision"])
    keep = {}
    for m in manifests:
        if m["task"] not in args.tasks:
            continue
        steps = cut_step(m["generated_token_ids"], tokenizer.decode, args.stop)
        if steps is None:
            steps = m["decode_steps"]
        keep[m["request_id"]] = steps
        answer = tokenizer.decode(m["generated_token_ids"][:steps + 1]).strip()
        print(f"{m['request_id']}: keep {steps} of {m['decode_steps']} steps | {answer[:90]!r}")
    source_path = args.run_dir / "trace.jsonl"
    if not source_path.exists():
        source_path = args.run_dir / "trace.jsonl.gz"
    gz = source_path.suffix == ".gz"
    out = args.out or args.run_dir / ("trace_truncated.jsonl" + (".gz" if gz else ""))
    kept = dropped = 0
    with open_trace(source_path) as source, \
            (gzip.open(out, "wt", encoding="utf-8") if out.suffix == ".gz"
             else out.open("w", encoding="utf-8")) as target:
        for line in source:
            row = json.loads(line)
            limit = keep.get(row["request_id"])
            if limit is not None and row["decode_step"] >= limit:
                dropped += 1
                continue
            target.write(line)
            kept += 1
    (out.parent / (out.name.split(".")[0] + ".json")).write_text(json.dumps(dict(stop=args.stop, tasks=args.tasks,
                                                          kept_steps=keep), indent=2) + "\n")
    print(f"Wrote {out}: kept {kept} rows, dropped {dropped}")


if __name__ == "__main__":
    main()
