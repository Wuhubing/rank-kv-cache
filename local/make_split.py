"""Build splits/longctx_v2.json: fresh held-out documents that no earlier split has used.

The longctx_v1 pool (every GovReport/QMSum context whose prompt is 16-32K tokens) is used up by
its 14 dev + 21 held-out documents. v2 takes LongBench contexts that are LONGER than that pool,
plus NarrativeQA, and middle-truncates each one (first and last halves kept, as LongBench does)
to a deterministic target prompt length in [16.5K, 31.5K) tokens.

  heldout      primary group: 18 GovReport + 6 QMSum (same tasks and prompts as dev)
  heldout_ood  secondary group: 8 NarrativeQA ("summarise the whole story"); a new task, reported apart

Excluded: every LongBench _id in splits/longbench_v1.json or splits/longctx_v1.json, and every
context hash of longctx_v1. Contexts are de-duplicated (QMSum and NarrativeQA repeat them across
questions). Order within a task is sha256("longctx-v2:" + rid). Run once, on the capture machine:

  python local/make_split.py --out splits/longctx_v2.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local"))
from capture import TEMPLATES, build_context, chat_prompt, load_split, longbench_rows  # noqa: E402

SEED = "longctx-v2:"
LO, SPAN, MIN_FULL = 16500, 15000, 16500
GROUPS = {"heldout": {"gov_report": 18, "qmsum": 6}, "heldout_ood": {"narrativeqa": 8}}


def h(s):
    return int(hashlib.sha256((SEED + s).encode()).hexdigest(), 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "splits/longctx_v2.json")
    args = ap.parse_args()
    v1, v1_sha = load_split(ROOT / "splits/longctx_v1.json")
    short = json.loads((ROOT / "splits/longbench_v1.json").read_text())
    used_ids = {e["rid"].split("/")[-1] for e in v1["prompts"]}
    used_ids |= {p["request_id"].split("/")[-1] for p in short["prompts"]}
    used_ctx = {e["ctx"] for e in v1["prompts"]}
    from transformers import AutoTokenizer
    mid, rev = v1["model"].split("@")
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    tasks = [t for g in GROUPS.values() for t in g]
    rows = longbench_rows(v1, tasks)
    prompts, pool = [], {}
    for group, quota in GROUPS.items():
        for task, n in quota.items():
            overhead = len(chat_prompt(tok, task, ""))
            seen, cands = set(), []
            for rid, (index, row) in rows.items():
                if not rid.startswith(task + "/") or rid.split("/")[1] in used_ids:
                    continue
                ctx = hashlib.sha256(row["context"].encode()).hexdigest()[:16]
                if ctx in used_ctx or ctx in seen:
                    continue
                seen.add(ctx)
                cands.append((h(rid), rid, index, row, ctx))
            cands.sort()
            picked = []
            for _, rid, index, row, ctx in cands:
                ids = tok.encode(row["context"], add_special_tokens=False)
                full = overhead + len(ids)
                if full < MIN_FULL:
                    continue
                target = LO + h(ctx) % SPAN
                trunc = None
                if full > target:
                    keep = target - overhead
                    trunc = dict(context_tokens=len(ids), head=keep // 2, tail=keep - keep // 2)
                prompt = chat_prompt(tok, task, build_context(tok, row["context"], trunc))
                picked.append(dict(rid=rid, row=index, ctx=ctx, prompt_tokens=len(prompt), task=task,
                                   band="24-32K" if len(prompt) >= 24000 else "16-24K", split=group,
                                   truncate=trunc))
                if len(picked) == n:
                    break
            pool[task] = len(cands)
            if len(picked) < n:
                print(f"WARNING {task}: only {len(picked)} eligible of {n} wanted")
            prompts += picked
            for p in picked:
                t = p["truncate"]
                print(f"{group:12s} {p['rid'][:40]:40s} {p['prompt_tokens']:6d} tokens  "
                      f"{'full' if not t else 'kept %d of %d context tokens' % (t['head'] + t['tail'], t['context_tokens'])}")
    out = dict(
        name="longctx_v2", model=v1["model"], data=v1["data"],
        prompt="local/capture.py TEMPLATES (gov_report and qmsum identical to longctx_v1), Qwen3 chat template, enable_thinking=True",
        selector=v1["selector"], decoding=v1["decoding"], budgets=v1["budgets"],
        excludes=dict(longctx_v1_sha256=v1_sha, longbench_v1_ids=True),
        rationale=[
            "longctx_v1 used every GovReport/QMSum context with a 16-32K prompt, so v2 draws fresh, longer "
            "contexts and middle-truncates them (first and last halves kept, as LongBench does) to a target "
            f"prompt length {LO} + sha256('{SEED}'+ctx) mod {SPAN} tokens.",
            "heldout keeps dev's tasks and prompts (primary, preregistered criteria); heldout_ood adds "
            "NarrativeQA whole-story summaries as a separately reported generalization check.",
            f"Distinct eligible-or-not contexts per task after exclusions: {pool}.",
        ],
        prompts=prompts,
    )
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}: {len(prompts)} documents; sha256 "
          f"{hashlib.sha256(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
