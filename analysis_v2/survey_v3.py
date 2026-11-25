"""F1 (followup-v3): document pool for a third, pre-registered held-out. Looks at prompt lengths only;
no trace is collected here.

  python -m analysis_v2.survey_v3 --out splits/longctx_v3.json --report results_v3/f1_pool/survey.json

Rules (docs/workshop/TASK_v3.md F1):
  * no document of longctx_v1, longctx_8b_v1, longctx64k_v1, ruler16k_v1, scbench_v1, longbench_v1;
  * sources: the original GovReport (ccdv/govreport-summarization, all splits) and QMSum
    (pszemraj/qmsum-cleaned "no-prefix", all splits; one entry per meeting), both pinned by revision;
  * de-duplication: identical context hash, or > 5 % of the candidate's 8-word shingles found in any
    used document (lower-cased \\w+ words), excludes a candidate; duplicates inside the pool are dropped;
  * length: the full chat prompt (collect.long gov_report / qmsum template, Qwen3 chat template with
    thinking) is 16 000 <= n < 32 000 tokens with BOTH the Qwen3-0.6B and the Qwen3-8B tokenizer;
  * order: sha256("longctx_v3:" + rid); the first N (21) are taken, the same documents for both models.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collect.long import build_turns, load_rows  # noqa: E402
from analysis_v2.make_splits_v2 import DECODING, LB, Q06, Q8, SELECTOR, tok_of  # noqa: E402

NAME = "longctx_v3"
GOV = "ccdv/govreport-summarization@4e21184e01ae8017e2c036e180fe5e541fef60a0"
QMS = "pszemraj/qmsum-cleaned@d926f9ccd037d45810f163b3ebf036ee3ad98897"
GOV_FILES = ["document/train-00000-of-00002.parquet", "document/train-00001-of-00002.parquet",
             "document/validation-00000-of-00001.parquet", "document/test-00000-of-00001.parquet"]
QMS_FILES = ["no-prefix/train-00000-of-00001.parquet", "no-prefix/validation-00000-of-00001.parquet",
             "no-prefix/test-00000-of-00001.parquet"]
USED = ["longctx_v1", "longctx_8b_v1", "longctx64k_v1", "ruler16k_v1", "scbench_v1", "longbench_v1"]
LO, HI = 16000, 32000
SHINGLE, MAX_OVERLAP = 8, 0.05
N_TAKE, N_MIN = 21, 14


def order_key(rid):
    return hashlib.sha256(f"{NAME}:{rid}".encode()).hexdigest()


def ctx_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def shingles(text, n=SHINGLE):
    w = re.findall(r"\w+", text.lower())
    return {hash(" ".join(w[i:i + n])) for i in range(max(0, len(w) - n + 1))}


def used_contexts():
    """{split name: [context text]} of every document of the six used splits."""
    out = {}
    for name in USED:
        d = json.loads((ROOT / f"splits/{name}.json").read_text())
        if name == "longbench_v1":
            spec = f"{d['dataset']}@{d['data_revision']}"
            wanted = [dict(file=p["task"], row=p["row_index"], task=p["task"]) for p in d["prompts"]]
            rows = load_rows(spec, wanted, "longbench")
        else:
            wanted = d["prompts"]
            rows = load_rows(d["data"], wanted, d.get("source", "longbench"))
        out[name] = [r["context"] for r in rows.values()]
        if len(out[name]) != len(d["prompts"]):
            raise SystemExit(f"{name}: loaded {len(out[name])} of {len(d['prompts'])} contexts")
    return out


def parquet(spec, f):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    repo, _, rev = spec.partition("@")
    return pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision=rev))


def pool_rows():
    """Yield (prompt entry, context text) for every distinct source document."""
    seen = set()
    for f in GOV_FILES:
        d = parquet(GOV, f)
        for i, text in enumerate(d["report"]):
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            sp = f.split("/")[1].split("-")[0]
            yield dict(rid=f"govreport/{sp}/{h[:16]}", source_spec=GOV, file=f, row=i, field="report",
                       task="gov_report", ctx=h[:16]), text
    for f in QMS_FILES:
        d = parquet(QMS, f)
        for i in range(len(d)):
            text = d["input"].iloc[i]
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            yield dict(rid=f"qmsum/{d['id'].iloc[i]}", source_spec=QMS, file=f, row=i, field="input",
                       task="qmsum", ctx=h[:16]), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    used = used_contexts()
    used_hash = {ctx_hash(t) for ts in used.values() for t in ts}
    U = set()
    for ts in used.values():
        for t in ts:
            U |= shingles(t)
    tok6, tok8 = tok_of(Q06), tok_of(Q8)
    split = {"source": "longbench"}  # same prompt construction as collect.long for gov_report / qmsum
    stats = Counter()
    cand, excluded = [], []
    for p, text in pool_rows():
        stats[f"distinct:{p['task']}"] += 1
        if not 30000 <= len(text) <= 250000:  # cheap prefilter; a 16-32K-token prompt is ~60-150K chars
            stats["char_prefilter_out"] += 1
            continue
        n6 = len(build_turns(tok6, split, p, {"context": text})[0])
        if not LO <= n6 < HI:
            stats["len_out_06b"] += 1
            continue
        n8 = len(build_turns(tok8, split, p, {"context": text})[0])
        if not LO <= n8 < HI:
            stats["len_out_8b"] += 1
            continue
        stats[f"in_range:{p['task']}"] += 1
        S = shingles(text)
        ov = len(S & U) / max(1, len(S))
        p.update(prompt_tokens=n6, prompt_tokens_8b=n8, shingle_overlap=round(ov, 5))
        if p["ctx"] in used_hash or ov > MAX_OVERLAP:
            stats[f"dedup_out:{p['task']}"] += 1
            excluded.append(dict(rid=p["rid"], overlap=ov, hash_hit=p["ctx"] in used_hash))
            continue
        cand.append(p)
    cand.sort(key=lambda p: order_key(p["rid"]))
    take = cand[:N_TAKE]
    for p in take:
        p["split"] = "heldout"
        p["band"] = "16-24K" if p["prompt_tokens"] < 24000 else "24-32K"
    n = len(take)
    verdict = ("F can be done with 21 documents" if n >= N_TAKE else
               f"F can be done with {n} documents (fewer than 21)" if n >= N_MIN else
               "fewer than 14 documents: F is not done")
    out = dict(name=NAME, model=Q06, model_8b=Q8, data={"gov_report": GOV, "qmsum": QMS}, source="parquet",
               selector=SELECTOR, decoding=DECODING, budgets={"primary_B": 96, "secondary_B": 128},
               rationale=[
                   "Third held-out (followup-v3 F1). Pool: every distinct document of the original GovReport "
                   "(ccdv/govreport-summarization, train/validation/test, field 'report') and of QMSum "
                   "(pszemraj/qmsum-cleaned no-prefix, train/validation/test, field 'input' = the meeting transcript, "
                   "one entry per distinct transcript), both pinned by revision. Prompt = collect.long USER template of "
                   "the task in the Qwen3 chat template with thinking.",
                   f"Excluded: any document of {', '.join(USED)} (same context hash, or more than 5 % of the "
                   "candidate's 8-word shingles occur in a used document).",
                   "Length: 16000 <= prompt tokens < 32000 with both the Qwen3-0.6B and the Qwen3-8B tokenizer; the same "
                   "documents are used for both models.",
                   "Order: sha256('longctx_v3:'+rid); the first 21 are taken. All are held-out; nothing here is used "
                   "for selection or tuning. Held-out traces are collected only after docs/preregistration_v3.md and "
                   "splits/heldout_frozen_v3.json are committed and pushed."],
               candidates=len(cand), prompts=sorted(take, key=lambda p: p["rid"]))
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    rep = dict(verdict=verdict, n_taken=n, candidates_after_dedup=len(cand), stats=dict(stats),
               used_docs={k: len(v) for k, v in used.items()}, used_shingles=len(U),
               excluded_by_dedup=sorted(excluded, key=lambda x: -x["overlap"]),
               overlap_of_candidates=dict(max=max((p["shingle_overlap"] for p in cand), default=None),
                                          taken_max=max((p["shingle_overlap"] for p in take), default=None)),
               taken=[dict(rid=p["rid"], task=p["task"], tok06=p["prompt_tokens"], tok8b=p["prompt_tokens_8b"],
                           overlap=p["shingle_overlap"]) for p in take],
               task_mix_of_candidates=Counter(p["task"] for p in cand))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(verdict, json.dumps(dict(stats)), "candidates", len(cand))


if __name__ == "__main__":
    main()
