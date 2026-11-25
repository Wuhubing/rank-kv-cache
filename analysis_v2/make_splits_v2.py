"""Build the S6 split files (fixed seeds, no document of splits/longctx_v1.json is reused).

  python -m analysis_v2.make_splits_v2 8b      --out splits/longctx_8b_v1.json
  python -m analysis_v2.make_splits_v2 64k     --out splits/longctx64k_v1.json
  python -m analysis_v2.make_splits_v2 ruler   --out splits/ruler16k_v1.json
  python -m analysis_v2.make_splits_v2 scbench --out splits/scbench_v1.json

Order inside every stratum is sha256("<split name>:" + rid), so the selection is reproducible and
does not depend on the order of the source files. Token counts use the split's model tokenizer and
the full chat prompt (the prompt actually fed to the model).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collect.long import USER, build_turns  # noqa: E402

LB = "zai-org/LongBench@5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
Q06 = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
Q8 = "Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218"
SCB = "microsoft/SCBench@283310bb8c5ba6909dd9a6b1be087d2937f76f6d"
RUL = "simonjegou/ruler@24adceac8a0e6532936e8d721cd9e9084d2e4686"
SELECTOR = {"page_size": 16, "top_k": 64}
DECODING = {"greedy": True, "max_new_tokens": 1024, "stop": "eos | max_new | newest 32-gram repeats"}
TEMPLATES = {
    "narrativeqa": "Story:\n\n{context}\n\nWrite a detailed summary of the whole story, covering every part "
                   "of the plot, in at least eight paragraphs.",
    "multi_news": "News articles:\n\n{context}\n\nWrite a detailed summary of all the news articles, covering "
                  "every event, in at least eight paragraphs.",
}


def h(name, rid):
    return hashlib.sha256(f"{name}:{rid}".encode()).hexdigest()


def tok_of(spec):
    from transformers import AutoTokenizer
    m, _, r = spec.partition("@")
    return AutoTokenizer.from_pretrained(m, revision=r)


def lb_rows(files):
    from huggingface_hub import hf_hub_download
    repo, _, rev = LB.partition("@")
    z = ZipFile(hf_hub_download(repo, "data.zip", repo_type="dataset", revision=rev))
    for f in files:
        member, = [n for n in z.namelist() if Path(n).name == f"{f}.jsonl"]
        for i, line in enumerate(z.open(member)):
            yield f, i, json.loads(line)


def used_ctx():
    return {p["ctx"] for p in json.loads((ROOT / "splits/longctx_v1.json").read_text())["prompts"]}


def candidates(files, task_of, tok, model, lo, hi, template_of=None, max_ctx=None):
    used, seen, out = used_ctx(), set(), []
    for f, i, row in lb_rows(files):
        digest = hashlib.sha256(row["context"].encode()).hexdigest()
        if digest in seen or digest[:16] in used:
            continue
        seen.add(digest)
        task = task_of(f)
        p = dict(rid=f"{f}/{row['_id']}", file=f, row=i, ctx=digest[:16], task=task)
        if template_of and template_of(f):
            p["user_template"] = template_of(f)
        n_raw = len(tok.encode(row["context"], add_special_tokens=False))
        if max_ctx and n_raw > max_ctx:
            p["max_context_tokens"] = max_ctx
        n = len(build_turns(tok, {"source": "longbench"}, p, row)[0])
        if lo <= n < hi:
            p["prompt_tokens"] = n
            out.append(p)
    return out


def split_8b(args):
    name = "longctx_8b_v1"
    tok = tok_of(Q8)
    task_of = lambda f: "qmsum" if f.startswith("qmsum") else "gov_report"
    pool = candidates(["gov_report", "gov_report_e", "qmsum"], task_of, tok, Q8, 12000, 32000)
    a = sorted([p for p in pool if p["prompt_tokens"] >= 16000], key=lambda p: h(name, p["rid"]))
    b = sorted([p for p in pool if p["prompt_tokens"] < 16000], key=lambda p: h(name, p["rid"]))
    need = 35
    floor_relaxed = len(a) < need
    take_b = b[:max(0, need - len(a))]
    take_a = a[:need]
    for p in take_a:
        p["band"] = "16-32K"
    for p in take_b:
        p["band"] = "12-16K"
    # dev/heldout stratified by band: 14/35 of each band to dev (rounded), rest held-out
    ndev_a = round(14 * len(take_a) / need)
    for i, p in enumerate(take_a):
        p["split"] = "dev" if i < ndev_a else "heldout"
    for i, p in enumerate(take_b):
        p["split"] = "dev" if i < 14 - ndev_a else "heldout"
    prompts = take_a + take_b
    out = dict(name=name, model=Q8, data=LB, source="longbench", selector=SELECTOR, decoding=DECODING,
               budgets={"primary_B": 96, "curve_B": [72, 80, 96, 128, 256]},
               rationale=[
                   "Second model (D3/D6 of docs/workshop/rerun_plan.md). Pool: every distinct GovReport context in LongBench "
                   "gov_report and gov_report_e, and every distinct QMSum context, that is NOT in splits/longctx_v1.json "
                   "(dedup by sha256 of the context across files). Prompt = collect/long.py USER template in the Qwen3 chat "
                   "template with thinking, counted with the Qwen3-8B tokenizer.",
                   f"16-32K candidates: {len(a)} (all from gov_report_e: the 16-32K GovReport/QMSum contexts of gov_report "
                   f"and qmsum are all used by longctx_v1). Fewer than 35, so the length floor is relaxed to 12K as the plan "
                   f"prescribes: all 16-32K candidates are taken, the rest ({len(take_b)}) from 12-16K (pool {len(b)}).",
                   "Order inside a band: sha256('longctx_8b_v1:'+rid). dev/held-out stratified by band "
                   f"({ndev_a} + {14 - ndev_a} dev).",
                   "Held-out documents are collected only after docs/preregistration_8b.md is committed."],
               floor_relaxed_to_12k=floor_relaxed, candidates={"16-32K": len(a), "12-16K": len(b)},
               prompts=sorted(prompts, key=lambda p: (p["split"], p["rid"])))
    return out


def split_64k(args):
    name = "longctx64k_v1"
    tok = tok_of(Q06)
    files = ["gov_report", "gov_report_e", "narrativeqa", "multi_news", "multi_news_e"]
    task_of = lambda f: f.removesuffix("_e")
    tmpl = lambda f: TEMPLATES.get(f.removesuffix("_e"))
    pool = candidates(files, task_of, tok, Q06, 32000, 64000, tmpl, max_ctx=62000)
    pool = sorted(pool, key=lambda p: (p.get("max_context_tokens", 0) > 0, h(name, p["rid"])))
    take = pool[:14]
    for p in take:
        p["split"] = "dev"
        p["band"] = "32-64K"
    return dict(name=name, model=Q06, data=LB, source="longbench", selector=SELECTOR, decoding=DECODING,
                rope_scaling={"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768},
                max_position_embeddings=65536 + 2048,
                budgets={"primary_B": 96, "curve_B": [72, 80, 96, 128, 256]},
                rationale=[
                    "Exploratory 64K setting for Qwen3-0.6B (dev only, no held-out; D6). Qwen3's native context is 32K, so "
                    "YaRN rope scaling (factor 2, original 32768, as on the Qwen3 model card) is applied: a different model "
                    "configuration, reported separately from the 32K results.",
                    "Pool: distinct LongBench contexts with a 32-64K token chat prompt, not in longctx_v1: GovReport "
                    "(gov_report, gov_report_e), NarrativeQA and Multi-News (summary prompts in the same 'detailed summary, "
                    "at least eight paragraphs' form). Contexts above 62K tokens are middle-truncated to 62K (LongBench's "
                    "own truncation rule) and used only if needed to reach 14 documents.",
                    "Order: untruncated first, then sha256('longctx64k_v1:'+rid)."],
                candidates=len(pool), prompts=sorted(take, key=lambda p: p["rid"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["8b", "64k", "ruler", "scbench"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from analysis_v2 import make_splits_more  # ruler / scbench builders live next to their loaders
    fn = {"8b": split_8b, "64k": split_64k, "ruler": make_splits_more.split_ruler,
          "scbench": make_splits_more.split_scbench}[a.which]
    out = fn(a)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    from collections import Counter
    print(a.out, len(out["prompts"]), Counter((p["split"], p.get("task"), p.get("band")) for p in out["prompts"]))


if __name__ == "__main__":
    main()
