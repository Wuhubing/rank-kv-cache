"""RULER and SCBench split builders (used by analysis_v2.make_splits_v2)."""
import hashlib

from analysis_v2.make_splits_v2 import DECODING, Q06, RUL, SCB, SELECTOR, h, tok_of

RULER_TASKS = {"cwe": 4, "niah_multivalue": 4, "niah_multiquery": 3, "niah_multikey_3": 3}
SCB_FILES = {"scbench_summary": 7, "scbench_qa_eng": 7}


def split_ruler(args):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from collect.sources import build_turns
    name = "ruler16k_v1"
    repo, _, rev = RUL.partition("@")
    f = "16384/test-00000-of-00001.parquet"
    d = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision=rev))
    tok = tok_of(Q06)
    prompts = []
    for task, n in RULER_TASKS.items():
        rows = [i for i in range(len(d)) if d.task.iloc[i] == task]
        rows.sort(key=lambda i: h(name, f"ruler/{task}/{i}"))
        for i in rows[:n]:
            r = {k: d[k].iloc[i] for k in ("context", "question")}
            p = dict(rid=f"ruler/{task}/{i}", file=f, row=i, task=task,
                     ctx=hashlib.sha256(r["context"].encode()).hexdigest()[:16], split="dev", band="16K")
            p["prompt_tokens"] = len(build_turns(tok, "ruler", p, r)[0])
            prompts.append(p)
    return dict(name=name, model=Q06, data=RUL, source="ruler", selector=SELECTOR, decoding=DECODING,
                budgets={"primary_B": 96, "curve_B": [80, 96, 128, 256]},
                rationale=[
                    "Exploratory RULER setting (dev only). Pre-generated RULER at the 16384-token configuration "
                    "(simonjegou/ruler). RULER answers are short, so only the tasks with the longest reference outputs "
                    "are used (cwe: 10 words, max 120 new tokens in RULER; niah_multivalue / niah_multiquery: 4 numbers; "
                    "niah_multikey_3: a uuid; all 120-128 in RULER). Decoding is the standard one here (thinking on, up "
                    "to 1024 tokens), so the thinking trace provides the decode steps.",
                    "Documents with fewer than 100 decode steps are listed separately and excluded from the main "
                    "statistics (docs/workshop/rerun_plan.md S6).",
                    "Rows per task ordered by sha256('ruler16k_v1:ruler/<task>/<row>')."],
                prompts=prompts)


def split_scbench(args):
    from huggingface_hub import hf_hub_download
    import json
    name = "scbench_v1"
    repo, _, rev = SCB.partition("@")
    prompts = []
    for f, n in SCB_FILES.items():
        rows = [json.loads(l) for l in open(hf_hub_download(repo, f"data/{f}.jsonl", repo_type="dataset", revision=rev))]
        order = sorted(range(len(rows)), key=lambda i: h(name, f"{f}/{rows[i]['id']}"))
        for i in order[:n]:
            prompts.append(dict(rid=f"{f}/{rows[i]['id']}", file=f, row=i, task=f, split="dev", band="24K+turns",
                                ctx=hashlib.sha256(rows[i]["context"].encode()).hexdigest()[:16],
                                max_context_tokens=24000, turns=5))
    dec = dict(DECODING, max_new_tokens=512, per_turn=True)
    return dict(name=name, model=Q06, data=SCB, source="scbench", selector=SELECTOR, decoding=dec,
                budgets={"primary_B": 96, "curve_B": [80, 96, 128, 256]},
                rationale=[
                    "Exploratory multi-turn setting (dev only). SCBench sessions share one context across 5 queries; the "
                    "contexts are 300K+ characters, so the context is middle-truncated to 24K Qwen3 tokens to fit the 32K "
                    "window with 5 turns of at most 512 new tokens each. Some queries may refer to truncated text; the "
                    "setting measures cache behaviour, not answer quality.",
                    "The KV cache, page bounds and page ids persist across turns (collect.long.capture_session); the "
                    "trace concatenates the decode steps of all turns (turn_of_step), and the replay keeps the cache "
                    "between turns. Prefill of each new turn is dense.",
                    "Tasks: scbench_summary (En.Sum) and scbench_qa_eng (En.QA), 7 sessions each, ordered by "
                    "sha256('scbench_v1:<file>/<id>')."],
                prompts=prompts)
