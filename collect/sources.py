"""Non-LongBench sources for collect/long.py (S6, exploratory, dev only).

ruler    simonjegou/ruler (pre-generated RULER, 16384-token config, parquet). Prompt: the task's own
         context followed by its question, as one user turn in the Qwen3 chat template with
         thinking (the answer prefix is not forced; the model answers freely).
parquet  followup-v3 held-out (splits/longctx_v3.json): one row of a pinned parquet file per prompt; the prompt entry
         names the dataset (source_spec "repo@revision"), the file, the row and the text field. The prompt is built
         exactly as for LongBench (collect.long USER template of the task, Qwen3 chat template with thinking).
scbench  microsoft/SCBench multi-turn sessions (data/<file>.jsonl): one shared context and 5 queries.
         Turn 1 = context (middle-truncated to max_context_tokens) + query 1; turn i>1 = query i as a
         new user message appended to the same KV cache after the previous answer.
"""
import ast
import hashlib
import json


def _rev(spec):
    repo, _, rev = spec.partition("@")
    return repo, rev or None


def load_rows(source, spec, wanted):
    from huggingface_hub import hf_hub_download
    repo, rev = _rev(spec)
    rows = {}
    if source == "ruler":
        import pandas as pd
        files = {p["file"] for p in wanted}
        for f in files:
            d = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision=rev))
            for p in wanted:
                if p["file"] == f:
                    r = d.iloc[int(p["row"])]
                    rows[f, int(p["row"])] = {k: (r[k].tolist() if hasattr(r[k], "tolist") else r[k]) for k in d.columns}
        return rows
    if source == "parquet":
        import pandas as pd
        frames = {}
        for p in wanted:
            spec, f = p["source_spec"], p["file"]
            if (spec, f) not in frames:
                r, _, v = spec.partition("@")
                frames[spec, f] = pd.read_parquet(hf_hub_download(r, f, repo_type="dataset", revision=v or None))
            rows[f, int(p["row"])] = {"context": frames[spec, f][p["field"]].iloc[int(p["row"])]}
        return rows
    if source == "scbench":
        for f in {p["file"] for p in wanted}:
            path = hf_hub_download(repo, f"data/{f}.jsonl", repo_type="dataset", revision=rev)
            idx = {int(p["row"]) for p in wanted if p["file"] == f}
            with open(path) as fh:
                for i, line in enumerate(fh):
                    if i in idx:
                        rows[f, i] = json.loads(line)
        return rows
    raise ValueError(f"unknown source {source}")


def context_hash(source, row):
    return hashlib.sha256(row["context"].encode()).hexdigest()[:16]


def turns_of(row):
    out = []
    for t in row["multi_turns"]:
        if isinstance(t, str):
            t = ast.literal_eval(t)
        out.append(t)
    return out


def build_turns(tok, source, p, row):
    from collect.long import USER, middle_truncate
    if source == "parquet":
        msg = [{"role": "user", "content": USER[p["task"]].format(context=row["context"])}]
        return [tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=True)]
    if source == "ruler":
        msg = [{"role": "user", "content": row["context"].rstrip() + "\n\n" + row["question"].strip()}]
        return [tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=True)]
    if source == "scbench":
        context = middle_truncate(tok, row["context"], int(p["max_context_tokens"]))
        qs = [t["input"] for t in turns_of(row)][: int(p.get("turns", 5))]
        first = [{"role": "user", "content": context + "\n\n" + qs[0]}]
        out = [tok.apply_chat_template(first, tokenize=True, add_generation_prompt=True, enable_thinking=True)]
        for q in qs[1:]:
            # the previous answer ends with <|im_end|> (fed as the first token of this chunk)
            out.append(tok.encode(f"\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
                                  add_special_tokens=False))
        return out
    raise ValueError(source)
