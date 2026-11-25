"""Long-document Quest trace collection (schema v2 npz) for replication-v2.

Ported from notebooks/long_trace_capture.py. The selector (QuestPages), the prompt templates, the
stop rule (EOS | max_new | newest 32-gram repeats) and the precision (bf16 weights, sdpa prefill,
float32 page bounds, float16 stored scores) are unchanged; tests/test_collect_long.py checks that
the selection is identical to the notebook's on random inputs.

  python -m collect.long --split splits/longctx_v1.json --subset dev --out data/dev
  python -m collect.long --split ... --subset dev --docs RID [RID ...] --out DIR

Each document gives DIR/<task>__<id>.npz (ranked int16 [T, L, K], scores float16 [T, L, P], rid,
ctx; optional attn_mass float16 [T, L, K] aligned with ranked, true_rho float32 [T, L]) and
DIR/<task>__<id>.json (manifest). Writes go to a temporary name first and are renamed; finished
documents are skipped, so the command can be re-run after an interruption. DIR/_status.json holds
progress. Every trace is read back with kvpolicy.load (the strict reader) right after it is written.

Optional fields (off by default; they never change the selected pages):
  --record-attn-mass       softmax attention mass of each selected page, summed over query heads,
                           under the sparse attention actually used (selected pages + recent tail)
  --record-true-attn-rank  per layer and step, Spearman rho between the Quest page scores and the
                           dense attention mass of each complete page (mean over query heads)

A split may carry "rope_scaling" (e.g. YaRN for the 64K setting); it is applied to the model config
and recorded in the manifest. A prompt entry may carry "user_template" to override USER[task].
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

USER = {
    "qmsum": "Meeting transcript:\n\n{context}\n\nWrite a detailed summary of the whole meeting, "
             "covering every topic that was discussed, in at least eight paragraphs.",
    "gov_report": "Government report:\n\n{context}\n\nWrite a detailed summary of the whole report, "
                  "covering every section, in at least eight paragraphs.",
}


def chat_prompt(tok, task, row, user=USER):
    msg = [{"role": "user", "content": user[task].format(context=row["context"])}]
    return tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=True)


def _avg_ranks(x):
    order = x.argsort(stable=True)
    xs = x[order]
    new = torch.ones(len(x), dtype=torch.bool, device=x.device)
    new[1:] = xs[1:] != xs[:-1]
    gid = new.cumsum(0) - 1
    n = int(gid[-1]) + 1
    pos = torch.arange(len(x), device=x.device, dtype=torch.float64)
    s = torch.zeros(n, dtype=torch.float64, device=x.device).index_add_(0, gid, pos)
    c = torch.zeros(n, dtype=torch.float64, device=x.device).index_add_(0, gid, torch.ones_like(pos))
    out = torch.empty(len(x), dtype=torch.float64, device=x.device)
    out[order] = (s / c)[gid]
    return out


def spearman(a, b):
    """Spearman rho of two 1-D tensors (average ranks for ties); nan if undefined."""
    if len(a) < 2:
        return float("nan")
    ra, rb = _avg_ranks(a.double()), _avg_ranks(b.double())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = (ra.square().sum() * rb.square().sum()).sqrt()
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


class QuestPages:
    """Same selector as notebooks/long_trace_capture.py; the record_* flags only add read-outs."""

    def __init__(s, page_size, top_k, record_attn_mass=False, record_true_attn_rank=False):
        s.page_size, s.top_k, s.bounds, s.step = page_size, top_k, {}, None
        s.record_attn_mass, s.record_true_attn_rank = record_attn_mass, record_true_attn_rank

    def __call__(s, module, query, key, value, attention_mask, scaling, **kw):
        layer = module.layer_idx
        kvh, length, dim = key.shape[1:]
        hp = (length - 1) // s.page_size
        past = s.bounds.get(layer)
        cached = 0 if past is None else past[0].shape[1]
        if hp > cached:
            b = key[0, :, cached*s.page_size:hp*s.page_size].reshape(kvh, hp-cached, s.page_size, dim)
            lo, hi = b.amin(2).float(), b.amax(2).float()
            if past is not None:
                lo, hi = torch.cat((past[0], lo), 1), torch.cat((past[1], hi), 1)
            s.bounds[layer] = lo, hi
        g = query.shape[1] // kvh
        lo, hi = s.bounds[layer]
        q = query[0, :, 0].reshape(kvh, g, dim).float()
        scores = (torch.einsum("hgd,hpd->hgp", q.clamp_min(0), hi)
                  + torch.einsum("hgd,hpd->hgp", q.clamp_max(0), lo)).mean((0, 1))
        ranked = scores.argsort(descending=True, stable=True)[:s.top_k]
        s.RANK[s.step, layer] = ranked.to(torch.int16)
        s.SCORE[s.step, layer, :hp] = scores.to(torch.float16)
        off = torch.arange(s.page_size, device=key.device)
        sel = ranked.sort().values
        pos = torch.cat(((sel[:, None]*s.page_size + off).flatten(),
                         torch.arange(hp*s.page_size, length, device=key.device)))
        k = key.index_select(2, pos).repeat_interleave(g, 1)
        v = value.index_select(2, pos).repeat_interleave(g, 1)
        out = F.scaled_dot_product_attention(query, k, v, scale=scaling)
        # Read-outs below use only copies in float32 and never touch the selection or `out`.
        if s.record_attn_mass:
            qf = query[0, :, 0].float()                                     # [H, d]
            w = (torch.einsum("hd,hnd->hn", qf, k[0].float()) * scaling).softmax(-1)
            n = len(sel)
            mass_sorted = w[:, :n*s.page_size].reshape(-1, n, s.page_size).sum((0, 2))
            inv = torch.searchsorted(sel, ranked)                          # ranked order
            s.ATTN[s.step, layer, :n] = mass_sorted[inv].to(torch.float16)
        if s.record_true_attn_rank and hp >= 2:
            qf = query[0, :, 0].float()
            kt = key[0, :, :length].float().repeat_interleave(g, 0)         # [H, length, d]
            full = torch.einsum("hd,hnd->hn", qf, kt) * scaling
            p = full.softmax(-1)[:, :hp*s.page_size].reshape(-1, hp, s.page_size).sum(2).mean(0)
            s.RHO[s.step, layer] = spearman(scores.float(), p)
        return out.transpose(1, 2).contiguous(), None


def looping(gen, n=32):
    if len(gen) < 2*n:
        return False
    last = tuple(gen[-n:])
    return any(tuple(gen[i:i+n]) == last for i in range(len(gen)-n))


@torch.inference_mode()
def capture(model, prompt, page_size=16, top_k=64, max_new=1024, eos_ids=(),
            record_attn_mass=False, record_true_attn_rank=False, device="cuda"):
    from transformers import AttentionInterface
    cfg = model.config
    L = cfg.num_hidden_layers
    be = QuestPages(page_size, top_k, record_attn_mass, record_true_attn_rank)
    pmax = (len(prompt) + max_new) // page_size
    be.RANK = torch.full((max_new, L, top_k), -1, dtype=torch.int16, device=device)
    be.SCORE = torch.full((max_new, L, pmax), float("nan"), dtype=torch.float16, device=device)
    if record_attn_mass:
        be.ATTN = torch.full((max_new, L, top_k), float("nan"), dtype=torch.float16, device=device)
    if record_true_attn_rank:
        be.RHO = torch.full((max_new, L), float("nan"), dtype=torch.float32, device=device)
    AttentionInterface.register("quest_npz", be)
    ids = torch.tensor([prompt], device=device)
    orig = cfg._attn_implementation
    try:
        cfg._attn_implementation = "sdpa"
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        tokn = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = [tokn.item()]
        cfg._attn_implementation = "quest_npz"
        stop = "length"
        for step in range(max_new - 1):
            if gen[-1] in eos_ids:
                stop = "eos"
                break
            if looping(gen):
                stop = "loop"
                break
            be.step = step
            out = model(input_ids=tokn, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            tokn = out.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(tokn.item())
        n = len(gen) - 1
        npages = (len(prompt) + n - 1) // page_size
        extra = {}
        if record_attn_mass:
            extra["attn_mass"] = be.ATTN[:n].cpu().numpy()
        if record_true_attn_rank:
            extra["true_rho"] = be.RHO[:n].cpu().numpy()
        return be.RANK[:n].cpu().numpy(), be.SCORE[:n, :, :npages].cpu().numpy(), gen, stop, extra
    finally:
        cfg._attn_implementation = orig
        be.bounds.clear()
        del be.RANK, be.SCORE


def make_sampler(temperature, seed, top_p=0.95, top_k=20, device="cuda"):
    """Greedy when temperature is 0; else Qwen3's recommended thinking-mode sampling (top-k 20,
    top-p 0.95) at the given temperature with a seeded generator."""
    if not temperature:
        return lambda logits: logits.argmax(-1, keepdim=True)
    g = torch.Generator(device=device).manual_seed(int(seed))

    def sample(logits):
        x = logits.float() / temperature
        v, i = x.topk(top_k, -1)
        p = v.softmax(-1)
        c = p.cumsum(-1)
        p = torch.where(c - p > top_p, torch.zeros_like(p), p)
        j = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=g)
        return i.gather(-1, j)
    return sample


@torch.inference_mode()
def capture_session(model, turns, page_size=16, top_k=64, max_new=1024, eos_ids=(), sampler=None,
                    record_attn_mass=False, record_true_attn_rank=False, device="cuda"):
    """Like capture(), for multi-turn sessions and/or sampled decoding. turns[0] is the full first
    prompt; turns[i>0] are the token ids appended after turn i-1's answer (its final token is fed
    first). The KV cache, the page bounds and the page ids persist across turns; prefill of every
    turn is dense (sdpa), decoding uses the Quest selector. Each turn stops at EOS, max_new or a
    repeated 32-gram. Returns ranked, scores, generated tokens per turn, stop per turn, extra
    (incl. turn_of_step int16 [T])."""
    from transformers import AttentionInterface
    sampler = sampler or (lambda logits: logits.argmax(-1, keepdim=True))
    cfg = model.config
    L = cfg.num_hidden_layers
    be = QuestPages(page_size, top_k, record_attn_mass, record_true_attn_rank)
    total = max_new * len(turns)
    ntok = sum(len(t) for t in turns)
    pmax = (ntok + total) // page_size + 1
    be.RANK = torch.full((total, L, top_k), -1, dtype=torch.int16, device=device)
    be.SCORE = torch.full((total, L, pmax), float("nan"), dtype=torch.float16, device=device)
    if record_attn_mass:
        be.ATTN = torch.full((total, L, top_k), float("nan"), dtype=torch.float16, device=device)
    if record_true_attn_rank:
        be.RHO = torch.full((total, L), float("nan"), dtype=torch.float32, device=device)
    AttentionInterface.register("quest_npz", be)
    orig = cfg._attn_implementation
    gens, stops, turn_of_step = [], [], []
    step, length, cache, last = 0, 0, None, None
    try:
        for ti, toks in enumerate(turns):
            feed = list(toks) if last is None else [last] + list(toks)
            cfg._attn_implementation = "sdpa"
            out = model(input_ids=torch.tensor([feed], device=device), past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            length += len(feed)
            cache = out.past_key_values
            tokn = sampler(out.logits[:, -1])
            gen = [tokn.item()]
            cfg._attn_implementation = "quest_npz"
            stop = "length"
            for _ in range(max_new - 1):
                if gen[-1] in eos_ids:
                    stop = "eos"
                    break
                if looping(gen):
                    stop = "loop"
                    break
                be.step = step
                out = model(input_ids=tokn, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                tokn = sampler(out.logits[:, -1])
                gen.append(tokn.item())
                turn_of_step.append(ti)
                step += 1
                length += 1
            gens.append(gen)
            stops.append(stop)
            last = gen[-1]
        n = step
        npages = (length - 1) // page_size
        extra = {"turn_of_step": np.array(turn_of_step, dtype=np.int16)}
        if record_attn_mass:
            extra["attn_mass"] = be.ATTN[:n].cpu().numpy()
        if record_true_attn_rank:
            extra["true_rho"] = be.RHO[:n].cpu().numpy()
        return be.RANK[:n].cpu().numpy(), be.SCORE[:n, :, :npages].cpu().numpy(), gens, stops, extra
    finally:
        cfg._attn_implementation = orig
        be.bounds.clear()
        del be.RANK, be.SCORE


@torch.inference_mode()
def dense_reference(model, prompt, max_new, eos_ids, device="cuda"):
    """Dense greedy generation with the same stop rule (for the manifest)."""
    cfg = model.config
    orig = cfg._attn_implementation
    try:
        cfg._attn_implementation = "sdpa"
        out = model(input_ids=torch.tensor([prompt], device=device), use_cache=True, logits_to_keep=1)
        cache, tokn = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        gen = [tokn.item()]
        stop = "length"
        for _ in range(max_new - 1):
            if gen[-1] in eos_ids:
                stop = "eos"
                break
            if looping(gen):
                stop = "loop"
                break
            out = model(input_ids=tokn, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache, tokn = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(tokn.item())
        return gen, stop
    finally:
        cfg._attn_implementation = orig


def trace_name(rid):
    return rid.replace("/", "__") + ".npz"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_info():
    info = {"name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        q = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=index,uuid,name,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, check=True)
        index, uuid, name, driver = [x.strip() for x in q.stdout.strip().split(",")]
        info.update(index=index, uuid=uuid, driver=driver)
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi_error"] = str(e)
    return info


def split_id(spec):
    name, _, rev = spec.partition("@")
    return name, rev or None


def row_file(p):
    return p.get("file", p["task"])


def load_rows(data_spec, wanted, source="longbench"):
    """Source rows by (file, row index): LongBench from the pinned data.zip (file = jsonl name,
    default the task); other sources via collect.sources."""
    if source != "longbench":
        from collect import sources
        return sources.load_rows(source, data_spec, wanted)
    from huggingface_hub import hf_hub_download
    repo, rev = split_id(data_spec)
    archive = hf_hub_download(repo, "data.zip", repo_type="dataset", revision=rev)
    need = {}
    for p in wanted:
        need.setdefault(row_file(p), set()).add(int(p["row"]))
    rows = {}
    with ZipFile(archive) as z:
        for f, idxs in need.items():
            member, = [n for n in z.namelist() if Path(n).name == f"{f}.jsonl"]
            with z.open(member) as data:
                for i, line in enumerate(data):
                    if i in idxs:
                        rows[f, i] = json.loads(line)
    return rows


def middle_truncate(tok, text, max_tokens):
    """LongBench's rule: keep the first and last max_tokens/2 tokens of the text."""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    half = max_tokens // 2
    return tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True)


def context_hash(source, row):
    if source != "longbench":
        from collect import sources
        return sources.context_hash(source, row)
    return hashlib.sha256(row["context"].encode()).hexdigest()[:16]


def build_turns(tok, split, p, row):
    """Token ids of each turn: [first prompt] for single-turn sources, plus continuation turns for
    multi-turn sessions (collect.sources)."""
    source = split.get("source", "longbench")
    if source != "longbench":
        from collect import sources
        return sources.build_turns(tok, source, p, row)
    template = p.get("user_template") or USER[p["task"]]
    context = row["context"]
    if p.get("max_context_tokens"):
        context = middle_truncate(tok, context, int(p["max_context_tokens"]))
    msg = [{"role": "user", "content": template.format(context=context)}]
    return [tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=True)]


def atomic_write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def load_model(split):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    model_id, model_rev = split_id(split["model"])
    tok = AutoTokenizer.from_pretrained(model_id, revision=model_rev)
    cfg = AutoConfig.from_pretrained(model_id, revision=model_rev)
    if split.get("rope_scaling"):
        cfg.rope_scaling = dict(split["rope_scaling"])
        if split.get("max_position_embeddings"):
            cfg.max_position_embeddings = int(split["max_position_embeddings"])
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=model_rev, config=cfg,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    return tok, model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--subset", choices=["dev", "heldout"], required=True)
    ap.add_argument("--docs", nargs="*", help="only these rids (default: the whole subset)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dense-reference", action="store_true")
    ap.add_argument("--record-attn-mass", action="store_true")
    ap.add_argument("--record-true-attn-rank", action="store_true")
    ap.add_argument("--max-new", type=int, help="override the split's max_new_tokens (per turn)")
    ap.add_argument("--page-size", type=int, help="override the split's page size (exploratory sweeps)")
    ap.add_argument("--top-k", type=int, help="override the split's K (exploratory sweeps)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (with --temperature)")
    args = ap.parse_args(argv)

    import transformers
    import kvpolicy as kp

    split_text = args.split.read_bytes()
    split = json.loads(split_text)
    prompts = [p for p in split["prompts"] if p["split"] == args.subset]
    if args.docs:
        unknown = set(args.docs) - {p["rid"] for p in prompts}
        if unknown:
            sys.exit(f"not in {args.subset} of {args.split}: {sorted(unknown)}")
        prompts = [p for p in prompts if p["rid"] in set(args.docs)]
    ps = args.page_size or split["selector"]["page_size"]
    top_k = args.top_k or split["selector"]["top_k"]
    max_new = args.max_new or split["decoding"]["max_new_tokens"]
    source = split.get("source", "longbench")
    args.out.mkdir(parents=True, exist_ok=True)
    status_path = args.out / "_status.json"
    status = {"split": str(args.split), "subset": args.subset, "total": len(prompts), "done": [],
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}

    todo = []
    for p in prompts:
        npz = args.out / trace_name(p["rid"])
        man = npz.with_suffix(".json")
        if man.exists() and npz.exists():
            if json.loads(man.read_text()).get("trace_sha256") == sha256_file(npz):
                status["done"].append(p["rid"])
                continue
        todo.append(p)
    atomic_write_json(status_path, status)
    if not todo:
        print("nothing to do")
        return

    tok, model = load_model(split)
    eos = model.generation_config.eos_token_id
    eos_ids = tuple(eos if isinstance(eos, (list, tuple)) else [eos])
    rows = load_rows(split["data"], todo, source)
    env = dict(gpu=gpu_info(), torch=torch.__version__, cuda=torch.version.cuda,
               transformers=transformers.__version__, numpy=np.__version__, python=platform.python_version(),
               host=platform.node(), model=split["model"], data=split["data"], source=source,
               rope_scaling=split.get("rope_scaling"),
               split_file=str(args.split), split_sha256=hashlib.sha256(split_text).hexdigest(),
               flags=dict(record_attn_mass=args.record_attn_mass,
                          record_true_attn_rank=args.record_true_attn_rank,
                          dense_reference=args.dense_reference, temperature=args.temperature,
                          seed=args.seed if args.temperature else None,
                          sampling=dict(top_k=20, top_p=0.95) if args.temperature else None,
                          page_size_override=args.page_size, top_k_override=args.top_k))
    for p in todo:
        row = rows[row_file(p), int(p["row"])]
        ctx = context_hash(source, row)
        if ctx != p["ctx"]:
            sys.exit(f"{p['rid']}: context hash {ctx} != split {p['ctx']}")
        turns = build_turns(tok, split, p, row)
        prompt = turns[0]
        if "prompt_tokens" in p and len(prompt) != p["prompt_tokens"]:
            sys.exit(f"{p['rid']}: {len(prompt)} prompt tokens != split {p['prompt_tokens']}")
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        if len(turns) == 1 and not args.temperature:   # the original path, unchanged
            R, S, gen, stop, extra = capture(model, prompt, ps, top_k, max_new, eos_ids,
                                             args.record_attn_mass, args.record_true_attn_rank)
            turn_gens, turn_stops = None, None
        else:
            sampler = make_sampler(args.temperature, args.seed)
            R, S, turn_gens, turn_stops, extra = capture_session(
                model, turns, ps, top_k, max_new, eos_ids, sampler,
                args.record_attn_mass, args.record_true_attn_rank)
            gen = [t for g in turn_gens for t in g]
            stop = turn_stops[-1]
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated()
        npz = args.out / trace_name(p["rid"])
        tmp = npz.with_name(npz.stem + ".tmp.npz")
        np.savez(tmp, ranked=R, scores=S, rid=np.array(p["rid"]), ctx=np.array(p["ctx"]),
                 prompt_tokens=np.array(len(prompt)), **extra)
        os.replace(tmp, npz)
        kp.load(npz)  # strict reader: schema checked right after collection
        man = dict(rid=p["rid"], ctx=p["ctx"], task=p["task"], row=int(p["row"]), subset=args.subset,
                   prompt_tokens=len(prompt),
                   prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                   generated=gen, steps=int(R.shape[0]), stop=stop, layers=int(R.shape[1]),
                   page_size=ps, top_k=top_k, max_new=max_new, eos_ids=list(eos_ids),
                   fields=sorted(["ranked", "scores", "rid", "ctx", "prompt_tokens", *extra]),
                   started_at=started, finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                   capture_seconds=round(wall, 2), peak_mem_bytes=int(peak),
                   trace_sha256=sha256_file(npz), trace_bytes=npz.stat().st_size, **env)
        if turn_gens is not None:
            man.update(turns=len(turns), turn_tokens=[len(t) for t in turns], turn_generated=turn_gens,
                       turn_stops=turn_stops)
        if p.get("max_context_tokens"):
            man["max_context_tokens"] = int(p["max_context_tokens"])
        if args.dense_reference:
            t1 = time.time()
            dgen, dstop = dense_reference(model, prompt, max_new, eos_ids)
            man.update(dense_generated=dgen, dense_stop=dstop, dense_seconds=round(time.time() - t1, 2))
        atomic_write_json(npz.with_suffix(".json"), man)
        status["done"].append(p["rid"])
        status["last"] = dict(rid=p["rid"], steps=man["steps"], stop=stop, seconds=man["capture_seconds"])
        atomic_write_json(status_path, status)
        print(f"{p['rid']}: {man['steps']} steps, stop={stop}, {wall:.0f}s", flush=True)
    status["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_write_json(status_path, status)


if __name__ == "__main__":
    main()
