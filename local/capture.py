"""Device-agnostic Quest page-trace capture (Apple MPS, CUDA or CPU) for the long-context workloads.

Same selector, prompt and stop rule as notebooks/long_trace_capture.py (Qwen3-0.6B, 16-token pages,
K=64, post-RoPE min/max bounds averaged over query heads, chat template with thinking, stop at EOS,
1024 tokens or a repeated 32-gram). Two changes only:
  * the device is chosen at run time (mps > cuda > cpu), and the rank buffer is int32 on device;
  * the prompt is prefilled in chunks (default 1024 tokens), so a 32K prompt fits in 32 GB.
Traces from different hardware are NOT interchangeable (bf16 near-ties flip rankings; see
docs/STATUS_REPORT.md 4.7): compare policies only within one capture device.

Output per document, readable by kvpolicy.load:
  OUT/<task>__<id>.npz   ranked int16 [T, L, K], scores float16 [T, L, P], rid
  OUT/<task>__<id>.json  manifest (prompt, generation, stop reason, timing, versions, array hashes)

  python local/capture.py check                                   # self-test, ~2 min
  python local/capture.py run --rids qmsum/cbbd3e6a... --out traces/pilot
  python local/capture.py run --subset dev --out traces/mps_dev   # resumable
"""
import argparse
import faulthandler
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # hub downloads fork after the tokenizer is used
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from long_trace_capture import USER, looping  # noqa: E402  (prompt templates and stop rule)

MODEL_ID, DATA_ID = "Qwen/Qwen3-0.6B", "zai-org/LongBench"
# longctx_v1 templates, plus one new task used only as an out-of-distribution held-out group.
TEMPLATES = {**USER, "narrativeqa": "Story:\n\n{context}\n\nWrite a detailed summary of the whole story, "
                                   "covering every part of the plot, in at least eight paragraphs."}
PAGE, TOPK, MAX_NEW = 16, 64, 1024


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class Quest:
    """Quest-style page selector used as a Transformers attention backend during decoding.

    Math is identical to notebooks/long_trace_capture.QuestPages. With check_every>0 it also
    verifies, on CPU, (a) the incrementally cached page bounds against a full recomputation and
    (b) that the device ranking is a valid stable descending sort of the device scores."""

    def __init__(self, page_size, top_k, check_every=0):
        self.page_size, self.top_k, self.bounds, self.step = page_size, top_k, {}, None
        self.check_every = check_every
        self.stats = dict(checked=0, max_rel_err=0.0, sort_errors=0, set_mismatch=0)

    def __call__(self, module, query, key, value, attention_mask, scaling, **kw):
        layer, ps = module.layer_idx, self.page_size
        kvh, length, dim = key.shape[1:]
        hp = (length - 1) // ps
        past = self.bounds.get(layer)
        cached = 0 if past is None else past[0].shape[1]
        if hp > cached:
            b = key[0, :, cached * ps:hp * ps].reshape(kvh, hp - cached, ps, dim)
            lo, hi = b.amin(2).float(), b.amax(2).float()
            if past is not None:
                lo, hi = torch.cat((past[0], lo), 1), torch.cat((past[1], hi), 1)
            self.bounds[layer] = lo, hi
        g = query.shape[1] // kvh
        lo, hi = self.bounds[layer]
        q = query[0, :, 0].reshape(kvh, g, dim).float()
        scores = (torch.einsum("hgd,hpd->hgp", q.clamp_min(0), hi)
                  + torch.einsum("hgd,hpd->hgp", q.clamp_max(0), lo)).mean((0, 1))
        ranked = scores.argsort(descending=True, stable=True)[:self.top_k]
        self.RANK[self.step, layer] = ranked.to(torch.int32)
        self.SCORE[self.step, layer, :hp] = scores.to(torch.float16)
        if self.check_every and self.step % self.check_every == 0:
            self._verify(query, key, scores, ranked, kvh, g, dim, hp)
        off = torch.arange(ps, device=key.device)
        pos = torch.cat(((ranked.sort().values[:, None] * ps + off).flatten(),
                         torch.arange(hp * ps, length, device=key.device)))
        k = key.index_select(2, pos).repeat_interleave(g, 1)
        v = value.index_select(2, pos).repeat_interleave(g, 1)
        out = F.scaled_dot_product_attention(query, k, v, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    def _verify(self, query, key, scores, ranked, kvh, g, dim, hp):
        ps, st = self.page_size, self.stats
        b = key[0, :, :hp * ps].reshape(kvh, hp, ps, dim).float().cpu()
        q = query[0, :, 0].reshape(kvh, g, dim).float().cpu()
        ref = (torch.einsum("hgd,hpd->hgp", q.clamp_min(0), b.amax(2))
               + torch.einsum("hgd,hpd->hgp", q.clamp_max(0), b.amin(2))).mean((0, 1)).numpy()
        dev = scores.cpu().numpy()
        r = ranked.cpu().numpy()
        scale = max(1e-6, float(np.abs(ref).max()))
        st["max_rel_err"] = max(st["max_rel_err"], float(np.abs(dev - ref).max()) / scale)
        # (b) stable descending sort of the device's own scores: smaller page ID first on ties
        want = np.lexsort((np.arange(hp), -dev))[:self.top_k]
        st["sort_errors"] += int((want != r).any())
        # informational: selected set from an fp32 CPU recomputation (near-ties may differ)
        st["set_mismatch"] += int(set(np.lexsort((np.arange(hp), -ref))[:self.top_k]) != set(r))
        st["checked"] += 1


@torch.inference_mode()
def capture(model, prompt, device, *, max_new=MAX_NEW, chunk=1024, eos_ids=(), check_every=0, progress=False):
    from transformers import AttentionInterface
    cfg = model.config
    L = cfg.num_hidden_layers
    be = Quest(PAGE, TOPK, check_every)
    pmax = (len(prompt) + max_new) // PAGE
    be.RANK = torch.full((max_new, L, TOPK), -1, dtype=torch.int32, device=device)
    be.SCORE = torch.full((max_new, L, pmax), float("nan"), dtype=torch.float16, device=device)
    AttentionInterface.register("quest_local", be)
    ids = torch.tensor([prompt], device=device)
    orig = cfg._attn_implementation
    try:
        cfg._attn_implementation = "sdpa"
        t0 = time.perf_counter()
        cache = None
        for s in range(0, len(prompt), chunk):
            out = model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            if progress:
                sync(device)
                print(f"    prefill {min(s + chunk, len(prompt))}/{len(prompt)}", end="\r", flush=True)
        tokn = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = [tokn.item()]
        t1 = time.perf_counter()
        cfg._attn_implementation = "quest_local"
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
            if progress and step % 200 == 199:
                print(f"    step {step + 1}  {time.perf_counter() - t1:.0f}s          ", end="\r", flush=True)
        sync(device)
        t2 = time.perf_counter()
        n = len(gen) - 1
        npages = (len(prompt) + n - 1) // PAGE
        ranked = be.RANK[:n].cpu().numpy().astype(np.int16)
        scores = be.SCORE[:n, :, :npages].cpu().numpy()
        timing = dict(prefill_s=round(t1 - t0, 3), decode_s=round(t2 - t1, 3),
                      ms_per_step=round(1000 * (t2 - t1) / max(n, 1), 2))
        return ranked, scores, gen, stop, timing, dict(be.stats)
    finally:
        cfg._attn_implementation = orig
        be.bounds.clear()
        del be.RANK, be.SCORE, cache


def chat_prompt(tok, task, context):
    msg = [{"role": "user", "content": TEMPLATES[task].format(context=context)}]
    ids = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


def build_context(tok, context, trunc):
    """LongBench-style middle truncation: keep the first `head` and last `tail` context tokens."""
    if not trunc:
        return context
    ids = tok.encode(context, add_special_tokens=False)
    if len(ids) != trunc["context_tokens"]:
        raise ValueError(f"context has {len(ids)} tokens, split expects {trunc['context_tokens']}")
    return tok.decode(ids[:trunc["head"]]) + tok.decode(ids[len(ids) - trunc["tail"]:])


def load_split(path):
    split = json.loads(Path(path).read_text())
    return split, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_and_tokenizer(split, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid, rev = split["model"].split("@")
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, torch_dtype=dtype,
                                                 attn_implementation="sdpa").to(device).eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    return model, tok, eos


def longbench_rows(split, tasks):
    from huggingface_hub import hf_hub_download
    did, rev = split["data"].split("@")
    archive = hf_hub_download(did, "data.zip", repo_type="dataset", revision=rev)
    rows = {}
    with ZipFile(archive) as z:
        for task in tasks:
            member, = [n for n in z.namelist() if Path(n).name == f"{task}.jsonl"]
            with z.open(member) as f:
                for i, line in enumerate(f):
                    row = json.loads(line)
                    rows[f"{task}/{row['_id']}"] = (i, row)
    return rows


def versions(device):
    import transformers
    return dict(device=str(device), torch=torch.__version__, transformers=transformers.__version__,
                python=platform.python_version(), platform=platform.platform(), machine=platform.machine())


def run(args):
    device = pick_device(args.device)
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[args.dtype]
    split, split_sha = load_split(args.split)
    entries = split["prompts"]
    if args.rids:
        chosen = [e for e in entries if e["rid"] in set(args.rids)]
        if len(chosen) != len(set(args.rids)):
            sys.exit(f"unknown rid(s): {sorted(set(args.rids) - {e['rid'] for e in chosen})}")
    else:
        chosen = [e for e in entries if e["split"] == args.subset]
    if any(e["split"].startswith("heldout") for e in chosen) and not args.allow_heldout:
        sys.exit("refusing to capture held-out documents without --allow-heldout")
    chosen = chosen[:args.limit] if args.limit else chosen
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [e for e in chosen if not (args.out / (e["rid"].replace("/", "__") + ".npz")).exists()]
    print(f"{len(chosen)} documents, {len(chosen) - len(todo)} already captured, device {device}, {args.dtype}")
    if not todo:
        return
    model, tok, eos = model_and_tokenizer(split, device, dtype)
    rows = longbench_rows(split, sorted({e["task"] for e in todo}))
    for e in todo:
        rid, task = e["rid"], e["task"]
        index, row = rows[rid]
        prompt = chat_prompt(tok, task, build_context(tok, row["context"], e.get("truncate")))
        ctx = hashlib.sha256(row["context"].encode()).hexdigest()[:16]
        if len(prompt) != e["prompt_tokens"] or index != e["row"]:
            msg = f"{rid}: {len(prompt)} tokens / row {index}; split says {e['prompt_tokens']} / {e['row']}"
            if not args.allow_mismatch:
                sys.exit(msg + " (pin transformers to the version in local/README.md, or pass --allow-mismatch)")
            print("WARNING", msg)
        print(f"{rid}: {len(prompt)} prompt tokens ...", flush=True)
        # Watchdog: an MPS hang blocks inside Metal, where Ctrl-C cannot reach Python. After
        # --doc-timeout seconds, print every thread's stack to stderr and exit; rerunning resumes.
        faulthandler.dump_traceback_later(args.doc_timeout, exit=True)
        ranked, scores, gen, stop, timing, _ = capture(model, prompt, device, max_new=args.max_new,
                                                       chunk=args.chunk, eos_ids=eos, progress=True)
        faulthandler.cancel_dump_traceback_later()
        stem = args.out / rid.replace("/", "__")
        np.savez(str(stem) + ".tmp.npz", ranked=ranked, scores=scores, rid=np.array(rid))
        Path(str(stem) + ".tmp.npz").replace(str(stem) + ".npz")
        manifest = dict(rid=rid, task=task, split=e["split"], truncate=e.get("truncate"), split_file=str(args.split), split_sha256=split_sha,
                        row_index=index, context_sha16=ctx, context_hash_matches=(ctx == e.get("ctx")),
                        prompt_tokens=len(prompt), prompt_ids_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                        model=split["model"], data=split["data"], dtype=args.dtype, prefill_chunk=args.chunk,
                        page_size=PAGE, top_k=TOPK, max_new_tokens=args.max_new, stop=stop,
                        decode_steps=int(ranked.shape[0]), generated_token_ids=gen,
                        ranked_sha256=hashlib.sha256(ranked.tobytes()).hexdigest(),
                        scores_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                        timing=timing, **versions(device))
        Path(str(stem) + ".json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"  {ranked.shape[0]} steps, stop={stop}, prefill {timing['prefill_s']}s, "
              f"{timing['ms_per_step']} ms/step", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()


def check(args):
    """Environment and selector self-test on a 4K-token slice of the first dev document."""
    device = pick_device(args.device)
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[args.dtype]
    print(versions(device))
    if device.type != "mps":
        print("NOTE: MPS not available; running on", device)
    split, _ = load_split(args.split)
    e = next(p for p in split["prompts"] if p["split"] == "dev")
    model, tok, eos = model_and_tokenizer(split, device, dtype)
    index, row = longbench_rows(split, [e["task"]])[e["rid"]]
    full = chat_prompt(tok, e["task"], row["context"])
    print(f"full prompt {len(full)} tokens (split says {e['prompt_tokens']}): "
          f"{'OK' if len(full) == e['prompt_tokens'] else 'MISMATCH'}")
    words = row["context"].split()
    prompt = chat_prompt(tok, e["task"], " ".join(words[:3000]))
    print(f"self-test prompt {len(prompt)} tokens, 48 decode steps, twice")
    a = capture(model, prompt, device, max_new=49, chunk=args.chunk, eos_ids=(), check_every=4)
    b = capture(model, prompt, device, max_new=49, chunk=args.chunk, eos_ids=())
    c = capture(model, prompt, device, max_new=49, chunk=len(prompt), eos_ids=())
    st = a[5]
    ok = dict(
        selector_bounds=st["max_rel_err"] < 1e-3,
        stable_sort=st["sort_errors"] == 0,
        rerun_identical=bool((a[0] == b[0]).all() and a[2] == b[2]),
    )
    print(f"checked {st['checked']} (step, layer) pairs: max rel score err {st['max_rel_err']:.2e}, "
          f"sort errors {st['sort_errors']}, fp32-CPU selected-set differences {st['set_mismatch']} (near-ties, info only)")
    print(f"rerun: ranks identical {bool((a[0] == b[0]).all())}, tokens identical {a[2] == b[2]}")
    same = float((np.sort(a[0], -1) == np.sort(c[0], -1)).all(-1).mean())
    print(f"chunked vs one-shot prefill: same selected set in {same:.1%} of (step, layer); "
          f"tokens identical {a[2] == c[2]} (info only: all our traces use the same chunk size)")
    print(f"speed: prefill {a[4]['prefill_s']}s for {len(prompt)} tokens, {b[4]['ms_per_step']} ms/decode step")
    for k, v in ok.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    sys.exit(0 if all(ok.values()) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "run"])
    ap.add_argument("--split", type=Path, default=ROOT / "splits/longctx_v1.json")
    ap.add_argument("--subset", default="dev")
    ap.add_argument("--rids", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "traces/mps_dev")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--allow-heldout", action="store_true")
    ap.add_argument("--allow-mismatch", action="store_true")
    ap.add_argument("--doc-timeout", type=int, default=900, help="seconds per document before the watchdog exits")
    args = ap.parse_args()
    check(args) if args.cmd == "check" else run(args)


if __name__ == "__main__":
    main()
