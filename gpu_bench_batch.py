"""Batched two-tier Quest KV cache (batch > 1) for S7 (docs/workshop/rerun_plan.md D7).

R requests of equal prompt length decode together in one HF forward per step. Per request and per
layer there are B GPU page slots (B may differ per layer: per-layer allocation); every completed
16-token page lives in host memory. Each step and layer:
  1. append the new token (a completed page goes to the host tier; its min/max bounds stay on GPU);
  2. Quest scores of all pages for every request (one batched einsum) and top-K per request;
  3. replacement policy per request, vectorised over the batch with exactly the keys of kvpolicy.py
     and gpu_bench.quest_zc: (prim, last, hit, -page), prim = 0 (LRU), EMA of z (score EMA; alpha = 0
     is the current score), or -next use (Belady, from a recorded trace);
  4. one host -> GPU transfer per layer for the whole batch: the missed pages of every request are
     gathered on the host into one pinned staging buffer, copied with one H2D copy and scattered to
     their slots (one sync per layer, to learn the miss ids, as gpu_bench.quest_tiered);
  5. attention over each request's K selected pages (ascending page id) plus its tail, one batched
     SDPA call.
Prefill is per request (batch 1, dense SDPA), then the KV is moved to the host tier.

Selection does not depend on the policy (the attention reads the same K pages whatever slot they
occupy), so the ranked pages and the generated tokens must be identical across policies; Belady uses
the ranked pages recorded by an earlier run of the same batch.
"""
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AttentionInterface

from gpu_bench import LenCache

PS = 16


class BTier:
    def __init__(s, R, kvh, dim, pmax, B, dtype, policy, alpha):
        dev = "cuda"
        s.R, s.kvh, s.dim, s.pmax, s.B, s.policy, s.alpha = R, kvh, dim, pmax, B, policy, alpha
        s.kh = torch.empty((R * pmax, kvh, PS, dim), dtype=dtype)       # host tier (pageable)
        s.vh = torch.empty_like(s.kh)
        s.lo = torch.zeros((R, kvh, pmax, dim), dtype=torch.float32, device=dev)
        s.hi = torch.zeros_like(s.lo)
        s.sk = torch.empty((R * B, kvh, PS, dim), dtype=dtype, device=dev)
        s.sv = torch.empty_like(s.sk)
        s.pos = torch.full((R, B), -1, dtype=torch.long, device=dev)
        s.slot = torch.full((R, pmax + 1), -1, dtype=torch.long, device=dev)
        s.last = torch.full((R, B), -1, dtype=torch.long, device=dev)
        s.hit = torch.zeros((R, B), dtype=torch.long, device=dev)
        s.ema = torch.zeros((R, B), dtype=torch.float32, device=dev)
        s.nx = torch.zeros((R, B), dtype=torch.long, device=dev)
        s.misses = torch.zeros(R, dtype=torch.long, device=dev)
        s.np = 0
        s.tk = s.tv = None


class BatchBench:
    def __init__(s, policy, budgets, alpha, K, timing=False, nxt=None, record_scores=False):
        s.policy, s.budgets, s.alpha, s.K, s.timing = policy, budgets, alpha, K, timing
        s.nxt = nxt                    # [T, R, L, K] next-use steps (Belady)
        s.tiers, s.step, s.rec, s.evs = {}, 0, None, []
        s.record_scores, s.scores = record_scores, None
        s.stage_k = s.stage_v = None
        s.bytes_moved = 0

    def ev(s):
        if not s.timing:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def attn(s, module, query, key, value, attention_mask, scaling, **kw):
        l = module.layer_idx
        T = s.tiers[l]
        R, kvh, dim, K = T.R, T.kvh, T.dim, s.K
        ar = torch.arange(R, device=query.device)
        e0 = s.ev()
        # 1. append; a completed page goes to the host tier (all requests have the same length)
        T.tk = torch.cat((T.tk, key), 2)
        T.tv = torch.cat((T.tv, value), 2)
        if T.tk.shape[2] > PS:
            pk, pv = T.tk[:, :, :PS], T.tv[:, :, :PS]          # [R, kvh, PS, d]
            pf = pk.float()
            T.lo[:, :, T.np] = pf.amin(2)
            T.hi[:, :, T.np] = pf.amax(2)
            idx = ar.cpu() * T.pmax + T.np
            T.kh.index_copy_(0, idx, pk.cpu())
            T.vh.index_copy_(0, idx, pv.cpu())
            T.np += 1
            T.tk, T.tv = T.tk[:, :, PS:], T.tv[:, :, PS:]
        g = query.shape[1] // kvh
        n = T.np
        e1 = s.ev()
        # 2. scores and top-K per request
        q = query[:, :, 0].reshape(R, kvh, g, dim).float()
        sc = (torch.einsum("rhgd,rhpd->rhgp", q.clamp_min(0), T.hi[:, :, :n])
              + torch.einsum("rhgd,rhpd->rhgp", q.clamp_max(0), T.lo[:, :, :n])).mean((1, 2))   # [R, n]
        ranked = sc.argsort(dim=1, descending=True, stable=True)[:, :K]
        s.rec[l] = ranked
        if s.record_scores:
            s.scores[l] = sc
        e2 = s.ev()
        # 3. policy (batched quest_zc)
        t, B, pmax = s.step, T.B, T.pmax
        occ = T.pos >= 0
        pg = T.pos.clamp_min(0)
        hs = T.slot.gather(1, ranked)
        ishit = hs >= 0
        miss = ~ishit
        if T.policy == "ema":
            kth, top = sc.gather(1, ranked[:, -1:]), sc.gather(1, ranked[:, :1])
            z = (sc - kth) / (top - kth + 1e-6)
            T.ema = torch.where(occ, T.alpha * T.ema + (1 - T.alpha) * z.gather(1, pg.clamp_max(n - 1)), T.ema)
        sel = torch.zeros((R, B + 1), dtype=torch.bool, device=sc.device)
        sel.scatter_(1, torch.where(ishit, hs, B), True)
        sel = sel[:, :B]
        sec = T.last * (pmax * 4) + T.hit * pmax * 2 + (pmax - 1 - pg)
        if T.policy == "belady":
            prim = -T.nx.double()
        elif T.policy == "ema":
            prim = T.ema.double()
        else:
            prim = torch.zeros((R, B), dtype=torch.float64, device=sc.device)
        prim = torch.where(occ, prim, torch.full_like(prim, -1e300))
        prim = torch.where(sel, torch.full_like(prim, 1e300), prim)
        o1 = torch.sort(sec, dim=1, stable=True).indices
        order = o1.gather(1, torch.sort(prim.gather(1, o1), dim=1, stable=True).indices)
        cum = torch.cumsum(miss.long(), 1) - 1
        tgt = torch.where(miss, order.gather(1, cum.clamp_min(0)), hs)
        old = T.pos.gather(1, tgt)
        T.slot.scatter_(1, torch.where(miss & (old >= 0), old, pmax), -1)
        T.slot[:, pmax] = -1
        T.pos.scatter_(1, tgt, ranked)
        T.slot.scatter_(1, ranked, tgt)
        T.misses += miss.sum(1)
        src = torch.where(miss, ranked, -1)
        src_cpu = src.cpu()                  # one sync per layer for the whole batch
        e3 = s.ev()
        # 4. one gather + one H2D copy for every missed page of the batch
        r_idx, k_idx = torch.nonzero(src_cpu >= 0, as_tuple=True)
        nm = int(r_idx.numel())
        if nm:
            flat = r_idx * pmax + src_cpu[r_idx, k_idx]
            torch.index_select(T.kh, 0, flat, out=s.stage_k[:nm])
            torch.index_select(T.vh, 0, flat, out=s.stage_v[:nm])
            gk = s.stage_k[:nm].to("cuda", non_blocking=True)
            gv = s.stage_v[:nm].to("cuda", non_blocking=True)
            r_g, k_g = r_idx.to("cuda"), k_idx.to("cuda")
            dst = r_g * B + tgt[r_g, k_g]
            T.sk[dst] = gk
            T.sv[dst] = gv
            s.bytes_moved += 2 * gk.numel() * gk.element_size()
        T.last.scatter_(1, tgt, t)
        T.hit.scatter_(1, tgt, ishit.long())
        if T.policy == "ema":
            T.ema.scatter_(1, tgt, torch.where(miss, z.gather(1, ranked), T.ema.gather(1, tgt)))
        if T.policy == "belady":
            T.nx.scatter_(1, tgt, s.nxt_gpu[t, :, l])
        e4 = s.ev()
        # 5. batched attention over the K selected pages (ascending page id) + tail
        srt = ranked.sort(dim=1).values
        ss = T.slot.gather(1, srt)                                  # [R, K]
        flat = (ar[:, None] * B + ss).reshape(-1)
        kp = T.sk[flat].reshape(R, K, kvh, PS, dim).permute(0, 2, 1, 3, 4).reshape(R, kvh, K * PS, dim)
        vp = T.sv[flat].reshape(R, K, kvh, PS, dim).permute(0, 2, 1, 3, 4).reshape(R, kvh, K * PS, dim)
        k = torch.cat((kp, T.tk), 2).repeat_interleave(g, 1)
        v = torch.cat((vp, T.tv), 2).repeat_interleave(g, 1)
        out = F.scaled_dot_product_attention(query, k, v, scale=scaling)
        e5 = s.ev()
        if s.timing:
            s.evs.append((e0, e1, e2, e3, e4, e5))
        return out.transpose(1, 2).contiguous(), None


@torch.inference_mode()
def run(model, prompts, policy, steps, budgets, alpha=0.0, K=64, nxt=None, warmup=8, timing=False,
        record_scores=False):
    """prompts: R token lists of equal length. Greedy decode `steps` tokens for all requests.
    Returns per-step wall latency (s), ranked [steps, R, L, K], generated [R, steps+1], misses [R, L],
    bytes moved, memory by phase, optional timing breakdown and scores."""
    R = len(prompts)
    N = len(prompts[0])
    if any(len(p) != N for p in prompts):
        raise ValueError("equal prompt lengths required")
    cfg = model.config
    L = cfg.num_hidden_layers
    if isinstance(budgets, int):
        budgets = [budgets] * L
    b = BatchBench(policy, budgets, alpha, K, timing, nxt, record_scores)
    name = "bench_batch"
    orig = cfg._attn_implementation
    pmax = (N + steps + 2) // PS + 1
    try:
        cfg._attn_implementation = "sdpa"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        first = []
        for r, prompt in enumerate(prompts):
            out = model(input_ids=torch.tensor([prompt], device="cuda"), use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            first.append(out.logits[:, -1].argmax(-1).item())
            del out
            for l in range(L):
                kf, vf = cache.key_cache[l][0], cache.value_cache[l][0]
                kvh, _, dim = kf.shape
                if r == 0:
                    b.tiers[l] = BTier(R, kvh, dim, pmax, budgets[l], kf.dtype, policy, alpha)
                    b.tiers[l].tk = torch.empty((R, kvh, N % PS, dim), dtype=kf.dtype, device="cuda")
                    b.tiers[l].tv = torch.empty_like(b.tiers[l].tk)
                T = b.tiers[l]
                n = N // PS
                kp = kf[:, :n * PS].reshape(kvh, n, PS, dim).transpose(0, 1)
                vp = vf[:, :n * PS].reshape(kvh, n, PS, dim).transpose(0, 1)
                T.kh[r * pmax: r * pmax + n].copy_(kp.cpu())
                T.vh[r * pmax: r * pmax + n].copy_(vp.cpu())
                kpf = kp.float()
                T.lo[r, :, :n] = kpf.amin(2).transpose(0, 1)
                T.hi[r, :, :n] = kpf.amax(2).transpose(0, 1)
                T.np = n
                T.tk[r] = kf[:, n * PS:]
                T.tv[r] = vf[:, n * PS:]
            del cache, kf, vf, kp, vp
        torch.cuda.synchronize()
        peak_prefill = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        T0 = b.tiers[0]
        b.stage_k = torch.empty((R * K, T0.kvh, PS, T0.dim), dtype=T0.kh.dtype).pin_memory()
        b.stage_v = torch.empty_like(b.stage_k).pin_memory()
        if policy == "belady":
            b.nxt_gpu = torch.as_tensor(nxt, device="cuda")
        AttentionInterface.register(name, b.attn)
        cfg._attn_implementation = name
        cache = LenCache(N)
        tokn = torch.tensor(first, device="cuda")[:, None]
        gen = [list(x) for x in zip(first)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated()
        lat, ranks, brk, scores = [], [], [], []
        peak_warm = None
        for step in range(steps):
            b.step, b.rec, b.evs = step, [None] * L, []
            if record_scores:
                b.scores = [None] * L
            torch.cuda.synchronize()
            if step == warmup and warmup > 0:
                peak_warm = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = model(input_ids=tokn, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            tokn = out.logits[:, -1].argmax(-1, keepdim=True)
            nxt_ids = tokn[:, 0].tolist()  # syncs
            lat.append(time.perf_counter() - t0)
            for r in range(R):
                gen[r].append(nxt_ids[r])
            ranks.append(torch.stack(b.rec, 1).cpu().numpy())        # [R, L, K]
            if record_scores:
                scores.append([sc.cpu().numpy() for sc in b.scores])
            if timing and b.evs:
                torch.cuda.synchronize()
                brk.append(np.array([[e[i].elapsed_time(e[i + 1]) for i in range(5)] for e in b.evs]).sum(0))
        torch.cuda.synchronize()
        peak_steady = torch.cuda.max_memory_allocated()
        misses = np.stack([b.tiers[l].misses.cpu().numpy() for l in range(L)], 1)   # [R, L]
        lat = np.array(lat)
        return dict(lat=lat, ranks=np.array(ranks), gen=gen, misses=misses, bytes=b.bytes_moved,
                    breakdown=np.array(brk) if brk else None, scores=scores if record_scores else None,
                    warmup=warmup, peak_prefill=peak_prefill, mem_start=mem_start,
                    peak_decode_warmup=peak_warm, peak_decode_steady=peak_steady,
                    host_bytes=sum(2 * T.kh.numel() * T.kh.element_size() for T in b.tiers.values()))
    finally:
        cfg._attn_implementation = orig
        b.tiers.clear()
        torch.cuda.empty_cache()


def next_use_batch(ranks):
    """ranks [T, R, L, K] -> next-use step after t for each (t, r, l, k) (1e9 if never)."""
    T, R, L, Kk = ranks.shape
    nxt = np.full(ranks.shape, 10 ** 9, dtype=np.int64)
    for r in range(R):
        for l in range(L):
            seen = {}
            for t in range(T - 1, -1, -1):
                row = ranks[t, r, l]
                for k in range(Kk):
                    nxt[t, r, l, k] = seen.get(int(row[k]), 10 ** 9)
                for k in range(Kk):
                    seen[int(row[k])] = t
    return nxt
