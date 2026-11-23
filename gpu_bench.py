"""GPU latency benchmark (v2 adds zc=True: sync-free, zero-copy gather)\n\nGPU latency benchmark of a real two-tier Quest KV cache (run in Colab next to the model).

Tiers: every completed 16-token page of every layer lives in pinned host memory; each layer has
B GPU page slots. Each decode step and layer: Quest scores all pages from GPU-resident min/max
bounds, takes the top-K, the replacement policy picks victims among unselected residents, missing
pages are copied host -> GPU (synchronously, no prefetch/overlap), then attention runs over the
K selected pages plus the incomplete tail page. Policies run vectorised on the GPU with the same
keys as kvpolicy.py (LRU (last, hit, -p); score EMA (ema, last, hit, -p); Belady from the trace).

modes: "resident" = same Quest attention with the whole KV on GPU (HF DynamicCache, no offload);
       "dense"    = full attention, whole KV on GPU; otherwise a tiered policy.
"""
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AttentionInterface, DynamicCache

PS, K = 16, 64


class LenCache(DynamicCache):
    """Stores nothing: hands the current token's k/v to the attention backend, tracks the length."""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self.n += k.shape[-2]
        return k, v

    def get_seq_length(self, layer_idx=0):
        return self.n


class Tier:
    """Per-layer state of the two-tier cache."""
    def __init__(self, kp, vp, tail_k, tail_v, B, pmax, policy, alpha, nxt):
        dev = "cuda"
        kvh, dim = kp.shape[1], kp.shape[3]
        self.kh = torch.empty((pmax, kvh, PS, dim), dtype=kp.dtype).pin_memory()
        self.vh = torch.empty_like(self.kh).pin_memory()
        n = kp.shape[0]
        self.kh[:n].copy_(kp)
        self.vh[:n].copy_(vp)
        self.sk = torch.empty((pmax if B is None else B, kvh, PS, dim), dtype=kp.dtype, device=dev)
        self.sv = torch.empty_like(self.sk)
        self.stage_k = torch.empty((K, kvh, PS, dim), dtype=kp.dtype).pin_memory()
        self.stage_v = torch.empty_like(self.stage_k).pin_memory()
        kf = kp.to(dev).float()
        self.lo, self.hi = kf.amin(2).transpose(0, 1).contiguous(), kf.amax(2).transpose(0, 1).contiguous()
        self.tk, self.tv = tail_k, tail_v  # [kvh, ntail, dim] on GPU
        self.np = n
        self.B, self.policy, self.alpha, self.nxt = B, policy, alpha, nxt
        self.pos = torch.full((B,), -1, dtype=torch.long, device=dev)       # page of slot
        self.slot = torch.full((pmax,), -1, dtype=torch.long, device=dev)   # slot of page
        self.last = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.hit = torch.zeros(B, dtype=torch.long, device=dev)
        self.ema = torch.zeros(B, dtype=torch.float32, device=dev)
        self.nx = torch.zeros(B, dtype=torch.long, device=dev)
        self.pmax = pmax
        self.misses = 0


class Bench:
    def __init__(s, mode, budgets=None, alpha=None, trace_ranked=None, timing=False):
        s.mode, s.budgets, s.alpha, s.timing = mode, budgets, alpha, timing
        s.trace = trace_ranked  # [T, L, K] numpy, Belady only
        s.step, s.tiers, s.times = 0, {}, []
        s.RANK = []

    # ---- attention backends -------------------------------------------------------------
    def quest_resident(s, module, query, key, value, attention_mask, scaling, **kw):
        l = module.layer_idx
        kvh, length, dim = key.shape[1:]
        hp = (length - 1) // PS
        lo, hi = s.bounds_res(l, key, hp)
        g = query.shape[1] // kvh
        q = query[0, :, 0].reshape(kvh, g, dim).float()
        sc = (torch.einsum("hgd,hpd->hgp", q.clamp_min(0), hi)
              + torch.einsum("hgd,hpd->hgp", q.clamp_max(0), lo)).mean((0, 1))
        ranked = sc.argsort(descending=True, stable=True)[:K]
        s.rec[l] = ranked
        off = torch.arange(PS, device=key.device)
        pos = torch.cat(((ranked.sort().values[:, None]*PS + off).flatten(),
                         torch.arange(hp*PS, length, device=key.device)))
        k = key.index_select(2, pos).repeat_interleave(g, 1)
        v = value.index_select(2, pos).repeat_interleave(g, 1)
        out = F.scaled_dot_product_attention(query, k, v, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    def bounds_res(s, l, key, hp):
        past = s.rb.get(l)
        cached = 0 if past is None else past[0].shape[1]
        if hp > cached:
            kvh, dim = key.shape[1], key.shape[3]
            b = key[0, :, cached*PS:hp*PS].reshape(kvh, hp-cached, PS, dim)
            lo, hi = b.amin(2).float(), b.amax(2).float()
            if past is not None:
                lo, hi = torch.cat((past[0], lo), 1), torch.cat((past[1], hi), 1)
            s.rb[l] = lo, hi
        return s.rb[l]

    def ev(s):
        if not s.timing:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def quest_tiered(s, module, query, key, value, attention_mask, scaling, **kw):
        l = module.layer_idx
        T = s.tiers[l]
        e0 = s.ev()
        # 1. append the new token; a completed page goes to host (bounds stay on GPU)
        T.tk = torch.cat((T.tk, key[0]), 1)
        T.tv = torch.cat((T.tv, value[0]), 1)
        if T.tk.shape[1] > PS:  # page T.np complete
            pk, pv = T.tk[:, :PS], T.tv[:, :PS]
            pf = pk.float()
            T.lo = torch.cat((T.lo, pf.amin(1)[:, None]), 1)
            T.hi = torch.cat((T.hi, pf.amax(1)[:, None]), 1)
            T.kh[T.np].copy_(pk, non_blocking=True)
            T.vh[T.np].copy_(pv, non_blocking=True)
            T.np += 1
            T.tk, T.tv = T.tk[:, PS:], T.tv[:, PS:]
        kvh, dim = key.shape[1], key.shape[3]
        g = query.shape[1] // kvh
        e1 = s.ev()
        # 2. Quest scores and top-K
        q = query[0, :, 0].reshape(kvh, g, dim).float()
        sc = (torch.einsum("hgd,hpd->hgp", q.clamp_min(0), T.hi)
              + torch.einsum("hgd,hpd->hgp", q.clamp_max(0), T.lo)).mean((0, 1))
        ranked = sc.argsort(descending=True, stable=True)[:K]
        s.rec[l] = ranked
        e2 = s.ev()
        # 3. replacement policy (vectorised, keys as in kvpolicy.py)
        t = s.step
        occ = T.pos >= 0
        pg = T.pos.clamp_min(0)
        hs = T.slot[ranked]
        ishit = hs >= 0
        if T.policy == "ema":
            kth, top = sc[ranked[-1]], sc[ranked[0]]
            z = (sc - kth) / (top - kth + 1e-6)
            a = T.alpha
            T.ema = torch.where(occ, a * T.ema + (1 - a) * z[pg], T.ema)
        sel = torch.zeros(T.B, dtype=torch.bool, device=sc.device)
        sel[hs[ishit]] = True
        big = T.pmax * 4
        sec = T.last * big + T.hit * T.pmax * 2 + (T.pmax - 1 - pg)   # (last, hit, -p)
        if T.policy == "belady":
            prim = -T.nx.double()
        elif T.policy == "ema":
            prim = T.ema.double()
        else:
            prim = torch.zeros(T.B, dtype=torch.float64, device=sc.device)
        prim = torch.where(occ, prim, torch.full_like(prim, -1e300))
        prim = torch.where(sel, torch.full_like(prim, 1e300), prim)
        o1 = torch.sort(sec, stable=True).indices
        order = o1[torch.sort(prim[o1], stable=True).indices]
        mp = ranked[~ishit]
        mp_cpu = mp.cpu()  # one sync per layer: miss count and page ids for the host gather
        nm = mp_cpu.numel()
        e3 = s.ev()
        # 4. host -> GPU transfer of the missing pages
        if nm:
            ts = order[:nm]
            old = T.pos[ts]
            T.slot[old[old >= 0]] = -1
            T.pos[ts] = mp
            T.slot[mp] = ts
            torch.index_select(T.kh, 0, mp_cpu, out=T.stage_k[:nm])
            torch.index_select(T.vh, 0, mp_cpu, out=T.stage_v[:nm])
            T.sk[ts] = T.stage_k[:nm].to("cuda", non_blocking=True)
            T.sv[ts] = T.stage_v[:nm].to("cuda", non_blocking=True)
            T.misses += nm
        rs = T.slot[ranked]
        T.last[rs] = t
        T.hit[rs] = ishit.long()
        if T.policy == "ema" and nm:
            T.ema[T.slot[mp]] = z[mp]
        if T.policy == "belady":
            T.nx[rs] = torch.as_tensor(s.nxt_all[t, l], device=sc.device)
        e4 = s.ev()
        # 5. attention over the selected pages (ascending page id, as the resident path) + tail
        srt = ranked.sort().values
        ss = T.slot[srt]
        k = torch.cat((T.sk[ss].permute(1, 0, 2, 3).reshape(kvh, -1, dim), T.tk), 1)[None]
        v = torch.cat((T.sv[ss].permute(1, 0, 2, 3).reshape(kvh, -1, dim), T.tv), 1)[None]
        out = F.scaled_dot_product_attention(query, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1),
                                             scale=scaling)
        e5 = s.ev()
        if s.timing:
            s.evs.append((e0, e1, e2, e3, e4, e5))
        return out.transpose(1, 2).contiguous(), None


def next_use(trace):
    """nxt[t, l, k] = next step > t where page trace[t, l, k] is selected again (large if never)."""
    T, L, Kk = trace.shape
    nxt = np.full((T, L, Kk), 10**9, dtype=np.int64)
    for l in range(L):
        seen = {}
        for t in range(T - 1, -1, -1):
            for k in range(Kk):
                p = int(trace[t, l, k])
                nxt[t, l, k] = seen.get(p, 10**9)
            for k in range(Kk):
                seen[int(trace[t, l, k])] = t
    return nxt


@torch.inference_mode()
def run(model, prompt, mode, steps, budgets=None, alpha=None, trace=None, timing=False, warmup=8, zc=False):
    """Greedy decode `steps` tokens after the prefill. Returns per-step latencies (s), the ranked
    pages per step [steps, L, K], generated ids, per-layer misses, and the timing breakdown."""
    cfg = model.config
    L = cfg.num_hidden_layers
    b = Bench(mode, budgets, alpha, timing=timing)
    name = f"bench_{mode}"
    ids = torch.tensor([prompt], device="cuda")
    orig = cfg._attn_implementation
    try:
        cfg._attn_implementation = "sdpa"
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        tokn = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = [tokn.item()]
        N = len(prompt)
        if mode in ("resident", "dense"):
            b.rb = {}
            if mode == "resident":
                AttentionInterface.register(name, b.quest_resident)
                cfg._attn_implementation = name
        else:
            pmax = (N + steps + 2) // PS + 1
            if mode == "belady":
                b.nxt_all = next_use(trace)
                b.nxt_gpu = torch.as_tensor(b.nxt_all, device="cuda")
            for l in range(L):
                kf, vf = cache.key_cache[l][0], cache.value_cache[l][0]
                kvh, _, dim = kf.shape
                n = N // PS
                kp = kf[:, :n*PS].reshape(kvh, n, PS, dim).transpose(0, 1).cpu()
                vp = vf[:, :n*PS].reshape(kvh, n, PS, dim).transpose(0, 1).cpu()
                b.tiers[l] = Tier(kp, vp, kf[:, n*PS:].clone(), vf[:, n*PS:].clone(), budgets[l], pmax,
                                  mode, alpha, None)
            del cache
            torch.cuda.empty_cache()
            cache = LenCache(N)
            for T in b.tiers.values():  # one dummy slot-table entry for masked scatters (zc path)
                T.slot = torch.cat((T.slot, T.slot.new_full((1,), -1)))
                T.mgpu = torch.zeros((), dtype=torch.long, device="cuda")
                T.src = torch.empty(K, dtype=torch.int32, device="cuda")
                T.dst = torch.empty(K, dtype=torch.int32, device="cuda")
                T.old = torch.empty(K, dtype=torch.long, device="cuda")
                T.pos2, T.last2, T.hit2 = T.pos.clone(), T.last.clone(), T.hit.clone()
                T.ema2, T.nx2 = T.ema.clone(), T.nx.clone()
            fn = b.quest_fused if zc == "fused" else (b.quest_zc if zc else b.quest_tiered)
            AttentionInterface.register(name, fn)
            cfg._attn_implementation = name
        torch.cuda.reset_peak_memory_stats()
        lat, ranks, brk = [], [], []
        for step in range(steps):
            b.step, b.rec, b.evs = step, [None] * L, []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=tokn, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            tokn = out.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(tokn.item())  # syncs
            lat.append(time.perf_counter() - t0)
            if mode != "dense":
                ranks.append(torch.stack(b.rec).cpu().numpy())
            if timing and b.evs:
                torch.cuda.synchronize()
                brk.append(np.array([[e[i].elapsed_time(e[i+1]) for i in range(5)] for e in b.evs]).sum(0))
        misses = ([int(b.tiers[l].mgpu) if zc else b.tiers[l].misses for l in range(L)]
                  if b.tiers else None)
        return dict(lat=np.array(lat), ranks=np.array(ranks) if ranks else None, gen=gen, misses=misses,
                    breakdown=np.array(brk) if brk else None, peak_mem=torch.cuda.max_memory_allocated())
    finally:
        cfg._attn_implementation = orig
        b.tiers.clear()
        torch.cuda.empty_cache()


# ---- sync-free variant: GPU gathers missed pages straight from mapped pinned host memory -------
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gather(src_ptr, dst_ptr, page_ptr, slot_ptr, n_el: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)
        j = tl.program_id(1)
        p = tl.load(page_ptr + i)
        if p >= 0:
            d = tl.load(slot_ptr + i)
            offs = j * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(src_ptr + p.to(tl.int64) * n_el + offs)
            tl.store(dst_ptr + d.to(tl.int64) * n_el + offs, x)
except ImportError:
    triton = None


class _HostPtr:
    """Lets Triton take a pinned host tensor: under UVA its host pointer is valid on the device."""
    def __init__(self, t):
        self.t, self.dtype = t, t.dtype

    def data_ptr(self):
        return self.t.data_ptr()


def quest_zc(s, module, query, key, value, attention_mask, scaling, **kw):
    """Same algorithm as quest_tiered, with no host sync: fixed-size (K) victim assignment and a
    Triton gather that reads only the missed pages over PCIe (zero-copy)."""
    l = module.layer_idx
    T = s.tiers[l]
    e0 = s.ev()
    T.tk = torch.cat((T.tk, key[0]), 1)
    T.tv = torch.cat((T.tv, value[0]), 1)
    if T.tk.shape[1] > PS:
        pk, pv = T.tk[:, :PS], T.tv[:, :PS]
        pf = pk.float()
        T.lo = torch.cat((T.lo, pf.amin(1)[:, None]), 1)
        T.hi = torch.cat((T.hi, pf.amax(1)[:, None]), 1)
        T.kh[T.np].copy_(pk, non_blocking=True)
        T.vh[T.np].copy_(pv, non_blocking=True)
        T.np += 1
        T.tk, T.tv = T.tk[:, PS:], T.tv[:, PS:]
    kvh, dim = key.shape[1], key.shape[3]
    g = query.shape[1] // kvh
    e1 = s.ev()
    q = query[0, :, 0].reshape(kvh, g, dim).float()
    sc = (torch.einsum("hgd,hpd->hgp", q.clamp_min(0), T.hi)
          + torch.einsum("hgd,hpd->hgp", q.clamp_max(0), T.lo)).mean((0, 1))
    ranked = sc.argsort(descending=True, stable=True)[:K]
    s.rec[l] = ranked
    e2 = s.ev()
    t = s.step
    B = T.B
    occ = T.pos >= 0
    pg = T.pos.clamp_min(0)
    hs = T.slot[ranked]
    ishit = hs >= 0
    miss = ~ishit
    if T.policy == "ema":
        kth, top = sc[ranked[-1]], sc[ranked[0]]
        z = (sc - kth) / (top - kth + 1e-6)
        T.ema = torch.where(occ, T.alpha * T.ema + (1 - T.alpha) * z[pg], T.ema)
    sel = torch.zeros(B + 1, dtype=torch.bool, device=sc.device)
    sel[torch.where(ishit, hs, B)] = True
    sel = sel[:B]
    sec = T.last * (T.pmax * 4) + T.hit * T.pmax * 2 + (T.pmax - 1 - pg)
    if T.policy == "belady":
        prim = -T.nx.double()
    elif T.policy == "ema":
        prim = T.ema.double()
    else:
        prim = torch.zeros(B, dtype=torch.float64, device=sc.device)
    prim = torch.where(occ, prim, torch.full_like(prim, -1e300))
    prim = torch.where(sel, torch.full_like(prim, 1e300), prim)
    o1 = torch.sort(sec, stable=True).indices
    order = o1[torch.sort(prim[o1], stable=True).indices]
    cum = torch.cumsum(miss.long(), 0) - 1
    tgt = torch.where(miss, order[cum.clamp_min(0)], hs)          # slot of every ranked page
    old = T.pos[tgt]
    T.slot[torch.where(miss & (old >= 0), old, T.pmax)] = -1      # index pmax is a dummy entry
    T.slot[T.pmax] = -1
    T.pos[tgt] = ranked
    T.slot[ranked] = tgt
    T.mgpu += miss.sum()
    e3 = s.ev()
    src = torch.where(miss, ranked, -1).to(torch.int32)
    dst = tgt.to(torch.int32)
    n_el = kvh * PS * dim
    _gather[(K, n_el // 1024)](_HostPtr(T.kh), T.sk, src, dst, n_el, 1024)
    _gather[(K, n_el // 1024)](_HostPtr(T.vh), T.sv, src, dst, n_el, 1024)
    T.last[tgt] = t
    T.hit[tgt] = ishit.long()
    if T.policy == "ema":
        T.ema[tgt] = torch.where(miss, z[ranked], T.ema[tgt])
    if T.policy == "belady":
        T.nx[tgt] = s.nxt_gpu[t, l]
    e4 = s.ev()
    srt = ranked.sort().values
    ss = T.slot[srt]
    k = torch.cat((T.sk[ss].permute(1, 0, 2, 3).reshape(kvh, -1, dim), T.tk), 1)[None]
    v = torch.cat((T.sv[ss].permute(1, 0, 2, 3).reshape(kvh, -1, dim), T.tv), 1)[None]
    out = F.scaled_dot_product_attention(query, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1),
                                         scale=scaling)
    e5 = s.ev()
    if s.timing:
        s.evs.append((e0, e1, e2, e3, e4, e5))
    return out.transpose(1, 2).contiguous(), None


Bench.quest_zc = quest_zc


# ---- fused path: one Triton kernel per layer for the whole policy step, one for the K/V gather --
if triton is not None:
    @triton.jit
    def _policy(sc_ptr, rk_ptr, pos_ptr, slot_ptr, last_ptr, hit_ptr, ema_ptr, nx_ptr, nxrow_ptr,
                pos2, last2, hit2, ema2, nx2, src_ptr, dst_ptr, old_ptr, cnt_ptr, t, alpha, pmax, B,
                POL: tl.constexpr, KK: tl.constexpr, BB: tl.constexpr):
        """POL 0 = LRU, 1 = score EMA, 2 = Belady. Keys as kvpolicy.py: (prim, last, hit, -page).
        Reads the current per-slot state and writes the next state to the second buffer set (no
        in-place read-after-write inside the kernel); the page->slot table is updated by _slotup."""
        i = tl.arange(0, BB)
        vi = i < B
        k = tl.arange(0, KK)
        rk = tl.load(rk_ptr + k)
        pos = tl.load(pos_ptr + i, mask=vi, other=-1)
        occ = pos >= 0
        pg = tl.where(occ, pos, 0)
        last = tl.load(last_ptr + i, mask=vi, other=0)
        hit = tl.load(hit_ptr + i, mask=vi, other=0)
        hs = tl.load(slot_ptr + rk)
        ishit = hs >= 0
        miss = hs < 0
        sel = tl.sum(((hs[:, None] == i[None, :]) & ishit[:, None]).to(tl.int32), 0) > 0
        top = tl.load(sc_ptr + tl.load(rk_ptr))
        kth = tl.load(sc_ptr + tl.load(rk_ptr + KK - 1))
        den = top - kth + 1e-6
        if POL == 1:
            ema = tl.load(ema_ptr + i, mask=vi, other=0.0)
            z = (tl.load(sc_ptr + pg, mask=vi & occ, other=0.0) - kth) / den
            ema = tl.where(occ, alpha * ema + (1 - alpha) * z, ema)
            prim = ema.to(tl.float64)
        elif POL == 2:
            nx = tl.load(nx_ptr + i, mask=vi, other=0)
            prim = -nx.to(tl.float64)
        else:
            prim = tl.zeros([BB], dtype=tl.float64)
        prim = tl.where(occ, prim, -1e300)
        prim = tl.where(sel | (i >= B), 1e300, prim)
        sec = last.to(tl.int64) * (pmax * 4) + hit.to(tl.int64) * pmax * 2 + (pmax - 1 - pg).to(tl.int64)
        pa, pb = prim[:, None], prim[None, :]
        sa, sb = sec[:, None], sec[None, :]
        ia, ib = i[:, None], i[None, :]
        less = (pa < pb) | ((pa == pb) & ((sa < sb) | ((sa == sb) & (ia < ib))))
        rank = tl.sum(less.to(tl.int32), 0)
        cum = tl.cumsum(miss.to(tl.int32), 0) - 1
        vic = tl.sum(tl.where((rank[None, :] == cum[:, None]) & miss[:, None], i[None, :], 0), 1)
        tgt = tl.where(miss, vic, hs)
        m2 = tgt[:, None] == i[None, :]                       # [KK, BB]: ranked page k goes to slot i
        touched = tl.sum(m2.to(tl.int32), 0) > 0
        old = tl.sum(tl.where(m2, pos[None, :], 0), 1)        # page previously in tgt slot
        pos_f = tl.where(touched, tl.sum(tl.where(m2, rk[:, None], 0), 0), pos)
        hit_f = tl.where(touched, tl.sum(tl.where(m2, ishit[:, None].to(tl.int64), 0), 0), hit)
        last_f = tl.where(touched, t, last)
        tl.store(pos2 + i, pos_f, mask=vi)
        tl.store(hit2 + i, hit_f, mask=vi)
        tl.store(last2 + i, last_f.to(tl.int64), mask=vi)
        if POL == 1:
            zr = (tl.load(sc_ptr + rk) - kth) / den
            mv = m2 & miss[:, None]
            isv = tl.sum(mv.to(tl.int32), 0) > 0
            ema_f = tl.where(isv, tl.sum(tl.where(mv, zr[:, None], 0.0), 0), ema)
            tl.store(ema2 + i, ema_f, mask=vi)
        if POL == 2:
            nr = tl.load(nxrow_ptr + k)
            nx_f = tl.where(touched, tl.sum(tl.where(m2, nr[:, None], 0), 0), nx)
            tl.store(nx2 + i, nx_f, mask=vi)
        tl.store(src_ptr + k, tl.where(miss, rk, -1).to(tl.int32))
        tl.store(dst_ptr + k, tgt.to(tl.int32))
        tl.store(old_ptr + k, tl.where(miss & (old >= 0), old, pmax))
        tl.atomic_add(cnt_ptr, tl.sum(miss.to(tl.int64), 0))

    @triton.jit
    def _slotup(slot_ptr, rk_ptr, dst_ptr, old_ptr, KK: tl.constexpr):
        """Write-only update of the page->slot table: evicted pages -> -1, selected pages -> slot.
        Evicted pages are never selected, so the two sets of addresses are disjoint (pmax = dummy)."""
        k = tl.arange(0, KK)
        tl.store(slot_ptr + tl.load(old_ptr + k), -1)
        tl.store(slot_ptr + tl.load(rk_ptr + k), tl.load(dst_ptr + k).to(tl.int64))

    @triton.jit
    def _gather2(ksrc, vsrc, kdst, vdst, page_ptr, slot_ptr, n_el: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)
        j = tl.program_id(1)
        p = tl.load(page_ptr + i)
        if p >= 0:
            d = tl.load(slot_ptr + i).to(tl.int64)
            offs = j * BLOCK + tl.arange(0, BLOCK)
            p64 = p.to(tl.int64)
            tl.store(kdst + d * n_el + offs, tl.load(ksrc + p64 * n_el + offs))
            tl.store(vdst + d * n_el + offs, tl.load(vsrc + p64 * n_el + offs))


def quest_fused(s, module, query, key, value, attention_mask, scaling, **kw):
    l = module.layer_idx
    T = s.tiers[l]
    T.tk = torch.cat((T.tk, key[0]), 1)
    T.tv = torch.cat((T.tv, value[0]), 1)
    if T.tk.shape[1] > PS:
        pk, pv = T.tk[:, :PS], T.tv[:, :PS]
        pf = pk.float()
        T.lo = torch.cat((T.lo, pf.amin(1)[:, None]), 1)
        T.hi = torch.cat((T.hi, pf.amax(1)[:, None]), 1)
        T.kh[T.np].copy_(pk, non_blocking=True)
        T.vh[T.np].copy_(pv, non_blocking=True)
        T.np += 1
        T.tk, T.tv = T.tk[:, PS:], T.tv[:, PS:]
    kvh, dim = key.shape[1], key.shape[3]
    g = query.shape[1] // kvh
    q = query[0, :, 0].reshape(kvh, g, dim).float()
    sc = (torch.einsum("hgd,hpd->hgp", q.clamp_min(0), T.hi)
          + torch.einsum("hgd,hpd->hgp", q.clamp_max(0), T.lo)).mean((0, 1))
    ranked = sc.argsort(descending=True, stable=True)[:K]
    s.rec[l] = ranked
    pol = {"lru": 0, "ema": 1, "belady": 2}[T.policy]
    nxrow = s.nxt_gpu[s.step, l] if pol == 2 else T.nx
    _policy[(1,)](sc, ranked, T.pos, T.slot, T.last, T.hit, T.ema, T.nx, nxrow,
                  T.pos2, T.last2, T.hit2, T.ema2, T.nx2, T.src, T.dst, T.old, T.mgpu,
                  s.step, float(T.alpha or 0.0), T.pmax, T.B, POL=pol, KK=K,
                  BB=triton.next_power_of_2(T.B), num_warps=8)
    _slotup[(1,)](T.slot, ranked, T.dst, T.old, KK=K)
    T.pos, T.pos2 = T.pos2, T.pos
    T.last, T.last2 = T.last2, T.last
    T.hit, T.hit2 = T.hit2, T.hit
    if pol == 1:
        T.ema, T.ema2 = T.ema2, T.ema
    if pol == 2:
        T.nx, T.nx2 = T.nx2, T.nx
    n_el = kvh * PS * dim
    _gather2[(K, n_el // 1024)](_HostPtr(T.kh), _HostPtr(T.vh), T.sk, T.sv, T.src, T.dst, n_el, 1024)
    srt = ranked.sort().values
    ss = T.slot[srt]
    k = torch.cat((T.sk[ss].permute(1, 0, 2, 3).reshape(kvh, -1, dim), T.tk), 1)[None]
    v = torch.cat((T.sv[ss].permute(1, 0, 2, 3).reshape(kvh, -1, dim), T.tv), 1)[None]
    out = F.scaled_dot_product_attention(query, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1),
                                         scale=scaling)
    return out.transpose(1, 2).contiguous(), None


Bench.quest_fused = quest_fused
