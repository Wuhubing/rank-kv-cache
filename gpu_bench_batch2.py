"""Batched two-tier Quest KV cache, fixed path (followup-v3 E1). gpu_bench_batch.py is unchanged.

Differences from gpu_bench_batch.py (R5 of the followup-v3 task):
  * the host tier is pinned memory (was pageable);
  * the replacement policy of every request runs in one fused Triton kernel per layer (grid = requests; the batch-1
    kernel gpu_bench._policy with per-request offsets), with no host synchronisation: the miss list stays on the GPU;
  * missed pages are fetched by a zero-copy Triton gather that reads only the missed pages from the pinned host tier
    over PCIe (UVA) and writes them into their slots; it is issued on a separate CUDA stream after the policy kernel
    and overlaps the rest of the layer's bookkeeping on the main stream; the main stream waits for it only right
    before attention;
  * a newly completed page is written to the host tier asynchronously on the same side stream (zero-copy store), so
    it is ordered before any later fetch of that page; tensors the side stream reads are kept alive until the end of the
    step (the main stream waits for the side stream in every layer), instead of Tensor.record_stream, which crashed the
    caching allocator (segfault in empty_cache) after several runs in one process;
  * per-step miss counts are read after the step's timing ends (the token read-out already synchronises).
Policies: lru, ema (alpha; alpha=0 is the current score), belady (next uses from an earlier run of the same batch),
"nofetch" (LRU bookkeeping at the given budget but the fetch and the write-back are skipped: the same pipeline with
zero transfer; attention then reads whatever the slots hold, so its numbers are meaningless and the generated tokens
may differ, only the timing is used) and "allmiss" (the cache is emptied every step, so every selected page is
fetched: K misses per request and layer) for the upper bound of E3. A true all-hit run (B >= every page) does not fit
in 48 GB at batch 32, hence "nofetch".
Slow tier (E5, simulation): delay_ns_per_byte > 0 adds a GPU spin on the side stream before each fetch, lasting
(missed bytes of the layer) x delay_ns_per_byte, read on the GPU (no sync). The effective bandwidth is then
1 / (1/bw_link + delay) and is reported as simulated.
Keys and tie-breaking as kvpolicy.py: (prim, last, hit, -page).
"""
import time

import numpy as np
import torch
import triton
import triton.language as tl
from torch.nn import functional as F
from transformers import AttentionInterface

from gpu_bench import LenCache, _HostPtr

PS = 16
POLS = {"lru": 0, "ema": 1, "belady": 2, "nofetch": 0, "allmiss": 0}


@triton.jit
def _policy_b(sc_ptr, sc_stride, rk_ptr, pos_ptr, slot_ptr, slot_stride, last_ptr, hit_ptr, ema_ptr, nx_ptr,
              nxrow_ptr, pos2, last2, hit2, ema2, nx2, src_ptr, dst_ptr, old_ptr, cnt_ptr, t, alpha, pmax, B,
              POL: tl.constexpr, KK: tl.constexpr, BB: tl.constexpr):
    """gpu_bench._policy for request r = program_id(0); per-slot arrays are [R, B], slot tables [R, pmax+1]."""
    r = tl.program_id(0)
    sc_ptr += r * sc_stride
    rk_ptr += r * KK
    slot_ptr += r * slot_stride
    pos_ptr += r * B
    last_ptr += r * B
    hit_ptr += r * B
    ema_ptr += r * B
    nx_ptr += r * B
    pos2 += r * B
    last2 += r * B
    hit2 += r * B
    ema2 += r * B
    nx2 += r * B
    nxrow_ptr += r * KK
    src_ptr += r * KK
    dst_ptr += r * KK
    old_ptr += r * KK
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
    m2 = tgt[:, None] == i[None, :]
    touched = tl.sum(m2.to(tl.int32), 0) > 0
    old = tl.sum(tl.where(m2, pos[None, :], 0), 1)
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
    tl.atomic_add(cnt_ptr + r, tl.sum(miss.to(tl.int64), 0))


@triton.jit
def _slotup_b(slot_ptr, slot_stride, rk_ptr, dst_ptr, old_ptr, KK: tl.constexpr):
    r = tl.program_id(0)
    slot_ptr += r * slot_stride
    k = tl.arange(0, KK)
    tl.store(slot_ptr + tl.load(old_ptr + r * KK + k), -1)
    tl.store(slot_ptr + tl.load(rk_ptr + r * KK + k), tl.load(dst_ptr + r * KK + k).to(tl.int64))


@triton.jit
def _gather_b(ksrc, vsrc, kdst, vdst, page_ptr, slot_ptr, pmax, B, n_el: tl.constexpr, BLOCK: tl.constexpr,
              KK: tl.constexpr):
    """program (r*KK + k, j): if page_ptr[r, k] >= 0 copy host page r*pmax + page to slot r*B + slot."""
    i = tl.program_id(0)
    j = tl.program_id(1)
    r = i // KK
    p = tl.load(page_ptr + i)
    if p >= 0:
        d = (r * B + tl.load(slot_ptr + i)).to(tl.int64)
        s = (r * pmax + p).to(tl.int64)
        offs = j * BLOCK + tl.arange(0, BLOCK)
        tl.store(kdst + d * n_el + offs, tl.load(ksrc + s * n_el + offs))
        tl.store(vdst + d * n_el + offs, tl.load(vsrc + s * n_el + offs))


@triton.jit
def _store_b(ksrc, vsrc, kdst, vdst, row0, pmax, n_el: tl.constexpr, BLOCK: tl.constexpr):
    """Write-back of the newly completed page of request r (device [R, ...]) to host row r*pmax + row0."""
    r = tl.program_id(0)
    j = tl.program_id(1)
    offs = j * BLOCK + tl.arange(0, BLOCK)
    d = (r * pmax + row0).to(tl.int64)
    tl.store(kdst + d * n_el + offs, tl.load(ksrc + r.to(tl.int64) * n_el + offs))
    tl.store(vdst + d * n_el + offs, tl.load(vsrc + r.to(tl.int64) * n_el + offs))


@triton.jit
def _spin(cnt_ptr, prev_ptr, R, ns_per_miss):
    """Busy-wait for (new misses of this layer summed over requests) x ns_per_miss nanoseconds (slow-tier simulation).
    prev_ptr holds the counts seen at the previous call and is updated."""
    tot = tl.load(cnt_ptr) * 0
    for r in range(R):
        c = tl.load(cnt_ptr + r)
        tot += c - tl.load(prev_ptr + r)
        tl.store(prev_ptr + r, c)
    dummy = tl.full([], 0, tl.int32)
    t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64, is_pure=False,
                                   pack=1)
    target = t0 + (tot.to(tl.float64) * ns_per_miss).to(tl.int64)
    now = t0
    while now < target:
        now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64,
                                        is_pure=False, pack=1)


class BTier2:
    def __init__(s, R, kvh, dim, pmax, B, dtype, policy, alpha):
        dev = "cuda"
        s.R, s.kvh, s.dim, s.pmax, s.B, s.policy, s.alpha = R, kvh, dim, pmax, B, policy, alpha
        s.kh = torch.empty((R * pmax, kvh, PS, dim), dtype=dtype).pin_memory()      # host tier (pinned)
        s.vh = torch.empty_like(s.kh).pin_memory()
        s.lo = torch.zeros((R, kvh, pmax, dim), dtype=torch.float32, device=dev)
        s.hi = torch.zeros_like(s.lo)
        s.sk = torch.zeros((R * B, kvh, PS, dim), dtype=dtype, device=dev)   # zeros: "nofetch" reads them
        s.sv = torch.zeros_like(s.sk)
        z = lambda dt, v: torch.full((R, B), v, dtype=dt, device=dev)  # noqa: E731
        s.pos, s.pos2 = z(torch.long, -1), z(torch.long, -1)
        s.last, s.last2 = z(torch.long, -1), z(torch.long, -1)
        s.hit, s.hit2 = z(torch.long, 0), z(torch.long, 0)
        s.ema, s.ema2 = z(torch.float32, 0), z(torch.float32, 0)
        s.nx, s.nx2 = z(torch.long, 0), z(torch.long, 0)
        s.slot = torch.full((R, pmax + 1), -1, dtype=torch.long, device=dev)
        s.mgpu = torch.zeros(R, dtype=torch.long, device=dev)
        s.mprev = torch.zeros(R, dtype=torch.long, device=dev)
        s.np = 0
        s.tk = s.tv = None


class BatchBench2:
    def __init__(s, policy, budgets, alpha, K, nxt=None, timing=False, delay_ns_per_byte=0.0, record_scores=False,
                 overlap=False):
        s.policy, s.budgets, s.alpha, s.K, s.timing = policy, budgets, alpha, K, timing
        s.nxt = nxt
        s.tiers, s.step, s.rec, s.evs = {}, 0, None, []
        # overlap=False: fetch and write-back run on the main stream (no overlap). The side-stream version (overlap=True)
        # passed the CUDA tests but faulted nondeterministically ("illegal memory access") in long multi-run benchmarks and
        # not under CUDA_LAUNCH_BLOCKING=1, i.e. an unresolved race; part E uses overlap=False.
        s.side = torch.cuda.Stream() if overlap else torch.cuda.current_stream()
        s.delay = delay_ns_per_byte
        s.record_scores, s.scores = record_scores, None
        s.keep = []   # tensors read by the side stream; released after each step instead of record_stream

    def ev(s, stream=None):
        if not s.timing:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream)
        return e

    def attn(s, module, query, key, value, attention_mask, scaling, **kw):
        l = module.layer_idx
        T = s.tiers[l]
        R, kvh, dim, K, B, pmax = T.R, T.kvh, T.dim, s.K, T.B, T.pmax
        main = torch.cuda.current_stream()
        n_el = kvh * PS * dim
        e0 = s.ev()
        T.tk = torch.cat((T.tk, key), 2)
        T.tv = torch.cat((T.tv, value), 2)
        if T.tk.shape[2] > PS and T.policy == "nofetch":
            pf = T.tk[:, :, :PS].float()
            T.lo[:, :, T.np] = pf.amin(2)
            T.hi[:, :, T.np] = pf.amax(2)
            T.np += 1
            T.tk, T.tv = T.tk[:, :, PS:], T.tv[:, :, PS:]
        elif T.tk.shape[2] > PS:
            pk, pv = T.tk[:, :, :PS].contiguous(), T.tv[:, :, :PS].contiguous()
            pf = pk.float()
            T.lo[:, :, T.np] = pf.amin(2)
            T.hi[:, :, T.np] = pf.amax(2)
            wb = torch.cuda.Event()
            wb.record(main)
            s.side.wait_event(wb)
            with torch.cuda.stream(s.side):
                _store_b[(R, n_el // 1024)](pk, pv, _HostPtr(T.kh), _HostPtr(T.vh), T.np, pmax, n_el, 1024)
            s.keep += [pk, pv]       # alive until the step ends (the main stream waits for the side stream every layer)
            T.np += 1
            T.tk, T.tv = T.tk[:, :, PS:], T.tv[:, :, PS:]
        g = query.shape[1] // kvh
        n = T.np
        e1 = s.ev()
        q = query[:, :, 0].reshape(R, kvh, g, dim).float()
        sc = (torch.einsum("rhgd,rhpd->rhgp", q.clamp_min(0), T.hi[:, :, :n])
              + torch.einsum("rhgd,rhpd->rhgp", q.clamp_max(0), T.lo[:, :, :n])).mean((1, 2)).contiguous()
        ranked = sc.argsort(dim=1, descending=True, stable=True)[:, :K].contiguous()
        s.rec[l] = ranked
        if s.record_scores:
            s.scores[l] = sc
        e2 = s.ev()
        if T.policy == "allmiss":                 # E3: empty cache every step -> every selected page is fetched
            T.slot.fill_(-1)
            T.pos.fill_(-1)
        pol = POLS[T.policy]
        nxrow = s.nxt_gpu[s.step, :, l].contiguous() if pol == 2 else T.nx
        _policy_b[(R,)](sc, sc.stride(0), ranked, T.pos, T.slot, T.slot.stride(0), T.last, T.hit, T.ema, T.nx, nxrow,
                        T.pos2, T.last2, T.hit2, T.ema2, T.nx2, s.src, s.dst, s.old, T.mgpu,
                        s.step, float(T.alpha or 0.0), pmax, B, POL=pol, KK=K, BB=triton.next_power_of_2(B),
                        num_warps=8)
        _slotup_b[(R,)](T.slot, T.slot.stride(0), ranked, s.dst, s.old, KK=K)
        T.pos, T.pos2 = T.pos2, T.pos
        T.last, T.last2 = T.last2, T.last
        T.hit, T.hit2 = T.hit2, T.hit
        if pol == 1:
            T.ema, T.ema2 = T.ema2, T.ema
        if pol == 2:
            T.nx, T.nx2 = T.nx2, T.nx
        e3 = s.ev()
        ready = torch.cuda.Event()
        ready.record(main)
        s.side.wait_event(ready)
        src, dst = s.src.clone(), s.dst.clone()   # the next layer reuses s.src / s.dst on the main stream
        with torch.cuda.stream(s.side):
            es0 = s.ev(s.side)
            if s.delay > 0:
                _spin[(1,)](T.mgpu, T.mprev, R, s.delay * 2 * n_el * T.kh.element_size())
            es1 = s.ev(s.side)
            if T.policy != "nofetch":
                _gather_b[(R * K, n_el // 1024)](_HostPtr(T.kh), _HostPtr(T.vh), T.sk, T.sv, src, dst, pmax, B,
                                                 n_el, 1024, KK=K)
            es2 = s.ev(s.side)
            done = torch.cuda.Event()
            done.record(s.side)
        s.keep += [src, dst]
        # main stream, overlapping the fetch: attention index of the K selected pages (ascending page id)
        ar = torch.arange(R, device=query.device)
        srt = ranked.sort(dim=1).values
        ss = T.slot.gather(1, srt)
        flat = (ar[:, None] * B + ss).reshape(-1)
        e4 = s.ev()
        main.wait_event(done)
        kp_ = T.sk[flat].reshape(R, K, kvh, PS, dim).permute(0, 2, 1, 3, 4).reshape(R, kvh, K * PS, dim)
        vp_ = T.sv[flat].reshape(R, K, kvh, PS, dim).permute(0, 2, 1, 3, 4).reshape(R, kvh, K * PS, dim)
        k = torch.cat((kp_, T.tk), 2).repeat_interleave(g, 1)
        v = torch.cat((vp_, T.tv), 2).repeat_interleave(g, 1)
        out = F.scaled_dot_product_attention(query, k, v, scale=scaling)
        e5 = s.ev()
        if s.timing:
            s.evs.append((e0, e1, e2, e3, e4, e5, es0, es1, es2))
        return out.transpose(1, 2).contiguous(), None


@torch.inference_mode()
def run(model, prompts, policy, steps, budgets, alpha=0.0, K=64, nxt=None, warmup=8, timing=False,
        delay_ns_per_byte=0.0, record_scores=False, overlap=False):
    """Same interface and outputs as gpu_bench_batch.run, plus per-step miss counts (step_misses [steps]) and
    page-completion flags (page_step [steps]). Latency per step is wall time from the forward call to the token
    read-out (which synchronises)."""
    R = len(prompts)
    N = len(prompts[0])
    if any(len(p) != N for p in prompts):
        raise ValueError("equal prompt lengths required")
    cfg = model.config
    L = cfg.num_hidden_layers
    if isinstance(budgets, int):
        budgets = [budgets] * L
    b = BatchBench2(policy, budgets, alpha, K, nxt, timing, delay_ns_per_byte, record_scores, overlap)
    name = "bench_batch2"
    orig = cfg._attn_implementation
    pmax = (N + steps + 2) // PS + 1
    try:
        cfg._attn_implementation = "sdpa"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        first = []
        peak_fwd = peak_mig = 0
        for r, prompt in enumerate(prompts):
            out = model(input_ids=torch.tensor([prompt], device="cuda"), use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            # memory by phase (not timed): peak of this request's forward, then of its migration to the host tier
            peak_fwd = max(peak_fwd, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            first.append(out.logits[:, -1].argmax(-1).item())
            del out
            for l in range(L):
                kf, vf = cache.key_cache[l][0], cache.value_cache[l][0]
                kvh, _, dim = kf.shape
                if r == 0:
                    b.tiers[l] = BTier2(R, kvh, dim, pmax, budgets[l], kf.dtype, policy, alpha)
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
            peak_mig = max(peak_mig, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        peak_prefill = max(peak_fwd, peak_mig, torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
        b.src = torch.empty((R, K), dtype=torch.int32, device="cuda")
        b.dst = torch.empty((R, K), dtype=torch.int32, device="cuda")
        b.old = torch.empty((R, K), dtype=torch.long, device="cuda")
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
        lat, ranks, brk, scores, step_m, page_step = [], [], [], [], [], []
        prev = 0
        peak_warm = None
        for step in range(steps):
            b.step, b.rec, b.evs = step, [None] * L, []
            b.keep = []
            if record_scores:
                b.scores = [None] * L
            torch.cuda.synchronize()
            if step == warmup and warmup > 0:
                peak_warm = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            np_before = b.tiers[0].np
            t0 = time.perf_counter()
            out = model(input_ids=tokn, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            tokn = out.logits[:, -1].argmax(-1, keepdim=True)
            nxt_ids = tokn[:, 0].tolist()  # syncs the main stream, which waited for every fetch
            lat.append(time.perf_counter() - t0)
            for r in range(R):
                gen[r].append(nxt_ids[r])
            tot = int(sum(int(b.tiers[l].mgpu.sum()) for l in range(L)))
            step_m.append(tot - prev)
            prev = tot
            page_step.append(b.tiers[0].np > np_before)
            ranks.append(torch.stack(b.rec, 1).cpu().numpy())
            if record_scores:
                scores.append([sc.cpu().numpy() for sc in b.scores])
            if timing and b.evs:
                torch.cuda.synchronize()
                brk.append(np.array([[e[i].elapsed_time(e[i + 1]) for i in range(5)]
                                     + [e[6].elapsed_time(e[7]), e[7].elapsed_time(e[8])] for e in b.evs]).sum(0))
        torch.cuda.synchronize()
        peak_steady = torch.cuda.max_memory_allocated()
        misses = np.stack([b.tiers[l].mgpu.cpu().numpy() for l in range(L)], 1)
        T0 = b.tiers[0]
        page_bytes = 2 * T0.kvh * PS * T0.dim * T0.kh.element_size()
        return dict(lat=np.array(lat), ranks=np.array(ranks), gen=gen, misses=misses,
                    bytes=int(misses.sum()) * page_bytes, page_bytes=page_bytes,
                    step_misses=np.array(step_m), page_step=np.array(page_step),
                    breakdown=np.array(brk) if brk else None, scores=scores if record_scores else None,
                    warmup=warmup, peak_prefill=peak_prefill, peak_prefill_forward=peak_fwd, peak_migration=peak_mig,
                    mem_start=mem_start,
                    peak_decode_warmup=peak_warm, peak_decode_steady=peak_steady,
                    host_bytes=sum(2 * T.kh.numel() * T.kh.element_size() for T in b.tiers.values()))
    finally:
        cfg._attn_implementation = orig
        torch.cuda.synchronize()
        b.keep = []
        b.tiers.clear()
        torch.cuda.empty_cache()
