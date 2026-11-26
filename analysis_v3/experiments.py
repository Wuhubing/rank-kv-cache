"""Experiment registry for analysis_v3.run (followup-v3, mechanism-v3; dev only).

Each experiment: configs {name: (policy, kwargs)}, families {family: [config names]} tuned leave-one-document-out,
budgets, contrasts [(a, b)] reported as paired G(a) - G(b) (a family label is "<family>(lodo)"), and optionally
signal="attn_all" (the policy reads the trace's dense attention mass instead of the Quest scores; the selection,
LRU and Belady are unchanged).
"""
import math

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]   # same grid as analysis_v2.s4_dev
BUDGETS = [80, 96, 128, 256]


class Exp:
    def __init__(self, name, budgets=BUDGETS, signal="scores", doc=""):
        self.name, self.budgets, self.signal, self.doc = name, list(budgets), signal, doc
        self.configs, self.families, self.contrasts = {}, {}, []

    def add(self, family, name, pol, **kw):
        if name in self.configs:
            raise ValueError(f"duplicate config {name}")
        self.configs[name] = (pol, kw)
        self.families.setdefault(family, []).append(name)

    def base(self, ema_signals=("z",)):
        for n in ("lru", "belady", "score"):
            self.add(n, n, n)
        for sig in ema_signals:
            for a in ALPHAS:
                self.add(f"ema_{sig}", f"ema_{sig}:{a}", "ema", alpha=a, signal=sig)
        return self


def b1():
    e = Exp("b1", doc="B1: kernel shape x normalisation x zero point; candidate explanations of the power-law kernel")
    e.base()
    for norm in (False, True):
        for zero in ("kth", "median", "min"):
            tag = f"{'norm' if norm else 'sum'}_{zero}"
            for a in (0.3, 0.5, 0.7, 0.9):
                e.add(f"exp_{tag}", f"exp_{tag}:{a}", "kernel", kernel="exp", a=a, normalize=norm, zero=zero)
            for g in (0.5, 1.0, 2.0):
                e.add(f"pow_{tag}", f"pow_{tag}:{g}", "kernel", kernel="power", gamma=g, normalize=norm, zero=zero)
    for clock in ("resid", "since_sel"):
        for a in (0.3, 0.5, 0.7):
            for lam in (0.05, 0.1, 0.2, 0.4, 0.8):
                e.add(f"ema_age_{clock}", f"ema_age_{clock}:{a}:{lam}", "ema_age", alpha=a, lam=lam, clock=clock)
    for a in (0.3, 0.5, 0.7):
        for lam in (0.01, 0.03, 0.1, 0.3, 1.0, math.inf):
            e.add("ema_deficit", f"ema_deficit:{a}:{lam}", "ema_deficit", alpha=a, lam=lam)
        for lam in (0.05, 0.1, 0.2, 0.4, 0.8):
            e.add("ema_unsel", f"ema_unsel:{a}:{lam}", "ema_unsel", alpha=a, lam=lam)
    for f in (0.0, 0.3, 0.5):
        for s in (0.8, 0.9, 0.95, 0.99):
            for w in (0.25, 0.5, 0.75):
                e.add("twoscale", f"twoscale:{f}:{s}:{w}", "twoscale", fast=f, slow=s, w=w)
    for w in (1, 4, 16, 64):
        e.add("hybrid", f"hybrid:{w}:0.5", "hybrid", w=w, alpha=0.5)
    cand = ["pow_sum_kth", "pow_norm_kth", "pow_sum_median", "pow_sum_min", "exp_sum_kth", "exp_norm_kth", "exp_sum_min",
            "pow_norm_median", "ema_age_resid", "ema_age_since_sel", "ema_deficit", "ema_unsel", "twoscale", "hybrid"]
    for c in cand:
        e.contrasts += [(f"{c}(lodo)", "ema_z(lodo)"), (f"{c}(lodo)", "pow_sum_kth(lodo)")]
    return e


def b3():
    e = Exp("b3", doc="B3: finite-lookahead Belady (horizon H; beyond it LRU or EMA order)")
    e.base()
    for H in (1, 4, 16, 64, 256, None):
        for tie in ("lru", "ema"):
            e.add(f"look_{tie}_H{H}", f"look_{tie}_H{H}", "lookahead", H=H, tie=tie, alpha=0.5)
    return e


def b4():
    e = Exp("b4", signal="attn_all", doc="B4: eviction signal = dense attention mass of every page (selection unchanged)")
    e.base(ema_signals=("z", "allrank", "raw"))
    return e


def b4_quest():
    e = Exp("b4_quest", doc="B4 reference: the same configurations on the Quest scores (same traces)")
    e.base(ema_signals=("z", "allrank", "raw"))
    return e


def b7():
    e = Exp("b7", doc="B7: update scope (selected + M top / bottom unselected residents) and rank reference set")
    e.base(ema_signals=("z", "allrank"))
    for sig in ("z", "allrank"):
        for a in (0.3, 0.5, 0.7):
            for M in (0, 4, 8, 16, 32):
                for which in ("top", "bottom"):
                    if M == 0 and which == "bottom":
                        continue
                    fam = f"scope_{sig}_{which if M else 'sel'}_M{M}"
                    e.add(fam, f"{fam}:{a}", "scoped_ema", alpha=a, M=M, which=which, signal=sig)
    for R in (64, 128, 256, 512, None):
        for a in (0.3, 0.5, 0.7):
            e.add(f"refrank_R{R}", f"refrank_R{R}:{a}", "refrank_ema", alpha=a, R=R)
    return e


def c3():
    e = Exp("c3", doc="C3: shrunken initial value of newly admitted pages (no noise; the noisy case is analysis_v3.noise)")
    e.base(ema_signals=("z", "allrank"))
    for sig in ("z", "allrank"):
        for prior in ("mean", "zero"):
            for a in (0.3, 0.5, 0.7):
                for c in (0.25, 0.5, 0.75):
                    e.add(f"shrink_{sig}_{prior}", f"shrink_{sig}_{prior}:{a}:{c}", "shrink_ema", alpha=a, c=c,
                          prior=prior, signal=sig)
        for prior in ("mean", "zero"):
            e.contrasts.append((f"shrink_{sig}_{prior}(lodo)", f"ema_{sig}(lodo)"))
    return e


def d5():
    e = Exp("d5", doc="D5: FIFO protection zone (rho) and pinned sink / recent pages on LRU and EMA")
    e.base()
    for base in ("lru", "ema"):
        for rho in (0.05, 0.1, 0.2, 0.3, 0.5):
            e.add(f"prot_{base}_rho{rho}", f"prot_{base}_rho{rho}", "protect", base=base, alpha=0.5, rho=rho)
        pins = [(n, 0) for n in range(1, 5)] + [(0, n) for n in range(1, 5)] + [(n, n) for n in range(1, 5)]
        for sk, rc in pins:
            e.add(f"pin_{base}_s{sk}r{rc}", f"pin_{base}_s{sk}r{rc}", "protect", base=base, alpha=0.5, sink=sk,
                  recent=rc)
    e.add("ema_fixed", "ema_fixed:0.5", "ema", alpha=0.5)
    return e


def lru_fine():
    e = Exp("lru_fine", budgets=list(range(64, 1025, 8)), doc="D1: LRU on a fine budget grid (iso-miss capacity)")
    for n in ("lru", "belady"):
        e.add(n, n, n)
    return e


def base():
    e = Exp("base", doc="LRU, Belady, current score, EMA of z and of the all-page percentile (G2 and other settings)")
    e.base(ema_signals=("z", "allrank"))
    return e


def g1():
    e = Exp("g1", budgets=[96, 128], doc="G1 (post hoc, exploratory): fixed configurations on the evaluated held-out sets")
    for n in ("lru", "belady"):
        e.add(n, n, n)
    for a in (0.5, 0.6, 0.7):
        e.add(f"ema_{a}", f"ema_z:{a}", "ema", alpha=a)
    e.add("c1", "pow_norm_kth:1.0", "kernel", kernel="power", gamma=1.0, normalize=True, zero="kth")
    return e


def d5b():
    e = Exp("d5b", budgets=[96, 128], doc="D5: protection zone on an H2O-style accumulated-share score (CO2's setting)")
    for n in ("lru", "belady"):
        e.add(n, n, n)
    for rho in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        e.add(f"prot_accum_rho{rho}", f"prot_accum_rho{rho}", "protect", base="accum", rho=rho)
    return e


REGISTRY = {f.__name__: f for f in (b1, b3, b4, b4_quest, b7, c3, d5, lru_fine, base, g1, d5b)}


def get(name):
    return REGISTRY[name]()
