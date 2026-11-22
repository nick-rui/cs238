"""Procedurally generated extreme orderings.

Each generated ordering is a *spec*: classes per biased block ``k`` (1, 2, 3
or 5), number of blocks ``n`` (1-10), a class mode (``repeat``: every block
uses the same k digits; ``change``: block i uses the next k digits of a cyclic
walk through the episode's digit permutation; ``alternate``: blocks alternate
between two disjoint k-digit sets), block length ``L`` (batches), gap ``G``
between consecutive blocks (mixed batches), and the length ``R`` of the
shuffled recovery after the last block (0 allowed). The start position of
the first block is ``200 - R - (n L + (n - 1) G)``.

Streams keep the exact invariants of ``run.stream``: the same seeded balanced
bag (1,280 draws per digit, 12,800 examples), digits permuted per episode, the
block labels spread across each block with ``np.resize``, the remaining bag
shuffled into the non-block positions, then a within-batch shuffle; 200
batches of 64. The random-number calls are made in the same order as
``run.stream``, so the seven original orderings expressed as specs reproduce
the original streams exactly (tested).

Importing this module patches ``b.stream``/``b.batches``/``e.batches`` (via
``run``) and adds a dispatcher so that ordering names registered with
``register`` use the generated streams, while the original names still work.
"""

import functools
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import run as extreme
import torch

b, e, f = extreme.b, extreme.e, extreme.f
BATCHES, BATCH, PER_CLASS, CLASSES = 200, 64, 1280, 10
KS = (1, 2, 3, 5)
MODES = ("repeat", "change", "alternate")
MAX_RECOVERY = 100
P_NO_RECOVERY = 0.2
P_NO_GAP = 0.5


@dataclass(frozen=True)
class Spec:
    k: int
    n: int
    mode: str  # "single" when n == 1
    L: int
    G: int
    R: int

    @property
    def span(self):
        return self.n * self.L + (self.n - 1) * self.G

    @property
    def start(self):
        return BATCHES - self.R - self.span

    def block_starts(self):
        return [self.start + i * (self.L + self.G) for i in range(self.n)]

    def block_slots(self, i):
        """Indices into the episode digit permutation used by block i."""
        if self.mode in ("single", "repeat"):
            return list(range(self.k))
        if self.mode == "change":
            return [(i * self.k + j) % CLASSES for j in range(self.k)]
        if self.mode == "alternate":
            off = 0 if i % 2 == 0 else self.k
            return [off + j for j in range(self.k)]
        raise ValueError(self.mode)


ORIGINAL_SPECS = {
    "single_max": Spec(1, 1, "single", 20, 0, 100),
    "single_late": Spec(1, 1, "single", 20, 0, 5),
    "pair_max": Spec(2, 1, "single", 40, 0, 20),
    "half_block": Spec(5, 1, "single", 100, 0, 40),
    "staircase": Spec(1, 10, "change", 16, 0, 20),
    "class_sorted": Spec(1, 10, "change", 20, 0, 0),
    "alternating_halves": Spec(5, 8, "alternate", 25, 0, 0),
}


def slot_usage(spec):
    """Draws taken from each permutation slot (0..9) by the biased blocks."""
    use = np.zeros(CLASSES, dtype=np.int64)
    for i in range(spec.n):
        slots = spec.block_slots(i)
        labels = np.resize(np.array(slots), spec.L * BATCH)
        use += np.bincount(labels, minlength=CLASSES)
    return use


def feasible(spec):
    return (
        spec.start >= 0
        and spec.L >= 2
        and spec.G >= 0
        and 0 <= spec.R <= MAX_RECOVERY
        and (slot_usage(spec) <= PER_CLASS).all()
    )


def sample_spec(generator_seed):
    """Hierarchical sampler (prespecified; see results/extreme_generated/design.json).

    k ~ U{1,2,3,5}; n ~ U{1..10}; mode ~ U{repeat, change, alternate} for n > 1;
    R = 0 w.p. 0.2 else U{1..100}; L ~ U{2..Lmax} with Lmax the largest
    feasible block length given (k, n, mode, R) and the minimum gap (1 for
    repeat, else 0); G = 0 for n = 1; for repeat G ~ U{1..Gmax}; otherwise G = 0
    w.p. 0.5 else U{1..Gmax} (G = 0 if Gmax < 1), Gmax = floor((200 - R - nL)/(n-1)).
    """
    rng = np.random.default_rng(generator_seed)
    k = int(rng.choice(KS))
    n = int(rng.integers(1, 11))
    mode = "single" if n == 1 else str(rng.choice(MODES))
    R = 0 if rng.random() < P_NO_RECOVERY else int(rng.integers(1, MAX_RECOVERY + 1))
    min_gap = 1 if mode == "repeat" else 0
    Lmax = 1
    while feasible(Spec(k, n, mode, Lmax + 1, min_gap, R)):
        Lmax += 1
    assert Lmax >= 2, (generator_seed, k, n, mode, R)
    L = int(rng.integers(max(2, math.ceil(Lmax / 2)), Lmax + 1))
    if n == 1:
        G = 0
    else:
        gmax = (BATCHES - R - n * L) // (n - 1)
        if mode == "repeat":
            G = int(rng.integers(1, gmax + 1))
        elif rng.random() < P_NO_GAP or gmax < 1:
            G = 0
        else:
            G = int(rng.integers(1, gmax + 1))
    spec = Spec(k, n, mode, L, G, R)
    assert feasible(spec), spec
    return spec


# ------------------------------------------------------------------ streams

REGISTRY = {}


def register(name, spec):
    assert feasible(spec), spec
    if name in REGISTRY:
        assert REGISTRY[name] == spec, name
    REGISTRY[name] = spec


def events_for(spec, digits):
    return [
        dict(
            start=s,
            end=s + spec.L,
            digits=[digits[j] for j in spec.block_slots(i)],
            profile="pure",
        )
        for i, s in enumerate(spec.block_starts())
    ]


@functools.lru_cache(maxsize=4096)
def spec_stream(seed, spec):
    rng = np.random.default_rng(seed + 700000)
    bags = [rng.choice(q, PER_CLASS, replace=True) for q in b.class_ids]
    digits = rng.permutation(CLASSES).tolist()
    events = events_for(spec, digits)
    ids = np.full((BATCHES, BATCH), -1, dtype=np.int64)
    mask = np.zeros(BATCHES, dtype=bool)
    positions = [0] * CLASSES
    for ev in events:
        start, end = ev["start"], ev["end"]
        assert not mask[start:end].any()
        mask[start:end] = True
        labels = np.resize(np.array(ev["digits"]), (end - start) * BATCH)
        block = ids[start:end].reshape(-1)
        for d in ev["digits"]:
            where = np.flatnonzero(labels == d)
            lo = positions[d]
            hi = lo + len(where)
            assert hi <= len(bags[d]), (spec, d, hi)
            block[where] = bags[d][lo:hi]
            positions[d] = hi
    remaining = np.concatenate([q[pos:] for q, pos in zip(bags, positions)])
    rng.shuffle(remaining)
    ids[ids < 0] = remaining
    for row in ids:
        rng.shuffle(row)
    return torch.from_numpy(ids), events, mask


_ORIGINAL_STREAM = extreme.stream


def stream(seed, order):
    if order in REGISTRY:
        return spec_stream(int(seed), REGISTRY[order])
    return _ORIGINAL_STREAM(seed, order)


def batches(seed, order):
    return stream(int(seed), order)[0]


b.stream = stream
b.batches = batches
e.batches = batches


# ------------------------------------------------------------------ severity / nearness

def expected_class_profile(spec):
    """(200, 10) expected class frequencies per batch, in permutation-slot space."""
    use = slot_usage(spec)
    left = PER_CLASS - use
    free = BATCHES * BATCH - use.sum()
    prof = np.tile(left / free if free else np.zeros(CLASSES), (BATCHES, 1))
    for i, s in enumerate(spec.block_starts()):
        row = np.zeros(CLASSES)
        row[spec.block_slots(i)] = 1.0 / spec.k
        prof[s:s + spec.L] = row
    return prof


def kl_uniform(p):
    p = np.asarray(p, dtype=float)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] * CLASSES)).sum())


def severity(spec):
    prof = expected_class_profile(spec)
    kl = [kl_uniform(row) for row in prof]
    return dict(
        mean_kl=float(np.mean(kl)),
        biased_batches=spec.n * spec.L,
        biased_batch_kl=float(spec.n * spec.L * math.log(CLASSES / spec.k) / BATCHES),
    )


def profile_distance(a, c):
    """Mean over batches of the total-variation distance between expected class
    profiles. Profiles are in permutation-slot space (slot j = the episode's
    j-th permuted digit), so both orderings use the same canonical labels."""
    pa, pc = expected_class_profile(a), expected_class_profile(c)
    return float(0.5 * np.abs(pa - pc).sum(1).mean())


def near_duplicate(spec, orig, tol=10):
    """Prespecified rule: same k, same n, same mode (any mode when n == 1), and
    block length, start and recovery each within ``tol`` batches."""
    return (
        spec.k == orig.k
        and spec.n == orig.n
        and spec.mode == orig.mode
        and abs(spec.L - orig.L) <= tol
        and abs(spec.start - orig.start) <= tol
        and abs(spec.R - orig.R) <= tol
    )


def nearness(spec):
    dists = {name: profile_distance(spec, o) for name, o in ORIGINAL_SPECS.items()}
    nearest = min(dists, key=dists.get)
    dups = [name for name, o in ORIGINAL_SPECS.items() if near_duplicate(spec, o)]
    return dict(nearest=nearest, nearest_distance=dists[nearest], near_duplicate_of=";".join(dups))


def spec_dict(spec):
    return dict(asdict(spec), start=spec.start)


if __name__ == "__main__":
    for name, s in ORIGINAL_SPECS.items():
        print(name, json.dumps(spec_dict(s)), json.dumps(severity(s)))
