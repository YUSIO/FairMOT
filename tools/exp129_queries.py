"""Exp129: slot queries generated from ground-truth trajectories (no tracker output is read).

A query row is [tx1, ty1, tx2, ty2, px, py, s0, age, gx1, gy1, gx2, gy2, gvalid]: the template box at frame t - age, the
predicted centre in frame t, the template's long side, the age, and the member's ground-truth box at t (gvalid = 1 when
the member has a GT row at t). Labels are geometric and computed from the row: the member is "inside" when its centre is
within INSIDE sides of the predicted centre.
"""
import configparser
from pathlib import Path

import numpy as np

P_AGE = 0.15
MAX_AGE = 30
MIX = np.array([.5, .0, .25, .25])        # regular items: present, absent slot, on a neighbour, background
MIX_ABSENT = np.array([.0, 1., .0, .0])   # absent-event items
P_ABSENT_ITEM = 0.2                       # share of items built on an absence event (so 20% of all queries are absent slots)
EVENT_JITTER = 2
NOISE_SIDES = 0.3
NOISE_CAP_SIDES = 2.0
EXTRAPOLATION_CAP_SIDES = 3.0
BACKGROUND_CLEARANCE_SIDES = 4.0
N_FIELDS = 13


class SeqGT:
    def __init__(self, seq_dir):
        seq_dir = Path(seq_dir)
        rows = np.loadtxt(seq_dir / 'gt' / 'gt.txt', delimiter=',', ndmin=2)
        rows = rows[rows[:, 6] >= 1]
        self.frames = {}
        for f in np.unique(rows[:, 0]).astype(int):
            r = rows[rows[:, 0] == f]
            self.frames[int(f)] = (r[:, 1].astype(int), np.c_[r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]])
        info = configparser.ConfigParser()
        info.read(seq_dir / 'seqinfo.ini')
        self.n_frames = int(info['Sequence']['seqLength'])
        self.width, self.height = int(info['Sequence']['imWidth']), int(info['Sequence']['imHeight'])
        self.dir = seq_dir

    def get(self, f):
        return self.frames.get(f, (np.zeros(0, int), np.zeros((0, 4))))

    def absence_events(self):
        """(member, frame t, last frame with a GT row) for every frame t where the member has no row but had one <= MAX_AGE before."""
        if not hasattr(self, '_events'):
            present = {}
            for f, (ids, _) in self.frames.items():
                for g in ids:
                    present.setdefault(int(g), set()).add(f)
            ev = []
            for g, fs in present.items():
                for t in range(min(fs) + 1, self.n_frames + 1):
                    if t in fs:
                        continue
                    prev = [f for f in range(max(1, t - MAX_AGE), t) if f in fs]
                    if prev:
                        ev.append((g, t, prev[-1]))
            self._events = ev
        return self._events


def center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def side(b):
    return max(b[2] - b[0], b[3] - b[1])


def draw_age(rng):
    return int(min(rng.geometric(P_AGE), MAX_AGE))


def extrapolated(seq, gid, f_a, box_a, age):
    c, s = center(box_a), side(box_a)
    ids_p, box_p = seq.get(f_a - 1)
    k = np.where(ids_p == gid)[0]
    if len(k) == 0:
        return c
    disp = (c - center(box_p[k[0]])) * age
    cap = EXTRAPOLATION_CAP_SIDES * s
    n = np.linalg.norm(disp)
    return c + (disp * cap / n if n > cap else disp)


def sample_queries(seq, t, age, rng, n=8, mix=MIX):
    """n query rows for target frame t and template frame t - age, or None when the template frame has no member."""
    f_a = t - age
    ids_a, box_a = seq.get(f_a)
    ids_t, box_t = seq.get(t)
    if f_a < 1 or len(ids_a) == 0:
        return None
    in_t = set(int(x) for x in ids_t)
    rows = []
    for _ in range(n):
        cat = int(rng.choice(4, p=mix))
        both = [k for k, g in enumerate(ids_a) if int(g) in in_t]
        gone = [k for k, g in enumerate(ids_a) if int(g) not in in_t]
        if cat == 0 and not both:
            cat = 3
        if cat == 1 and not gone:
            cat = 0 if both else 3
        if cat == 2 and (len(ids_t) == 0 or (len(ids_t) == 1 and len(ids_a) == 1 and int(ids_t[0]) == int(ids_a[0]))):
            cat = 0 if both else 3
        k = int(rng.choice(both if cat == 0 else gone if cat == 1 else np.arange(len(ids_a))))
        gid, tb = int(ids_a[k]), box_a[k]
        s0 = side(tb)
        if cat in (0, 1):
            sigma = min(NOISE_SIDES * s0 * np.sqrt(age), NOISE_CAP_SIDES * s0)
            pred = extrapolated(seq, gid, f_a, tb, age) + rng.normal(0, sigma, 2)
        elif cat == 2:
            others = [m for m in range(len(ids_t)) if int(ids_t[m]) != gid] or list(range(len(ids_t)))
            m = int(rng.choice(others))
            pred = center(box_t[m]) + rng.normal(0, NOISE_SIDES * s0, 2)
        else:
            pred = None
            for _try in range(10):
                cand = np.array([rng.uniform(0, seq.width), rng.uniform(0, seq.height)])
                pred = cand
                if len(box_t) == 0 or np.linalg.norm(np.c_[(box_t[:, 0] + box_t[:, 2]) / 2, (box_t[:, 1] + box_t[:, 3]) / 2] - cand, axis=1).min() >= BACKGROUND_CLEARANCE_SIDES * s0:
                    break
        j = np.where(ids_t == gid)[0]
        gb, valid = (box_t[j[0]], 1.0) if len(j) else (np.zeros(4), 0.0)
        rows.append(np.r_[tb, pred, s0, age, gb, valid])
    return np.asarray(rows, np.float32)


def absent_item(seq, rng):
    """(t, age) of an absence event: the template frame is the member's last GT frame, moved back by up to EVENT_JITTER frames."""
    ev = seq.absence_events()
    if not ev:
        return None
    g, t, f_last = ev[int(rng.integers(len(ev)))]
    f_a = f_last - int(rng.integers(0, EVENT_JITTER + 1))
    ids, _ = seq.get(f_a)
    if f_a < 1 or g not in set(int(x) for x in ids) or t - f_a > MAX_AGE:
        f_a = f_last
    return t, t - f_a
