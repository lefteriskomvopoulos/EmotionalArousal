"""The balanced sample bal50 (docs/EXPERIMENTS.md Part 21, rule set before it ran).

Keeps the 44 of Part 20 (MPP converged among the first 50) and fills each group up to
neutral 20, negative 10, positive 10, mixed 10 from the eligible other-47 participants, by a seeded
draw within the group.  Eligible: MPP converged at 0.05 uS; a 300 s baseline and six 120 s clips;
mixed group: three negative and three positive clips.  Uses no film contrast and no outcome.

    python src/stage2/s2_sample_bal50.py        (from the repository root)
    -> data/stage2_sample_bal50.csv
       data/stage2_states_005_bal50.csv   the sample's rows of stage2_states_005.csv and
                                          stage2_states_005_extra47.csv (for s2_tests 'bal50')
"""
import csv
from collections import defaultdict

import numpy as np

SEED = 2611
ORDER = ["neutral", "negative", "positive", "mixed"]
TARGET = {"neutral": 20, "negative": 10, "positive": 10, "mixed": 10}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def design(states_path):
    """Per participant: baseline bins, clip bins by marker."""
    base, clips = defaultdict(int), defaultdict(lambda: defaultdict(int))
    for r in rows(states_path):
        m = r["marker"]
        if m in ("", "NaN"):
            continue
        m = int(float(m))
        if m == -1:
            base[r["id"]] += 1
        elif m >= 100:
            clips[r["id"]][m] += 1
    return base, clips


def main():
    p50 = [r for r in rows("data/stage2_params.csv") if r["min_height_uS"] == "0.05"]
    p47 = [r for r in rows("data/stage2_params_extra47.csv") if r["min_height_uS"] == "0.05"]
    base, clips = design("data/stage2_states_005_extra47.csv")

    keep = {g: sorted(int(r["id"]) for r in p50 if r["group"] == g and r["mpp_stop"] == "converged")
            for g in ORDER}
    pool, report = {g: [] for g in ORDER}, []
    for r in p47:
        i, g = r["id"], r["group"]
        c = clips[i]
        full = 295 <= base[i] <= 305 and len(c) == 6 and all(110 <= v <= 125 for v in c.values())
        mixed_ok = g != "mixed" or (sum(200 <= m < 300 for m in c) == 3 and sum(m >= 300 for m in c) == 3)
        ok = r["mpp_stop"] == "converged" and full and mixed_ok
        report.append((int(i), g, r["mpp_stop"], base[i], len(c), mixed_ok, ok))
        if ok:
            pool[g].append(int(i))

    avail = {g: len(keep[g]) + len(pool[g]) for g in ORDER}
    n = min(TARGET["negative"], avail["negative"], avail["positive"], avail["mixed"], avail["neutral"] // 2)
    target = {"neutral": 2 * n, "negative": n, "positive": n, "mixed": n}

    rng = np.random.default_rng(SEED)
    out = []
    for g in ORDER:
        base_ids = keep[g]
        need = target[g] - len(base_ids)
        if need >= 0:
            cand = sorted(pool[g])
            add = sorted(int(x) for x in rng.choice(cand, need, replace=False)) if need else []
            members = [(i, "first50") for i in base_ids] + [(i, "extra47") for i in add]
        else:                                   # the group already has more than the target
            drop = set(int(x) for x in rng.choice(base_ids, -need, replace=False))
            members = [(i, "first50") for i in base_ids if i not in drop]
        out += [(i, g, src) for i, g, src in [(i, g, s) for i, s in members]]
        print(f"{g:8s} kept {len(keep[g]):2d}, eligible extra {len(pool[g]):2d}, target {target[g]:2d}, "
              f"added {sorted(i for i, s in members if s == 'extra47')}")

    with open("data/stage2_sample_bal50.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "source"])
        for i, g, s in sorted(out, key=lambda x: (ORDER.index(x[1]), x[0])):
            w.writerow([i, g, s])
    print(f"bal50: {len(out)} participants, n = {n} -> data/stage2_sample_bal50.csv")

    ids = {str(i) for i, _, _ in out}
    rows_in, chosen = set(), set()
    with open("data/stage2_states_005_bal50.csv", "w", newline="") as f:
        w = None
        for src in ("data/stage2_states_005.csv", "data/stage2_states_005_extra47.csv"):
            with open(src, newline="") as g:
                rd = csv.reader(g)
                head = next(rd)
                if w is None:
                    w = csv.writer(f)
                    w.writerow(head)
                for row in rd:
                    if row[0] in ids:
                        w.writerow(row)
                        rows_in.add(row[0])
                        chosen.add(row[head.index("chosen")])
    assert rows_in == ids, ids - rows_in
    print(f"states of the {len(rows_in)} -> data/stage2_states_005_bal50.csv (chosen: {sorted(chosen)})")

    print("\nextra 47, eligibility (id, group, MPP stop, baseline bins, clips, mixed 3+3, eligible):")
    for t in sorted(report, key=lambda x: (ORDER.index(x[1]), x[0])):
        print("  ", t)


if __name__ == "__main__":
    main()
