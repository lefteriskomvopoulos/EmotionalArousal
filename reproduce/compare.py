"""
Compare a reproduction with the shipped results.  Called by reproduce.sh (REPRODUCE.md).

    python reproduce/compare.py SHIPPED WORK

SHIPPED is the delivered folder; WORK is the scratch copy the commands ran in.
WORK/logs/manifest.tsv lists every step: its name, its status (OK, FAILED, or SKIPPED: why) and
the files it writes (globs relative to the folder; "a=b" compares WORK/a with SHIPPED/b).  A file
counts as recomputed only if its step rewrote it after WORK/.reproduce_start was made.

Statuses
    IDENTICAL        the same bytes
    MATCH            the same numbers to a relative 1e-6 (the last printed digits)
    CLOSE            the same to 1e-3 but not to 1e-6: rounding on another machine or MATLAB
                     version.  Listed, but not counted as a failure.
    PARTIAL          as MATCH, except cells left blank because that run's per-sample signals are
                     not shipped (README, Reproducing).  Allowed only for the tables in PARTIAL_OK.
    REGENERATED      a figure: rewritten, not compared pixel by pixel
    NEW              written, but not shipped, so there is nothing to compare it with
    FAIL, NOT REGENERATED, STEP FAILED   counted as failures
    SKIPPED          the step did not run (no MATLAB, no raw files, or --fast)

Also checks that every shipped solver input is the signal the solver ran on (its _signals.csv),
and lists the shipped files no step recomputes.  Writes SHIPPED/reproduce_report.csv.  Exit
status 1 if anything failed.
"""
import csv
import glob
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-6, 1e-9
CLOSE_RTOL = 1e-3
IGNORE_COLS = {"seconds"}                   # the solver's run time per window
PARTIAL_OK = {"data/own_tune_scores.csv", "data/own_tune_scores_selected.csv",
              "data/own_final_test_scores.csv", "data/round8_compare.csv"}
NEEDS_RAW = "raw POPANE files not present"  # s2_tests' H6 rows when data/raw/Study6 is absent
BLANK = ["", "nan", "NaN", "NA"]
FIGURES = {".png", ".pdf"}
TAG = "own4_p1000_k14_f3_s8"
SOLVER_RECORD = {"P": f"hpc/own/final/output/{TAG}", "V": f"hpc/own/final/output/{TAG}",
                 "R": f"hpc/own/resynth/output/{TAG}"}
FAILED = {"FAIL", "NOT REGENERATED", "STEP FAILED"}


def close(a, b, rtol):
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore"):
        return both_nan | (np.abs(a - b) <= ATOL + rtol * np.maximum(np.abs(a), np.abs(b)))


def max_rel(a, b):
    m = ~(np.isnan(a) | np.isnan(b)) & np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return 0.0
    d = np.abs(a[m] - b[m]) / np.maximum(np.maximum(np.abs(a[m]), np.abs(b[m])), ATOL)
    return float(d.max())


def compare_csv(shipped, new, rel):
    a = pd.read_csv(shipped, dtype=str, keep_default_na=False)
    b = pd.read_csv(new, dtype=str, keep_default_na=False)
    if list(a.columns) != list(b.columns):
        return "FAIL", f"columns differ: shipped {list(a.columns)}, new {list(b.columns)}"
    if len(a) != len(b):
        return "FAIL", f"{len(a)} rows shipped, {len(b)} recomputed"
    raw_rows = np.zeros(len(b), bool)
    for c in b.columns:
        raw_rows |= b[c].str.contains(NEEDS_RAW, regex=False).to_numpy()
    worst, n_close, n_blank, first_bad = 0.0, 0, 0, None
    for c in a.columns:
        if c in IGNORE_COLS:
            continue
        sa, sb = a[c].to_numpy(str), b[c].to_numpy(str)
        diff = (sa != sb) & ~raw_rows
        if not diff.any():
            continue
        fa = pd.to_numeric(a[c], errors="coerce").to_numpy(float)
        fb = pd.to_numeric(b[c], errors="coerce").to_numpy(float)
        blank_a, blank_b = np.isin(sa, BLANK), np.isin(sb, BLANK)
        numeric = ~np.isnan(fa) & ~np.isnan(fb)
        tight = numeric & close(fa, fb, RTOL)
        loose = numeric & close(fa, fb, CLOSE_RTOL)
        blanked = blank_b & ~blank_a
        bad = diff & ~loose & ~(blank_a & blank_b) & ~blanked
        n_close += int((diff & loose & ~tight).sum())
        n_blank += int((diff & blanked).sum())
        m = numeric & diff
        if m.any():
            worst = max(worst, max_rel(fa[m], fb[m]))
        if bad.any() and first_bad is None:
            i = int(np.flatnonzero(bad)[0])
            first_bad = (f"{int(bad.sum())} cells differ in column {c}; first at line {i + 2}: "
                         f"shipped {sa[i]!r}, new {sb[i]!r}")
    note = f"; {int(raw_rows.sum())} rows need the raw files (not tested)" if raw_rows.any() else ""
    if first_bad:
        return "FAIL", first_bad + note
    if n_blank and rel not in PARTIAL_OK:
        return "FAIL", f"{n_blank} cells now blank" + note
    status = "PARTIAL" if n_blank else ("CLOSE" if n_close else "MATCH")
    detail = f"max relative difference {worst:.1e}"
    if n_close:
        detail += f", {n_close} cells beyond 1e-6"
    if n_blank:
        detail += f", {n_blank} cells blank (signals not shipped)"
    return status, detail + note


def load_arrays(path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    from scipy.io import loadmat
    return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}


def compare_arrays(shipped, new):
    a, b = load_arrays(shipped), load_arrays(new)
    if sorted(a) != sorted(b):
        return "FAIL", f"keys differ: shipped {sorted(a)}, new {sorted(b)}"
    worst, loose_only = 0.0, False
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            return "FAIL", f"{k}: shape {x.shape} shipped, {y.shape} new"
        if np.array_equal(x, y):
            continue
        if x.dtype.kind not in "biuf" or y.dtype.kind not in "biuf":
            return "FAIL", f"{k} differs"
        x, y = x.astype(float).ravel(), y.astype(float).ravel()
        if not close(x, y, CLOSE_RTOL).all():
            return "FAIL", f"{k}: max relative difference {max_rel(x, y):.1e}"
        loose_only |= not close(x, y, RTOL).all()
        worst = max(worst, max_rel(x, y))
    return ("CLOSE" if loose_only else "MATCH"), f"max relative difference {worst:.1e}"


def compare_file(shipped, new, rel):
    if shipped.stat().st_size == new.stat().st_size and shipped.read_bytes() == new.read_bytes():
        return "IDENTICAL", ""
    if shipped.suffix in FIGURES:
        return "REGENERATED", "figure, not compared"
    if shipped.suffix == ".csv":
        return compare_csv(shipped, new, rel)
    if shipped.suffix in {".npz", ".mat"}:
        return compare_arrays(shipped, new)
    return "FAIL", "differs"


def expand(root, pattern):
    return [Path(p).relative_to(root).as_posix() for p in glob.glob(str(root / pattern))]


def check_steps(shipped, work):
    stamp = (work / ".reproduce_start").stat().st_mtime
    rows = OrderedDict()                                   # shipped path -> row; a later step wins
    with open(work / "logs" / "manifest.tsv") as f:
        steps = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    for name, status, outs in steps:
        for tok in outs.split():
            if "=" in tok:
                pairs = [tuple(tok.split("=", 1))]
            else:
                pairs = [(r, r) for r in sorted(set(expand(work, tok)) | set(expand(shipped, tok)))]
                if not pairs:
                    rows[tok] = (name, tok, "FAIL" if status == "OK" else "SKIPPED", "no such file")
                    continue
            for n_rel, s_rel in pairs:
                rows.pop(s_rel, None)
                if status.startswith("SKIPPED"):
                    rows[s_rel] = (name, s_rel, "SKIPPED", status.split(":", 1)[-1].strip())
                    continue
                if status != "OK":
                    rows[s_rel] = (name, s_rel, "STEP FAILED", f"see logs/{name}.log in the work folder")
                    continue
                nf, sf = work / n_rel, shipped / s_rel
                if not nf.exists() or nf.stat().st_mtime < stamp:
                    rows[s_rel] = (name, s_rel, "NOT REGENERATED", "the step ran but did not rewrite it")
                elif not sf.exists():
                    rows[s_rel] = (name, s_rel, "NEW", "not shipped, nothing to compare")
                else:
                    try:
                        st, det = compare_file(sf, nf, s_rel)
                    except Exception as e:                       # report, never stop the check
                        st, det = "FAIL", f"could not compare: {e}"
                    rows[s_rel] = (name, s_rel, st, det)
    return list(rows.values())


def check_inputs(shipped):
    """Every shipped solver input against the signal the solver recorded it ran on."""
    rows = []
    for f in sorted((shipped / "data" / "tbme_input").glob("*_own4.mat")):
        rel = f"data/tbme_input/{f.name}"
        rec = SOLVER_RECORD.get(f.name[0])
        sig = shipped / rec / f"{f.stem}_signals.csv" if rec else None
        if sig is None or not sig.exists():
            rows.append(("solver inputs", rel, "NO RECORD", "no per-sample solver record shipped"))
            continue
        m = load_arrays(f)
        y, t = np.asarray(m["y"], float).ravel(), np.asarray(m["t"], float).ravel()
        s = pd.read_csv(sig, usecols=["time_s", "y"])
        ys, ts = s.y.to_numpy(float), s.time_s.to_numpy(float)
        how = ""
        if len(ys) != len(y):
            _, ia, ib = np.intersect1d(np.round(t, 3), np.round(ts, 3), return_indices=True)
            how = f" (aligned on time: {len(ia)} of {len(y)} samples)"
            ok = len(ib) == len(ts) and close(y[ia], ys[ib], 1e-9).all()
        else:
            ok = close(y, ys, 1e-9).all()
        rows.append(("solver inputs", rel, "MATCH" if ok else "FAIL",
                     ("the signal the solver ran on" if ok else "differs from the solver's record") + how))
    return rows


def not_covered(shipped, rows):
    done = {r[1] for r in rows}
    out = Counter()
    for pattern in ("data/*.csv", "data/tbme_synth/*.npz", "data/tbme_input/*.mat", "results/figures/*.*"):
        for p in expand(shipped, pattern):
            if p not in done:
                out[str(Path(p).parent)] += 1
    return out


def main(shipped, work):
    shipped, work = Path(shipped).resolve(), Path(work).resolve()
    rows = check_steps(shipped, work) + check_inputs(shipped)
    with open(shipped / "reproduce_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "file", "status", "detail"])
        w.writerows(rows)

    counts = Counter(r[2] for r in rows)
    print("\n=== Comparison with the shipped results ===")
    for step in OrderedDict((r[0], None) for r in rows):
        c = Counter(r[2] for r in rows if r[0] == step)
        print(f"  {step:32s} " + ", ".join(f"{n} {s}" for s, n in c.most_common()))
    problems = [r for r in rows if r[2] in FAILED]
    closes = [r for r in rows if r[2] == "CLOSE"]
    for title, group in (("Problems", problems), ("Close but not identical (rounding)", closes)):
        if group:
            print(f"\n{title}:")
            for r in group:
                print(f"  [{r[2]}] {r[1]} ({r[0]}): {r[3]}")
    skipped = [r for r in rows if r[2] == "SKIPPED"]
    if skipped:
        print("\nNot checked in this run (SKIPPED):")
        for why, n in Counter(r[3] for r in skipped).most_common():
            print(f"  {n} files: {why}")
    nc = not_covered(shipped, rows)
    if nc:
        print("\nShipped files that no step in this mode recomputes (REPRODUCE.md, What is not rerun):")
        for d, n in sorted(nc.items()):
            print(f"  {d}: {n}")
    print("\nTotals: " + ", ".join(f"{n} {s}" for s, n in counts.most_common()))
    print(f"Full table: {shipped / 'reproduce_report.csv'}")
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s), listed above.")
        return 1
    print("\nRESULT: every recomputed result matches the shipped one.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
