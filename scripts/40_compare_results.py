"""Compare regenerated results with the reference results of the article (CPU only).

Every CSV table that exists both in results/tables/ (a new run) and in reference_results/tables/ (the run
reported in the article) is aligned on its key columns (text columns and integer identifiers such as batch,
repeat, k or seed; duplicate keys are reduced to their median) and its numeric columns are compared with a
tolerance that depends on what they measure:

  * accuracy and rates (mAP, AP, top-1, recall, precision, shares, AUC):        absolute 0.003 (0.3 points)
  * hardware-dependent quantities (latency, throughput, energy, power, clock,
    fleet energy, cost and CO2):                                                relative 10 %
  * everything else (deviations, SQNR, propagation factors, counts, ratios):   relative 5 % (absolute 0.05
                                                                                 for values near zero)

Accuracy deviations are errors (exit status 1). Hardware-dependent deviations are warnings: they reproduce
only on the same GPU, driver and TensorRT version. Usage:

    python scripts/40_compare_results.py [--results results/tables] [--reference reference_results/tables]
                                         [--tables PATTERN ...] [--verbose]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INT_KEYS = {"batch", "repeat", "k", "seed", "severity", "n_calib", "order", "rank", "layer_index"}
ACCURACY = re.compile(r"(^|_)(map|ap|ap50|top1|recall|precision|share|auc|agreement|disagreement|error_rate|"
                      r"vanish|rate)(_|$|\d)", re.I)
HARDWARE = re.compile(r"(ms|img_s|throughput|energy|power|clock|mhz|temp|kwh|mwh|eur|tco2|jitter|latency|"
                      r"idle|sustained|_w$|mem_mb|engine_mb|breakeven|inferences)", re.I)


def column_class(col):
    if HARDWARE.search(col):
        return "hardware"
    if ACCURACY.search(col):
        return "accuracy"
    return "other"


def keys_of(df):
    text = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    return text + [c for c in df.columns if c in INT_KEYS and pd.api.types.is_integer_dtype(df[c])]


def reduce(df, keys):
    if not keys:
        return df.reset_index(drop=True)
    num = [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])]
    return df.groupby(keys, dropna=False)[num].median().reset_index()


def compare(new_p, ref_p, verbose):
    ref, new = pd.read_csv(ref_p), pd.read_csv(new_p)
    keys = [k for k in keys_of(ref) if k in new.columns]
    ref, new = reduce(ref, keys), reduce(new, keys)
    merged = ref.merge(new, on=keys, how="outer", suffixes=("_ref", "_new"), indicator=True) if keys else \
        ref.join(new, lsuffix="_ref", rsuffix="_new").assign(_merge="both")
    both = merged[merged["_merge"] == "both"]
    report = {"table": ref_p.name, "rows_ref": len(ref), "rows_matched": len(both),
              "rows_only_ref": int((merged["_merge"] == "left_only").sum()),
              "rows_only_new": int((merged["_merge"] == "right_only").sum()), "errors": [], "warnings": []}
    for col in [c for c in ref.columns if c not in keys and pd.api.types.is_numeric_dtype(ref[c])]:
        if f"{col}_new" not in both.columns:
            continue
        a, b = both[f"{col}_ref"].astype(float), both[f"{col}_new"].astype(float)
        ok = a.notna() & b.notna()
        if not ok.any():
            continue
        a, b = a[ok], b[ok]
        kind = column_class(col)
        if kind == "accuracy":
            dev, bad = (b - a).abs(), (b - a).abs() > 0.003
        elif kind == "hardware":
            dev = (b - a).abs() / a.abs().clip(lower=1e-12)
            bad = dev > 0.10
        else:
            dev = (b - a).abs() / a.abs().clip(lower=1e-12)
            bad = (dev > 0.05) & ((b - a).abs() > 0.05)
        if bad.any():
            worst = dev[bad].idxmax()
            row = both.loc[worst, keys].to_dict() if keys else {"row": int(worst)}
            entry = (f"{col} [{kind}]: {int(bad.sum())}/{len(a)} rows beyond tolerance, worst "
                     f"{a[worst]:.4g} -> {b[worst]:.4g} at {row}")
            (report["errors"] if kind == "accuracy" else report["warnings"]).append(entry)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "tables")
    ap.add_argument("--reference", type=Path, default=ROOT / "reference_results" / "tables")
    ap.add_argument("--tables", nargs="*", help="glob patterns of table names (default: all)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    refs = sorted(args.reference.glob("*.csv"))
    if args.tables:
        refs = [p for p in refs if any(p.match(t) for t in args.tables)]
    n_err = n_warn = n_missing = 0
    for ref_p in refs:
        new_p = args.results / ref_p.name
        if not new_p.exists():
            n_missing += 1
            if args.verbose:
                print(f"--  {ref_p.name}: not regenerated")
            continue
        r = compare(new_p, ref_p, args.verbose)
        status = "FAIL" if r["errors"] else ("warn" if r["warnings"] else "ok")
        unmatched = f", {r['rows_only_ref']} reference rows unmatched" if r["rows_only_ref"] else ""
        print(f"{status:4s} {r['table']}: {r['rows_matched']}/{r['rows_ref']} rows compared{unmatched}")
        for e in r["errors"]:
            print(f"       error: {e}")
        for w in r["warnings"] if (args.verbose or not r["errors"]) else []:
            print(f"       warning: {w}")
        n_err += bool(r["errors"])
        n_warn += bool(r["warnings"])
    print(f"\n{len(refs) - n_missing} tables compared, {n_err} with accuracy deviations, {n_warn} with "
          f"hardware-dependent or other deviations, {n_missing} not regenerated.")
    sys.exit(1 if n_err else 0)
