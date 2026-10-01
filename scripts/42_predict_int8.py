"""Predict the INT8 tolerance of a network from its quantizer table (Section 3.8 of the article).

Inputs are the per-quantizer tables of 26_quantizer_snr.py (FP32 activations and calibrated scales only; no
quantized model is executed) and, optionally, one measured head-input SQNR of the quantized engine (06_activations).
The relation between head-input SQNR and relative accuracy loss is calibrated on the five networks of the article
(tables/prediction.csv, written by 13_tables.py); a network that belongs to the calibration set is left out of the fit
with --leave-out, which reproduces the leave-one-out check of Table tab:prediction.

  python scripts/42_predict_int8.py --model yolov10x                      # FP32 statistics only (screening)
  python scripts/42_predict_int8.py --model yolov10x --head-sqnr 4.25     # after one quantized measurement
  python scripts/42_predict_int8.py --model yolov10x --leave-out          # exclude the model from the calibration

Steps (Table tab:formulas):
  1  injected noise of every quantizer, Lemma 1: INT8 52.9 dB - 20 log10(kappa), FP8 31.5 dB
  2  SQNR_add = -10 log10 sum_n rho_n^2 over the quantizers upstream of the head (unit propagation factors)
  3  with a measured SQNR_h: Gamma_bar = 10^((SQNR_add - SQNR_h) / 10)
  4  relative loss = 10^(a + b * SQNR) from the log-linear calibration (SQNR_add or SQNR_h)
  5  recommendation: INT8 if its predicted relative loss is at most --max-loss, otherwise FP8 or FP16
"""
import argparse

import numpy as np
import pandas as pd

from pepai.config import CODE_ROOT, load_config, results_dir


def sqnr_add(rows):
    """Head-input SQNR of Eq. (gammabar) with unit propagation factors, from the exact injected SQNR per quantizer."""
    return float(-10 * np.log10(np.sum(10 ** (-rows.sqnr_inj_db.values / 10))))


def calibration(pred, column, leave_out=None):
    """Least-squares line log10(relative loss) = a + b * SQNR over the calibration networks."""
    d = pred if leave_out is None else pred[pred.model != leave_out]
    b, a = np.polyfit(d[column].values, np.log10(d.int8_rel_loss.values), 1)
    return a, b, len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name as in the quantizer tables")
    ap.add_argument("--head-sqnr", type=float, help="measured head-input SQNR of the INT8 engine (dB)")
    ap.add_argument("--max-loss", type=float, default=1.0, help="acceptable relative accuracy loss (%%)")
    ap.add_argument("--leave-out", action="store_true", help="exclude --model from the calibration networks")
    ap.add_argument("--tables", help="directory with quantizer_snr*.csv (default: results/tables, else "
                                     "reference_results/tables)")
    args = ap.parse_args()

    cfg = load_config()
    t = results_dir(cfg, "tables")
    if args.tables:
        t = CODE_ROOT / args.tables if not args.tables.startswith("/") else args.tables
    elif not (t / "quantizer_snr.csv").exists():
        t = CODE_ROOT / "reference_results" / "tables"
    q = pd.read_csv(f"{t}/quantizer_snr.csv")
    q8_path = f"{t}/quantizer_snr_fp8.csv"
    q8 = pd.read_csv(q8_path) if pd.io.common.file_exists(q8_path) else None
    pred = pd.read_csv(f"{t}/prediction.csv") if pd.io.common.file_exists(f"{t}/prediction.csv") else \
        pd.read_csv(CODE_ROOT / "reference_results" / "tables" / "prediction.csv")

    up = q[(q.model == args.model) & q.upstream_of_head.astype(bool)]
    if up.empty:
        raise SystemExit(f"no quantizers of {args.model} in {t}/quantizer_snr.csv (run 26_quantizer_snr.py first)")
    leave = args.model if args.leave_out else None
    if leave is None and args.model in set(pred.model):
        print(f"note: {args.model} is one of the calibration networks; use --leave-out for an honest prediction")

    out = {"model": args.model, "quantizers": len(up),
           "int8_pred_median_db": up.sqnr_pred_db.median(), "int8_exact_median_db": up.sqnr_inj_db.median(),
           "median_kappa": up.kappa.median(), "clipped_quantizers": int((up.clip_share > 1e-3).sum()),
           "sqnr_add_db": sqnr_add(up)}
    print(f"\n{args.model}: {len(up)} activation quantizers upstream of the head")
    print(f"  step 1  INT8 noise per quantizer: median {out['int8_pred_median_db']:.1f} dB (Lemma 1), "
          f"exact {out['int8_exact_median_db']:.1f} dB; median kappa {out['median_kappa']:.1f}; "
          f"{out['clipped_quantizers']} quantizers clip more than 0.1% of the values")
    if q8 is not None:
        up8 = q8[(q8.model == args.model) & q8.upstream_of_head.astype(bool)]
        if not up8.empty:
            out["sqnr_add_fp8_db"] = sqnr_add(up8)
            print(f"          FP8 noise per quantizer: 31.5 dB (Lemma 1), exact median {up8.sqnr_inj_db.median():.1f} dB")

    a, b, n = calibration(pred, "sqnr_add_db", leave)
    out["rel_loss_int8_screening_pct"] = 100 * 10 ** (a + b * out["sqnr_add_db"])
    print(f"  step 2  SQNR_add (FP32 statistics only): INT8 {out['sqnr_add_db']:.1f} dB"
          + (f", FP8 {out['sqnr_add_fp8_db']:.1f} dB" if "sqnr_add_fp8_db" in out else ""))
    print(f"          screening ({n} calibration networks, x10 per {-1 / b:.1f} dB): INT8 relative loss "
          f"~{out['rel_loss_int8_screening_pct']:.1f}% (factor of up to ~6)")

    best = out["rel_loss_int8_screening_pct"]
    if args.head_sqnr is not None:
        out["sqnr_head_db"] = args.head_sqnr
        out["gamma_bar"] = 10 ** ((out["sqnr_add_db"] - args.head_sqnr) / 10)
        a, b, n = calibration(pred, "sqnr_head_db", leave)
        out["rel_loss_int8_pct"] = 100 * 10 ** (a + b * args.head_sqnr)
        best = out["rel_loss_int8_pct"]
        kind = "attenuates" if out["gamma_bar"] < 1 else "amplifies"
        print(f"  step 3  measured SQNR_h {args.head_sqnr:.1f} dB -> Gamma_bar {out['gamma_bar']:.2f} "
              f"(the network {kind} the injected noise)")
        print(f"          predicted INT8 relative loss ({n} calibration networks, x10 per {-1 / b:.1f} dB): "
              f"{out['rel_loss_int8_pct']:.1f}% (within a factor of about two)")

    if best <= args.max_loss:
        rec = "INT8"
    elif "sqnr_add_fp8_db" in out and out["sqnr_add_fp8_db"] > out["sqnr_add_db"] + 3:
        rec = ("FP8 injects less noise (SQNR_add above); its accuracy depends on the propagation as well, so check it "
               "with one FP8 measurement - or INT8 with selective precision (10_selective.py)")
    else:
        rec = "FP16, or INT8/FP8 with selective precision after measuring the layer sensitivities"
    out["recommendation"] = rec
    print(f"  step 5  acceptable loss {args.max_loss:g}% -> {rec}\n")
    pd.DataFrame([out]).to_csv(results_dir(cfg, "tables") / f"predict_{args.model}.csv", index=False)


if __name__ == "__main__":
    main()
