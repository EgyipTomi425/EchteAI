"""AIBO scenario: annual fleet energy, cost and CO2 per precision, and the break-even error penalty.

Cost model (per year, per model and precision p):

    C(p, lambda) = E_img(p) * N * price + K(p) * N * lambda

with N inferences per year, E_img the measured GPU energy per image at the fleet batch size, K the
number of critical errors per image relative to FP32 and lambda the penalty per critical error.
Critical errors (23_deployed_agreement.py): for detectors, FP32 road-user detections (person, bicycle,
car, motorcycle, bus, truck) that vanish in the deployed engine even under the score-tolerant
definition; for classifiers, top-1 disagreements with FP32. The break-even penalty
lambda* = (E_img(fp32) - E_img(p)) * price / K(p) is the penalty per error above which p costs more
than FP32.
"""
import numpy as np
import pandas as pd

from pepai.config import load_config, results_dir

if __name__ == "__main__":
    cfg = load_config()
    a = cfg["aibo"]
    tables = results_dir(cfg, "tables")
    bench = pd.concat([pd.read_csv(f) for f in (tables / "benchmark.csv", tables / "benchmark_bs6.csv") if f.exists()])
    bench = bench[bench.batch == a["batch"]].groupby(["model", "precision"]).median(numeric_only=True)
    inferences = (a["fleet_size"] * a["cameras_per_vehicle"] * a["fps"] * 3600
                  * a["hours_per_day"] * a["days_per_year"])
    agree_p = tables / "deployed_agreement.csv"
    agree = pd.read_csv(agree_p).set_index(["model", "precision"]) if agree_p.exists() else None

    def critical_per_image(model, precision):
        if precision == "fp32":
            return 0.0
        if agree is None or (model, precision) not in agree.index:
            return np.nan
        r = agree.loc[(model, precision)]
        v = r.get("road_user_vanished_per_image", np.nan)
        return r.image_error_rate if pd.isna(v) else v

    rows = []
    for (model, precision), b in bench.iterrows():
        for basis in ("energy_j_per_img", "energy_net_j_per_img"):
            kwh = inferences * b[basis] / 3.6e6
            row = {"model": model, "precision": precision, "energy_basis": basis,
                   "inferences_per_year": inferences, "energy_mj_per_img": 1000 * b[basis], "kwh_per_year": kwh,
                   "critical_per_image": critical_per_image(model, precision)}
            for label, price, grid in zip(("low", "ref", "high"), a["electricity_eur_per_kwh"],
                                          a["grid_gco2_per_kwh"]):
                row[f"eur_per_year_{label}"] = kwh * price
                row[f"tco2_per_year_{label}"] = kwh * grid / 1e6
            rows.append(row)
    df = pd.DataFrame(rows)

    out = []
    for (model, basis), g in df.groupby(["model", "energy_basis"]):
        ref = g[g.precision == "fp32"].iloc[0]
        for _, r in g.iterrows():
            saving = ref.eur_per_year_ref - r.eur_per_year_ref
            events = r.critical_per_image * inferences
            out.append({**r.to_dict(), "kwh_saved": ref.kwh_per_year - r.kwh_per_year, "eur_saved_ref": saving,
                        "eur_saved_low": ref.eur_per_year_low - r.eur_per_year_low,
                        "eur_saved_high": ref.eur_per_year_high - r.eur_per_year_high,
                        "tco2_saved_ref": ref.tco2_per_year_ref - r.tco2_per_year_ref,
                        "tco2_saved_low": ref.tco2_per_year_low - r.tco2_per_year_low,
                        "tco2_saved_high": ref.tco2_per_year_high - r.tco2_per_year_high,
                        "critical_per_year": events,
                        "breakeven_eur_per_error": saving / events if events > 0 else np.nan})
    pd.DataFrame(out).to_csv(tables / "aibo.csv", index=False)
    print(pd.DataFrame(out).query("energy_basis == 'energy_j_per_img'")[
        ["model", "precision", "energy_mj_per_img", "kwh_per_year", "eur_saved_ref", "tco2_saved_ref",
         "critical_per_year", "breakeven_eur_per_error"]].round(6).to_string(index=False))
