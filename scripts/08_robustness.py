"""Robustness under adverse imaging conditions: mAP per precision and FP32/INT8 deviation per severity."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pepai import data
from pepai.agreement import image_disagreement
from pepai.config import load_config, results_dir
from pepai.corruptions import CONDITIONS, SEVERITIES, corrupt_file
from pepai.evaluate import coco_map, detect_frcnn, detect_yolo
from pepai.risk import PairRunner

DETECTORS = ["frcnn_r50_fpn", "yolov10s", "yolov10x"]


def corrupted_dir(cfg, condition, severity):
    return cfg["coco"]["root"] / "coco_c" / condition / str(severity)


def make_corrupted_sets(cfg, coco, ids, conditions):
    jobs = []
    conditions = [c for c in conditions if not (cfg["coco"]["root"] / "coco_c" / c / ".complete").exists()]
    for c in conditions:
        for s in SEVERITIES:
            d = corrupted_dir(cfg, c, s)
            d.mkdir(parents=True, exist_ok=True)
            for img_id in ids:
                seed = (img_id * 1000 + s * 31 + sum(map(ord, c))) % (2 ** 32)
                jobs.append((data.coco_val_path(cfg, coco, img_id), d / f"{img_id}.png", c, s, seed))
    if jobs:
        with ProcessPoolExecutor(64) as pool:
            list(tqdm(pool.map(corrupt_file, jobs, chunksize=16), total=len(jobs), desc="corrupting"))
    for c in conditions:
        (cfg["coco"]["root"] / "coco_c" / c / ".complete").touch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DETECTORS)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16", "int8"])
    ap.add_argument("--n", type=int)
    args = ap.parse_args()
    cfg = load_config()
    coco, ids = data.coco_val_ids(cfg, args.n or cfg["coco"]["n_activation"])
    make_corrupted_sets(cfg, coco, ids, args.conditions)
    engine_dir = results_dir(cfg, "engines")
    table = results_dir(cfg, "tables") / "robustness.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    done = {(r["model"], r["condition"], r["severity"], r["precision"]) for r in rows}

    settings = [("clean", 0)] + [(c, s) for c in args.conditions for s in SEVERITIES]
    for name in args.models:
        runner = PairRunner(cfg, name)
        for condition, severity in settings:
            if condition == "clean":
                load = None
            else:
                d = corrupted_dir(cfg, condition, severity)
                load = lambda i, d=d: data.load_rgb(d / f"{i}.png")  # noqa: E731
            for precision in args.precisions:
                if (name, condition, severity, precision) in done:
                    continue
                if name == "frcnn_r50_fpn":
                    dets = detect_frcnn(cfg, engine_dir / f"{name}_{precision}_dyn.engine", coco, ids, load=load)
                else:
                    dets = detect_yolo(cfg, engine_dir / f"{name}_{precision}_bs1.engine", coco, ids, load=load)
                rows.append({"model": name, "condition": condition, "severity": severity,
                             "precision": precision, **coco_map(coco, dets, ids)})
            if (name, condition, severity, "int8fp32-deviation") not in done:
                dev = []
                for img_id in ids:
                    img = load(img_id) if load else data.load_rgb(data.coco_val_path(cfg, coco, img_id))
                    m, df, dq = runner(img)
                    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
                    gtb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                                     a["bbox"][1] + a["bbox"][3]] for a in anns]).reshape(-1, 4)
                    gtl = np.array([a["category_id"] for a in anns])
                    dis = image_disagreement(df, dq, gtb, gtl)
                    dev.append({**m, "error": dis["error"], "error_tol": dis["error_tol"]})
                dev = pd.DataFrame(dev)
                rows.append({"model": name, "condition": condition, "severity": severity,
                             "precision": "int8fp32-deviation",
                             "mre_proj_median": dev.mre_proj.median(), "mre_proj_mean": dev.mre_proj.mean(),
                             "rel_l2_median": dev.rel_l2.median(), "sqnr_db_median": dev.sqnr_db.median(),
                             "disagreement_rate": dev.error.mean(),
                             "disagreement_rate_tol": dev.error_tol.mean()})
            pd.DataFrame(rows).to_csv(table, index=False)
            last = [r for r in rows if r["model"] == name and r["condition"] == condition
                    and r["severity"] == severity]
            print(name, condition, severity,
                  {r["precision"]: round(r.get("mAP", r.get("mre_proj_median", float("nan"))), 4) for r in last},
                  flush=True)
        del runner
        torch.cuda.empty_cache()
