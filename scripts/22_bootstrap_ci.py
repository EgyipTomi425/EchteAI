"""Paired bootstrap confidence intervals for accuracy differences to FP32 (CPU only).

Detectors: COCOeval.evaluate() is run once per precision; every bootstrap replicate resamples the
images with replacement and re-runs only accumulate() on the per-image evaluation records, using the
same resample for FP32 and the quantized model (paired). Classifiers: resampling of per-image
correctness. Reports mAP / top-1 with 95% CIs and the paired difference with its 95% CI.
"""
import contextlib
import copy
import io
import json

import numpy as np
import pandas as pd
from pycocotools.cocoeval import COCOeval

from pepai import data
from pepai.config import load_config, results_dir

N_BOOT = 1000
DETECTORS = ["frcnn_r50_fpn", "yolov10s", "yolov10x"]
CLASSIFIERS = ["efficientnet_b0", "densenet121"]
PRECISIONS = ["fp32", "fp16", "int8", "int8fp32", "fp8"]


def evaluated(coco, results, ids):
    ev = COCOeval(coco, coco.loadRes(results), "bbox")
    ev.params.imgIds = ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
    # (catId, areaRng, imgId) -> record, so resampled image lists can be re-assembled
    table = {}
    k = 0
    for cat in ev.params.catIds:
        for area in ev.params.areaRng:
            for img in ev.params.imgIds:
                table[(cat, tuple(area), img)] = ev.evalImgs[k]
                k += 1
    return ev, table


def resampled_map(ev, table, sample):
    e = copy.copy(ev)
    e.params = copy.deepcopy(ev.params)
    e.params.imgIds = list(sample)
    e._paramsEval = copy.deepcopy(ev._paramsEval)     # accumulate() indexes records via _paramsEval
    e._paramsEval.imgIds = list(sample)
    e.evalImgs = [table[(cat, tuple(area), img)] for cat in e.params.catIds for area in e.params.areaRng
                  for img in sample]
    with contextlib.redirect_stdout(io.StringIO()):
        e.accumulate()
    prec = e.eval["precision"][:, :, :, 0, 2]            # all IoUs, recall, classes, area=all, maxDets=100
    return float(np.mean(prec[prec > -1]))


def ci(x):
    return np.percentile(x, [2.5, 97.5])


_STATE = {}


def _worker(job):
    precision, i = job
    ev, table = _STATE["evs"][precision]
    return resampled_map(ev, table, _STATE["samples"][i])


def parallel_boot(evs, samples, workers=64):
    """Forked workers share the evaluation records copy-on-write."""
    import multiprocessing as mp
    _STATE.update(evs=evs, samples=samples)
    jobs = [(p, i) for p in evs for i in range(len(samples))]
    with mp.get_context("fork").Pool(workers) as pool:
        values = pool.map(_worker, jobs, chunksize=4)
    out = {p: np.empty(len(samples)) for p in evs}
    for (p, i), v in zip(jobs, values):
        out[p][i] = v
    return out


if __name__ == "__main__":
    cfg = load_config()
    det_dir = results_dir(cfg, "detections")
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    coco, ids = data.coco_val_ids(cfg)
    ids = sorted(ids)
    for name in DETECTORS:
        evs = {}
        for p in PRECISIONS:
            f = det_dir / f"{name}_{p}.json"
            if f.exists():
                evs[p] = evaluated(coco, json.loads(f.read_text()), ids)
        if "fp32" not in evs:
            continue
        samples = [rng.choice(ids, len(ids), replace=True) for _ in range(N_BOOT)]
        boot = parallel_boot(evs, samples)
        for p in evs:
            full = resampled_map(*evs[p], ids)
            d = boot[p] - boot["fp32"]
            rows.append({"model": name, "precision": p, "metric": "mAP", "value": full,
                         "ci_lo": ci(boot[p])[0], "ci_hi": ci(boot[p])[1],
                         "delta": full - resampled_map(*evs["fp32"], ids), "delta_lo": ci(d)[0], "delta_hi": ci(d)[1]})
            print(rows[-1], flush=True)
    for name in CLASSIFIERS:
        preds = {p: np.load(det_dir / f"{name}_{p}_cls.npz") for p in PRECISIONS
                 if (det_dir / f"{name}_{p}_cls.npz").exists()}
        if "fp32" not in preds:
            continue
        correct = {p: (v["preds"] == v["labels"]).astype(float) for p, v in preds.items()}
        n = len(correct["fp32"])
        idx = rng.integers(0, n, (N_BOOT, n))
        for p, c in correct.items():
            b = c[idx].mean(1)
            d = b - correct["fp32"][idx].mean(1)
            rows.append({"model": name, "precision": p, "metric": "top1", "value": c.mean(),
                         "ci_lo": ci(b)[0], "ci_hi": ci(b)[1], "delta": c.mean() - correct["fp32"].mean(),
                         "delta_lo": ci(d)[0], "delta_hi": ci(d)[1]})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "accuracy_ci.csv", index=False)
