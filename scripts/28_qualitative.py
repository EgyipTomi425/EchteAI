"""Qualitative example: FP32, INT8 and FP8 detections and the head-input deviation map on one traffic scene,
clean and under reduced contrast (GPU; analysis engines, INT8 with FP32 fallback).

The scene is chosen reproducibly among the first --search analysis images with at least four FP32
road-user detections, two of them vehicles: the one where, under the reduced contrast, the most FP32 detections vanish in INT8
beyond those that vanish in FP8. For every setting and format the FP32 and quantized detections and the
relative-error map of the channel-max projection of the head inputs (mean over the pyramid levels,
resampled to the image) are stored; 12_figures.fig_qualitative draws the figure.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from pepai import data
from pepai.agreement import greedy_match
from pepai.config import load_config, results_dir
from pepai.risk import PairRunner

ROAD_USERS = {1, 2, 3, 4, 6, 8}
VEHICLES = {3, 4, 6, 8}                   # car, motorcycle, bus, truck


def keep(d, thr):
    k = d["scores"] >= thr
    return d["boxes"][k], d["labels"][k], d["scores"][k]


def vanished(df, dq):
    fb, fl, fs = keep(df, 0.5)
    qb, ql, _ = keep(dq, 0.3)
    iou, _ = greedy_match(fb, fl, fs, qb, ql, 0.5)
    return int((iou < 0.5).sum()), int(np.isin(fl, list(ROAD_USERS)).sum()), int(np.isin(fl, list(VEHICLES)).sum())


def image_path(cfg, coco, img_id, condition, severity):
    if condition == "clean":
        return data.coco_val_path(cfg, coco, img_id)
    return cfg["coco"]["root"] / "coco_c" / condition / str(severity) / f"{img_id}.png"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov10x")
    ap.add_argument("--condition", default="contrast")
    ap.add_argument("--severity", type=int, default=4)
    ap.add_argument("--search", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config()
    coco, ids = data.coco_val_ids(cfg, cfg["coco"]["n_activation"])
    out = results_dir(cfg, "qualitative")
    runners = {q: PairRunner(cfg, args.model, q) for q in ("int8fp32", "fp8")}

    scores = []
    for img_id in ids[:args.search]:
        img = data.load_rgb(image_path(cfg, coco, img_id, args.condition, args.severity))
        v = {q: vanished(*r(img)[1:]) for q, r in runners.items()}
        if v["int8fp32"][1] >= 4 and v["int8fp32"][2] >= 2:     # a traffic scene: road users incl. vehicles
            scores.append((v["int8fp32"][0] - v["fp8"][0], v["int8fp32"][0], img_id))
    scores.sort(reverse=True)
    chosen = scores[0][2]
    (out / f"{args.model}_chosen.json").write_text(json.dumps({"image": chosen, "condition": args.condition,
                                                               "severity": args.severity, "ranking": scores[:10]}))
    print("chosen image", chosen, scores[:5], flush=True)

    for q, runner in runners.items():
        for condition, severity in (("clean", 0), (args.condition, args.severity)):
            img = data.load_rgb(image_path(cfg, coco, chosen, condition, severity))
            metrics, df, dq = runner(img)
            sx, sy, ox, oy, in_h, in_w = runner.geometry
            maps = [F.interpolate(m[None, None], size=(int(in_h), int(in_w)), mode="bilinear", align_corners=False)[0, 0]
                    for m in runner.rel_maps]
            dev = torch.stack(maps).mean(0)
            w, h = img.size
            y0, x0 = int(round(oy)), int(round(ox))
            dev = dev[y0:y0 + int(round(h * sy)), x0:x0 + int(round(w * sx))]
            dev = F.interpolate(dev[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            fb, fl, fs = keep(df, 0.5)
            qb, ql, qs = keep(dq, 0.3)
            np.savez_compressed(out / f"{args.model}_{chosen}_{condition}{severity}_{q}.npz",
                                image=np.asarray(img), deviation=dev.cpu().numpy().astype(np.float32),
                                fp32_boxes=fb, fp32_labels=fl, fp32_scores=fs, q_boxes=qb, q_labels=ql, q_scores=qs,
                                sqnr_db=metrics["sqnr_db"], mre_proj=metrics["mre_proj"])
            print(q, condition, severity, {k: round(v, 3) for k, v in metrics.items()}, flush=True)
