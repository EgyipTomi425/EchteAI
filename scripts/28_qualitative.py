"""Qualitative example: FP32, INT8 and FP8 detections and the head-input deviation map on one traffic scene,
clean and under reduced contrast (GPU; analysis engines with FP32 fallback for INT8).

For every setting and quantized format it stores the FP32 and quantized detections (score >= 0.5) and the
median-free relative-error map of the channel-max projection of the head inputs (mean over the pyramid
levels), resampled to the original image. The figure is drawn by 12_figures.fig_qualitative.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from pepai import data
from pepai.config import load_config, results_dir
from pepai.risk import PairRunner

SETTINGS = [("clean", 0), ("contrast", 4)]


def to_arrays(d, thr=0.5):
    keep = d["scores"] >= thr
    return d["boxes"][keep], d["labels"][keep], d["scores"][keep]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov10x")
    ap.add_argument("--image", type=int, default=336232)
    args = ap.parse_args()
    cfg = load_config()
    coco, _ = data.coco_val_ids(cfg)
    out = results_dir(cfg, "qualitative")
    for quant in ("int8fp32", "fp8"):
        runner = PairRunner(cfg, args.model, quant)
        for condition, severity in SETTINGS:
            path = (data.coco_val_path(cfg, coco, args.image) if condition == "clean"
                    else cfg["coco"]["root"] / "coco_c" / condition / str(severity) / f"{args.image}.png")
            img = data.load_rgb(path)
            metrics, df, dq = runner(img)
            sx, sy, ox, oy, in_h, in_w = runner.geometry
            maps = [F.interpolate(m[None, None], size=(int(in_h), int(in_w)), mode="bilinear", align_corners=False)[0, 0]
                    for m in runner.rel_maps]
            dev = torch.stack(maps).mean(0)
            w, h = img.size
            y0, x0 = int(round(oy)), int(round(ox))
            dev = dev[y0:y0 + int(round(h * sy)), x0:x0 + int(round(w * sx))]
            dev = F.interpolate(dev[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            fb, fl, fs = to_arrays(df)
            qb, ql, qs = to_arrays(dq, 0.3)          # tolerant: counterparts down to 0.3 (as in the risk analysis)
            np.savez_compressed(out / f"{args.model}_{args.image}_{condition}{severity}_{quant}.npz",
                                image=np.asarray(img), deviation=dev.cpu().numpy().astype(np.float32),
                                fp32_boxes=fb, fp32_labels=fl, fp32_scores=fs,
                                q_boxes=qb, q_labels=ql, q_scores=qs, sqnr_db=metrics["sqnr_db"],
                                mre_proj=metrics["mre_proj"])
            print(quant, condition, severity, {k: round(v, 3) for k, v in metrics.items()}, len(fb), len(qb), flush=True)
        del runner
        torch.cuda.empty_cache()
