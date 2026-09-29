"""Upper bound of spatial early-exit savings (future-work estimate, CPU only).

If a future accelerator skipped the deep layers in background regions, where quantization deviation
is largest and later layers suppress it anyway, the skippable share of compute is at most
(background area fraction) x (FLOP share of the deep stages). Both factors are measured here for the
Faster R-CNN backbone on the 500 analysis images; the object/background deviation comes from
06_activations.py.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from pepai import data, models
from pepai.config import load_config, results_dir
from pepai.models import SPECS

DEEP = ("backbone.body.layer4", "backbone.fpn")

if __name__ == "__main__":
    cfg = load_config()
    net = models.FRCNNBackbone(models.load_frcnn()).eval()
    x = torch.randn(1, *SPECS["frcnn_r50_fpn"].bench_shape)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(x)
    counts = fc.get_flop_counts()
    total = sum(counts["Global"].values())
    per_stage = {k: sum(v.values()) for k, v in counts.items() if k.count(".") == 3 or k.endswith("fpn")}
    deep = sum(sum(v.values()) for k, v in counts.items()
               if any(k.endswith(d) for d in DEEP))

    coco, ids = data.coco_val_ids(cfg, cfg["coco"]["n_activation"])
    bg = []
    for img_id in ids:
        info = coco.loadImgs(img_id)[0]
        mask = np.zeros((info["height"], info["width"]), bool)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False)):
            x0, y0, w, h = a["bbox"]
            mask[int(y0):int(np.ceil(y0 + h)), int(x0):int(np.ceil(x0 + w))] = True
        bg.append(1 - mask.mean())
    bg = np.array(bg)

    act = results_dir(cfg, "activations") / "frcnn_r50_fpn_int8fp32.csv.gz"
    region = {}
    if act.exists():
        df = pd.read_csv(act, usecols=["op", "mre_proj_object", "mre_proj_background"])
        conv = df[df.op == "Conv"].dropna()
        region = {"mre_object_median": conv.mre_proj_object.median(),
                  "mre_background_median": conv.mre_proj_background.median()}

    row = {"model": "frcnn_r50_fpn", "gflops_total": total / 1e9, "deep_flop_share": deep / total,
           "background_fraction_median": np.median(bg), "background_fraction_mean": bg.mean(),
           "skippable_upper_bound_median": np.median(bg) * deep / total, **region}
    pd.DataFrame([row]).to_csv(results_dir(cfg, "tables") / "early_exit.csv", index=False)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    print({k: round(v / total, 3) for k, v in per_stage.items()})
