"""Robustness of the INT8 calibration variants under reduced contrast (test of the clipping mechanism).

Under low contrast, some activations of YOLOv10 grow beyond the range fixed by entropy calibration and are
clipped (26_quantizer_snr.py with --condition). Max calibration sets a wider range, so its penalty should
grow less. Evaluates the default (entropy) and the max-calibrated INT8 engines of the placement ablation
(17_placement_ablation.py) on the same 500 images and corruptions as 08_robustness.py.
"""
import pandas as pd

from pepai import data
from pepai.config import load_config, results_dir
from pepai.evaluate import coco_map, detect_yolo

VARIANTS = {"entropy": "{name}_default_int8_bs1.engine", "max": "{name}_calib_max_int8_bs1.engine"}
SETTINGS = [("clean", 0)] + [("contrast", s) for s in (1, 2, 3, 4, 5)]

if __name__ == "__main__":
    cfg = load_config()
    coco, ids = data.coco_val_ids(cfg, cfg["coco"]["n_activation"])
    abl = results_dir(cfg, "ablation")
    rows = []
    for name in ("yolov10s", "yolov10x"):
        for variant, pattern in VARIANTS.items():
            engine = abl / pattern.format(name=name)
            if not engine.exists():
                continue
            for condition, severity in SETTINGS:
                load = None
                if condition != "clean":
                    d = cfg["coco"]["root"] / "coco_c" / condition / str(severity)
                    load = lambda i, d=d: data.load_rgb(d / f"{i}.png")  # noqa: E731
                dets = detect_yolo(cfg, engine, coco, ids, load=load)
                rows.append({"model": name, "calibration": variant, "condition": condition, "severity": severity,
                             **coco_map(coco, dets, ids)})
                print(rows[-1]["model"], variant, condition, severity, round(rows[-1]["mAP"], 4), flush=True)
                pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "calibration_robustness.csv", index=False)
