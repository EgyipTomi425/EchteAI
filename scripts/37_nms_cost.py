"""GPU cost of appending class-wise NMS to the YOLOv10 output (GPU, timing).

The 300 detections per image that the NMS-free YOLOv10 head emits (stored detections of the deployed INT8
engine) are suppressed with torchvision's batched NMS on the GPU (IoU 0.7), timed with CUDA events after
warm-up, per image and for batches of eight images, and compared with the engine latency.
"""
import json

import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms

from pepai.config import load_config, results_dir

if __name__ == "__main__":
    cfg = load_config()
    rows = []
    for m in ("yolov10s", "yolov10x"):
        dets = json.loads((results_dir(cfg, "detections") / f"{m}_int8.json").read_text())
        by_img = {}
        for d in dets:
            by_img.setdefault(d["image_id"], []).append(d)
        imgs = []
        for v in list(by_img.values())[:2000]:
            b = torch.tensor([d["bbox"] for d in v], dtype=torch.float32)
            b[:, 2:] += b[:, :2]
            imgs.append((b.cuda(), torch.tensor([d["score"] for d in v]).cuda(),
                         torch.tensor([d["category_id"] for d in v]).cuda()))
        for bs in (1, 8):
            batches = []
            for i in range(0, len(imgs) - bs + 1, bs):
                chunk = imgs[i:i + bs]
                off = torch.cat([torch.full((len(c[0]),), k, device="cuda") for k, c in enumerate(chunk)])
                batches.append((torch.cat([c[0] for c in chunk]), torch.cat([c[1] for c in chunk]),
                                torch.cat([c[2] for c in chunk]) * 8 + off))   # class and image as NMS group
            for b, s, g in batches[:50]:
                batched_nms(b, s, g, 0.7)
            torch.cuda.synchronize()
            ms = []
            for b, s, g in batches:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                batched_nms(b, s, g, 0.7)
                e1.record()
                torch.cuda.synchronize()
                ms.append(e0.elapsed_time(e1))
            ms = np.array(ms)
            rows.append({"model": m, "batch": bs, "boxes_per_image": float(np.mean([len(c[0]) for c in imgs])),
                         "p50_ms": float(np.median(ms)), "p99_ms": float(np.percentile(ms, 99)),
                         "p50_ms_per_image": float(np.median(ms)) / bs})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "nms_cost.csv", index=False)
