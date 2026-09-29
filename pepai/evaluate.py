"""Task accuracy of TensorRT engines: COCO box mAP for detectors, top-1 for classifiers."""
import contextlib
import io
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from pepai import data, models
from pepai.trt import TRTModel

# Ultralytics' 80 contiguous class indices -> COCO category ids.
COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class TRTBackbone(nn.Module):
    """Drop-in replacement of the Faster R-CNN backbone that runs the TensorRT engine."""

    out_channels = 256

    def __init__(self, engine_path):
        super().__init__()
        self.trt = TRTModel(engine_path)

    def forward(self, x):
        x = x.to(self.trt.dtype("images")).contiguous()
        outs = self.trt(images=x)
        keys = ["0", "1", "2", "3", "pool"]
        # Clone: TRTModel reuses its output buffers between calls.
        return OrderedDict((k, outs[o].float().clone()) for k, o in zip(keys, models.FRCNN_OUTPUTS))


def frcnn_with_engine(engine_path):
    model = models.load_frcnn().cuda()
    model.backbone = TRTBackbone(engine_path)
    return model


def _loader(cfg, coco, load):
    return load or (lambda img_id: data.load_rgb(data.coco_val_path(cfg, coco, img_id)))


@torch.no_grad()
def detect_frcnn(cfg, engine_path, coco, ids, load=None):
    """load: optional img_id -> PIL image (e.g. corrupted images); defaults to COCO val2017 files."""
    load = _loader(cfg, coco, load)
    model = frcnn_with_engine(engine_path)
    results = []
    for img_id in tqdm(ids, desc=engine_path.stem, leave=False):
        img = load(img_id)
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255).cuda()
        out = model([x])[0]
        for box, score, label in zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()):
            x1, y1, x2, y2 = box
            results.append({"image_id": img_id, "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    return results


@torch.no_grad()
def detect_yolo(cfg, engine_path, coco, ids, conf=0.001, load=None):
    load = _loader(cfg, coco, load)
    model = TRTModel(engine_path)
    results = []
    for img_id in tqdm(ids, desc=engine_path.stem, leave=False):
        img = load(img_id)
        x, (r, left, top) = models.letterbox(img)
        x = x.unsqueeze(0).cuda().to(model.dtype("images")).contiguous()
        det = model(images=x)[model.outputs[0]][0].float().cpu().numpy()   # 300 x (x1 y1 x2 y2 score cls)
        det = det[det[:, 4] >= conf]
        w, h = img.size
        for x1, y1, x2, y2, score, cls in det:
            x1, x2 = np.clip([(x1 - left) / r, (x2 - left) / r], 0, w)
            y1, y2 = np.clip([(y1 - top) / r, (y2 - top) / r], 0, h)
            results.append({"image_id": img_id, "category_id": COCO80_TO_91[int(cls)],
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score)})
    return results


def coco_map(coco, results, ids):
    if not results:
        return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}
    dt = coco.loadRes(results)
    ev = COCOeval(coco, dt, "bbox")
    ev.params.imgIds = ids
    ev.evaluate()
    ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    s = ev.stats
    return {"mAP": s[0], "mAP50": s[1], "mAP75": s[2], "mAP_small": s[3], "mAP_medium": s[4],
            "mAP_large": s[5]}


@torch.no_grad()
def classify(engine_path, items, batch):
    """Top-1 predictions for (path, label) items with a static-batch engine."""
    model = TRTModel(engine_path)
    preds, labels = [], []
    for i in tqdm(range(0, len(items), batch), desc=engine_path.stem, leave=False):
        chunk = items[i:i + batch]
        xs = [models.classifier_preprocess(data.load_rgb(p)) for p, _ in chunk]
        n = len(xs)
        xs += [xs[-1]] * (batch - n)          # pad the last batch; padded rows are discarded
        x = torch.stack(xs).cuda().to(model.dtype("images")).contiguous()
        logits = model(images=x)["logits"][:n].float()
        preds += logits.argmax(1).tolist()
        labels += [lab for _, lab in chunk]
    return np.array(preds), np.array(labels)
