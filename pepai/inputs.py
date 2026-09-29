"""Preprocessed analysis inputs with ground-truth object masks."""
import numpy as np
import torch

from pepai import data, models
from pepai.models import SPECS


def analysis_inputs(cfg, name, n, image_dir=None):
    """Yields (image key, input tensor, object mask at input resolution or None).

    Classifiers: the first n ImageNetV2 evaluation images. Detectors: the first n images of the
    seeded COCO val2017 order; image_dir optionally points to corrupted copies named <id>.png.
    """
    if SPECS[name].task == "cls":
        _, items = data.imagenetv2_split(cfg)
        for path, _ in items[:n]:
            yield path.stem, models.classifier_preprocess(data.load_rgb(path)).unsqueeze(0).cuda(), None
        return
    coco, ids = data.coco_val_ids(cfg, n)
    frcnn = models.load_frcnn().cuda() if name == "frcnn_r50_fpn" else None
    for img_id in ids:
        path = image_dir / f"{img_id}.png" if image_dir else data.coco_val_path(cfg, coco, img_id)
        img = data.load_rgb(path)
        w, h = img.size
        if frcnn is not None:
            il = models.frcnn_preprocess(frcnn, [img])
            x = il.tensors
            (hh, ww) = il.image_sizes[0]
            sx, sy, ox, oy = ww / w, hh / h, 0, 0
        else:
            x, (r, left, top) = models.letterbox(img)
            x = x.unsqueeze(0).cuda()
            sx, sy, ox, oy = r, r, left, top
        mask = torch.zeros(x.shape[-2:], dtype=torch.bool, device="cuda")
        for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False)):
            bx, by, bw, bh = a["bbox"]
            x0, y0 = int(bx * sx + ox), int(by * sy + oy)
            x1, y1 = int(np.ceil((bx + bw) * sx + ox)), int(np.ceil((by + bh) * sy + oy))
            mask[y0:y1, x0:x1] = True
        yield img_id, x.contiguous(), mask


def calibration_inputs_iter(cfg, name, n):
    """Yields (key, input tensor) from the calibration split (never used for evaluation)."""
    if SPECS[name].task == "cls":
        calib, _ = data.imagenetv2_split(cfg)
        for path, _ in calib[:n]:
            yield path.stem, models.classifier_preprocess(data.load_rgb(path)).unsqueeze(0).cuda()
        return
    frcnn = models.load_frcnn().cuda() if name == "frcnn_r50_fpn" else None
    for path in data.coco_calib_paths(cfg)[:n]:
        img = data.load_rgb(path)
        if frcnn is not None:
            x = models.frcnn_preprocess(frcnn, [img]).tensors
        else:
            x = models.letterbox(img)[0].unsqueeze(0).cuda()
        yield path.stem, x.contiguous()
