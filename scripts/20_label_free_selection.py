"""Label-free selection criteria for quantization variants, evaluated on the calibration split.

For every variant of 17_placement_ablation.py two criteria are computed without any label and on
images disjoint from all evaluation data:
  head_sqnr_db     energy-based fidelity of the head-input features (from 17_placement_ablation.py)
  consistency      output consistency with the FP32 model: for detectors the COCO AP of the variant's
                   detections scored against the FP32 detections (score >= 0.25) as pseudo ground truth,
                   for classifiers the top-1 agreement with FP32.
The script then reports, per model, which variant each criterion would pick and its task accuracy.
"""
import contextlib
import io

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pepai import data, models
from pepai.config import load_config, results_dir
from pepai.evaluate import COCO80_TO_91, TRTBackbone
from pepai.models import SPECS
from pepai.trt import TRTModel

N_IMAGES = 256
PSEUDO_GT_SCORE = 0.25


def calib_images(cfg, name):
    if SPECS[name].task == "cls":
        calib, _ = data.imagenetv2_split(cfg)
        return [p for p, _ in calib[:N_IMAGES]]
    return data.coco_calib_paths(cfg)[:N_IMAGES]


@torch.no_grad()
def detections(cfg, name, engine_path, paths):
    """COCO-style result dicts (image ids are list indices) of an engine on the given images."""
    out = []
    if name == "frcnn_r50_fpn":
        model = models.load_frcnn().cuda()
        model.backbone = TRTBackbone(engine_path)
        for i, p in enumerate(paths):
            img = data.load_rgb(p)
            x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255).cuda()
            d = model([x])[0]
            for b, lab, s in zip(d["boxes"].tolist(), d["labels"].tolist(), d["scores"].tolist()):
                out.append({"image_id": i, "category_id": int(lab), "score": float(s),
                            "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
        return out
    m = TRTModel(engine_path)
    for i, p in enumerate(paths):
        img = data.load_rgb(p)
        x, (r, left, top) = models.letterbox(img)
        det = m(images=x.unsqueeze(0).cuda().to(m.dtype("images")).contiguous())[m.outputs[0]][0].float().cpu().numpy()
        det = det[det[:, 4] >= 0.001]
        w, h = img.size
        for x1, y1, x2, y2, s, c in det:
            x1, x2 = np.clip([(x1 - left) / r, (x2 - left) / r], 0, w)
            y1, y2 = np.clip([(y1 - top) / r, (y2 - top) / r], 0, h)
            out.append({"image_id": i, "category_id": COCO80_TO_91[int(c)], "score": float(s),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
    return out


def score_fidelity(ref_dets, dets, min_score=0.5):
    """Exploratory (defined after the YOLOv10 calibration case): 1 - mean |score difference| of the
    same-class, IoU >= 0.5 matched counterparts of confident FP32 detections (unmatched count as 1)."""
    from collections import defaultdict
    from pepai.agreement import box_iou
    by_img = defaultdict(list)
    for d in dets:
        by_img[d["image_id"]].append(d)
    errs = []
    for r in ref_dets:
        if r["score"] < min_score:
            continue
        cands = [d for d in by_img[r["image_id"]] if d["category_id"] == r["category_id"]]
        if not cands:
            errs.append(1.0)
            continue
        xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]  # noqa: E731
        iou = box_iou(np.array([xyxy(r["bbox"])]), np.array([xyxy(c["bbox"]) for c in cands]))[0]
        ok = iou >= 0.5
        errs.append(min(abs(r["score"] - c["score"]) for c, k in zip(cands, ok) if k) if ok.any() else 1.0)
    return 1.0 - float(np.mean(errs))


def consistency_ap(ref_dets, dets, paths):
    """AP of dets against the FP32 detections above PSEUDO_GT_SCORE as ground truth."""
    gt = COCO()
    anns = [{"id": k + 1, "image_id": d["image_id"], "category_id": d["category_id"], "bbox": d["bbox"],
             "area": d["bbox"][2] * d["bbox"][3], "iscrowd": 0}
            for k, d in enumerate(x for x in ref_dets if x["score"] >= PSEUDO_GT_SCORE)]
    gt.dataset = {"images": [{"id": i} for i in range(len(paths))], "annotations": anns,
                  "categories": [{"id": c} for c in sorted(set(COCO80_TO_91))]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


@torch.no_grad()
def top1(engine_path, paths):
    m = TRTModel(engine_path)
    preds = []
    for i in range(0, len(paths), 8):
        xs = [models.classifier_preprocess(data.load_rgb(p)) for p in paths[i:i + 8]]
        n = len(xs)
        xs += [xs[-1]] * (8 - n)
        logits = m(images=torch.stack(xs).cuda().to(m.dtype("images")).contiguous())["logits"][:n].float()
        preds += logits.argmax(1).tolist()
    return np.array(preds)


if __name__ == "__main__":
    cfg = load_config()
    abl = results_dir(cfg, "ablation")
    eng = results_dir(cfg, "engines")
    table = pd.read_csv(results_dir(cfg, "tables") / "placement_ablation.csv")
    rows = []
    for name, g in table.groupby("model"):
        paths = calib_images(cfg, name)
        cls = SPECS[name].task == "cls"
        bs = 8 if cls else 1
        ref_engine = eng / (f"{name}_fp32_bs{bs}.engine" if not SPECS[name].dynamic_hw else f"{name}_fp32_dyn.engine")
        ref = top1(ref_engine, paths) if cls else detections(cfg, name, ref_engine, paths)
        for _, r in g.iterrows():
            tag_engine = abl / f"{name}_{r.variant}_int8_bs{bs}.engine"
            if not tag_engine.exists():
                continue
            fid = np.nan
            if cls:
                c = float((top1(tag_engine, paths) == ref).mean())
            else:
                dets = detections(cfg, name, tag_engine, paths)
                c = consistency_ap(ref, dets, paths)
                fid = score_fidelity(ref, dets)
            rows.append({"model": name, "variant": r.variant, "head_sqnr_db": r.head_sqnr_db, "consistency": c,
                         "score_fidelity_exploratory": fid, "accuracy": r.top1 if cls else r.mAP})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir(cfg, "tables") / "label_free_selection.csv", index=False)
    for name, g in df.groupby("model"):
        best = g.loc[g.accuracy.idxmax()]
        for crit in ("head_sqnr_db", "consistency", "score_fidelity_exploratory"):
            if g[crit].isna().all():
                continue
            pick = g.loc[g[crit].idxmax()]
            print(f"{name}: {crit} picks {pick.variant} (accuracy {pick.accuracy:.4f}); best {best.variant} "
                  f"({best.accuracy:.4f})")
