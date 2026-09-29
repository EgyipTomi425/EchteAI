"""FP32 vs quantized detection agreement (PEP-AI 'provable' component: MRE -> detection risk)."""
import numpy as np


def box_iou(a, b):
    """IoU matrix of xyxy boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def greedy_match(ref_boxes, ref_labels, ref_scores, q_boxes, q_labels, iou_thr):
    """Match every reference detection (highest score first) to an unused same-class detection.

    Returns, per reference detection, the IoU of its match (0 when unmatched) and the match index.
    """
    iou = box_iou(ref_boxes, q_boxes)
    iou[ref_labels[:, None] != q_labels[None, :]] = 0
    used = np.zeros(len(q_boxes), bool)
    best_iou = np.zeros(len(ref_boxes))
    match = -np.ones(len(ref_boxes), int)
    for i in np.argsort(-ref_scores):
        cand = np.where(~used & (iou[i] > 0))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmax(iou[i, cand])]
        if iou[i, j] >= iou_thr:
            used[j] = True
        best_iou[i], match[i] = iou[i, j], j
    return best_iou, match


def image_disagreement(fp32, quant, gt_boxes, gt_labels, score_thr=0.5, iou_thr=0.5, tolerance=0.2):
    """Per-image comparison following the CITDS definition of a detection error.

    fp32 / quant: dicts with 'boxes' (N,4 xyxy), 'labels', 'scores'.
    A detection error is (i) a ground-truth object found by FP32 but not by the quantized model, or
    (ii) an FP32 detection whose same-class quantized counterpart has IoU < iou_thr or is missing.
    The strict variant applies score_thr to both models. The tolerant variant accepts quantized
    counterparts down to score_thr - tolerance, so that pure score-threshold flicker of borderline
    detections is not counted as a failure.
    """
    def keep(d, thr):
        m = d["scores"] >= thr
        return d["boxes"][m], d["labels"][m], d["scores"][m]

    def found(boxes, labels):
        if len(boxes) == 0:
            return np.zeros(len(gt_boxes), bool)
        iou = box_iou(gt_boxes, boxes)
        iou[gt_labels[:, None] != labels[None, :]] = 0
        return (iou >= iou_thr).any(1)

    fb, fl, fs = keep(fp32, score_thr)
    f_found = found(fb, fl)
    out = {"n_fp32": len(fb), "n_gt": len(gt_boxes)}
    for tag, q_thr in (("", score_thr), ("_tol", score_thr - tolerance)):
        qb, ql, qs = keep(quant, q_thr)
        best_iou, match = greedy_match(fb, fl, fs, qb, ql, iou_thr)
        matched = best_iou >= iou_thr
        shifted = (best_iou > 0) & ~matched
        vanished = best_iou == 0
        missed_gt = int((f_found & ~found(qb, ql)).sum())
        out.update({
            f"vanished{tag}": int(vanished.sum()), f"shifted{tag}": int(shifted.sum()),
            f"missed_gt{tag}": missed_gt,
            f"error{tag}": int(vanished.sum() + shifted.sum() + missed_gt > 0),
        })
        if tag == "":
            n_q = len(qb)
            used = set(match[matched | shifted].tolist())
            out["n_quant"] = n_q
            out["appeared"] = int(n_q - len(used))
            out["mean_matched_iou"] = float(best_iou[matched].mean()) if matched.any() else np.nan
            fc = (fb[matched, :2] + fb[matched, 2:]) / 2
            qc = (qb[match[matched], :2] + qb[match[matched], 2:]) / 2
            out["centre_shift_px"] = np.linalg.norm(fc - qc, axis=1).tolist()
            # vertical shift of the box bottom edge: drives the ground-plane distance error
            out["bottom_shift_px"] = (qb[match[matched], 3] - fb[matched, 3]).tolist()
    return out
