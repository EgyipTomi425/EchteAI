"""Paired FP32 / quantized execution returning head-input deviation and both detection sets."""
import numpy as np
import torch

from pepai import debug, models
from pepai.activations import analysed_tensors, final_tensors, tensor_metrics
from pepai.config import results_dir
from pepai.evaluate import COCO80_TO_91
from pepai.trt import TRTModel


def final_metrics(fo, qo, final):
    per = [tensor_metrics(fo[t], qo[t]) for t in final]
    num = sum((qo[t].float() - fo[t].float()).pow(2).sum().item() for t in final)
    den = sum(fo[t].float().pow(2).sum().item() for t in final)
    return {
        "mre_proj": float(np.mean([m["mre_proj"] for m in per])),
        "mre_elem": float(np.mean([m["mre_elem"] for m in per])),
        "rel_l2": float(np.sqrt(num / den)),
        "sqnr_db": float(10 * np.log10(den / max(num, 1e-30))),
    }


class PairRunner:
    """Runs the strict-FP32 and the quantized engine on the same image."""

    def __init__(self, cfg, name, quant="int8fp32"):
        onnx_dir = results_dir(cfg, "onnx")
        infos = analysed_tensors(onnx_dir / f"{name}_fp32.onnx", onnx_dir / f"{name}_{quant}.onnx")
        self.name = name
        self.final = final_tensors(cfg, name, onnx_dir / f"{name}_fp32.onnx", infos)
        self.ref = TRTModel(debug.ensure_engine(cfg, name, "fp32", "final"))
        self.qnt = TRTModel(debug.ensure_engine(cfg, name, quant, "final"))
        self.frcnn = models.load_frcnn().cuda() if name == "frcnn_r50_fpn" else None

    @torch.no_grad()
    def __call__(self, img):
        """img: PIL RGB. Returns (deviation metrics, FP32 detections, quantized detections)."""
        w, h = img.size
        if self.frcnn is not None:
            il = models.frcnn_preprocess(self.frcnn, [img])
            x = il.tensors.contiguous()
            hh, ww = il.image_sizes[0]
            self.geometry = (ww / w, hh / h, 0.0, 0.0, x.shape[-2], x.shape[-1])
        else:
            x, (r, left, top) = models.letterbox(img)
            x = x.unsqueeze(0).cuda().contiguous()
            self.geometry = (r, r, left, top, x.shape[-2], x.shape[-1])
        fo = {k: v.clone() for k, v in self.ref(images=x.to(self.ref.dtype("images"))).items()}
        qo = self.qnt(images=x.to(self.qnt.dtype("images")))
        # Relative-error maps of the channel-max projection of every head-input tensor (for boxes).
        self.rel_maps = []
        for t in self.final:
            af, aq = fo[t][0].float().amax(0), qo[t][0].float().amax(0)
            self.rel_maps.append(((aq - af).abs() / (af.abs() + 1e-6)))
        if self.frcnn is not None:
            df, dq = self._frcnn_heads(il, fo, (h, w)), self._frcnn_heads(il, qo, (h, w))
        else:
            df = self._yolo_dets(fo[self.ref.outputs[0]], r, left, top, w, h)
            dq = self._yolo_dets(qo[self.qnt.outputs[0]], r, left, top, w, h)
        return final_metrics(fo, qo, self.final), df, dq

    def local_deviation(self, box):
        """Median relative error of the head-input projections inside an original-image box (xyxy),
        averaged over the head-input tensors (pyramid levels)."""
        sx, sy, ox, oy, in_h, in_w = self.geometry
        x0, y0, x1, y1 = box[0] * sx + ox, box[1] * sy + oy, box[2] * sx + ox, box[3] * sy + oy
        vals = []
        for rel in self.rel_maps:
            fh, fw = rel.shape
            c0, c1 = int(x0 / in_w * fw), int(np.ceil(x1 / in_w * fw))
            r0, r1 = int(y0 / in_h * fh), int(np.ceil(y1 / in_h * fh))
            c1, r1 = max(c1, c0 + 1), max(r1, r0 + 1)
            vals.append(rel[r0:r1, c0:c1].median().item())
        return float(np.mean(vals))

    def _frcnn_heads(self, image_list, feats, orig_size):
        f = self.frcnn
        feats = dict(zip(["0", "1", "2", "3", "pool"], [feats[o].float() for o in models.FRCNN_OUTPUTS]))
        proposals, _ = f.rpn(image_list, feats)
        dets, _ = f.roi_heads(feats, proposals, image_list.image_sizes)
        d = f.transform.postprocess(dets, image_list.image_sizes, [orig_size])[0]
        return {k: d[k].cpu().numpy() for k in ("boxes", "labels", "scores")}

    @staticmethod
    def _yolo_dets(out, r, left, top, w, h):
        d = out[0].float().cpu().numpy()
        boxes = d[:, :4].copy()
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / r, 0, w)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / r, 0, h)
        labels = np.array([COCO80_TO_91[int(c)] for c in d[:, 5]])
        return {"boxes": boxes, "labels": labels, "scores": d[:, 4]}


def to_coco_results(img_id, det):
    return [{"image_id": img_id, "category_id": int(lab), "score": float(s),
             "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]}
            for b, lab, s in zip(det["boxes"], det["labels"], det["scores"])]
