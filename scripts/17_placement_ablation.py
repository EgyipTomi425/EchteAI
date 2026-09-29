"""QDQ placement ablations diagnosed by the activation analysis.

Each variant re-quantizes a model with a different set of operator types kept in high precision,
then reports task accuracy and the head-input deviation on a probe set, next to the default
placement. Variants are chosen from the operator-level diagnosis, never from test accuracy.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from pepai import data, debug
from pepai.activations import analysed_tensors, final_tensors, tensor_metrics
from pepai.config import load_config, results_dir
from pepai.evaluate import classify, coco_map, detect_frcnn, detect_yolo
from pepai.inputs import calibration_inputs_iter
from pepai.models import SPECS, engine_shapes
from pepai.quant import calibration_inputs, excluded_nodes, excluded_op_types, int8_with_fp16, to_int8
from pepai.trt import TRTModel, build_engine

# variant -> {"ops": op types kept in high precision, "method": calibration method}
VARIANTS = {
    # DenseNet: pre-activation BatchNorm after Concat cannot be folded; its per-channel re-scaling
    # amplifies per-tensor input quantization noise, so quantize the convolution inputs instead.
    "densenet121": {"bn_fp16": {"ops": ["BatchNormalization"]}, "calib_max": {"method": "max"},
                    "bn_fp16_calib_max": {"ops": ["BatchNormalization"], "method": "max"}},
    # YOLOv10: entropy calibration clips the SiLU features before the class-logit convolutions coarsely.
    "yolov10s": {"calib_max": {"method": "max"}},
    "yolov10x": {"calib_max": {"method": "max"}},
    "frcnn_r50_fpn": {"calib_max": {"method": "max"}},
}


@torch.no_grad()
def probe_deviation(cfg, name, quant_onnx, n, work):
    """Median head-input SQNR on n calibration-split images: a label-free, test-independent criterion."""
    fp32 = results_dir(cfg, "onnx") / f"{name}_fp32.onnx"
    infos = analysed_tensors(fp32, quant_onnx)
    final = final_tensors(cfg, name, fp32, infos)
    eng = work / (quant_onnx.stem + "_final.engine")
    if not eng.exists():
        build_engine(quant_onnx, eng, engine_shapes(SPECS[name]), mark_outputs=final)
    ref = TRTModel(debug.ensure_engine(cfg, name, "fp32", "final" if SPECS[name].task == "det" else "full"))
    qnt = TRTModel(eng)
    vals = []
    for _, x in calibration_inputs_iter(cfg, name, n):
        fo = {k: v.clone() for k, v in ref(images=x.to(ref.dtype("images"))).items()}
        qo = qnt(images=x.to(qnt.dtype("images")))
        vals.append([tensor_metrics(fo[t], qo[t])["sqnr_db"] for t in final])
    return float(np.median(np.mean(vals, axis=1)))


def accuracy(cfg, name, engine):
    if SPECS[name].task == "cls":
        _, items = data.imagenetv2_split(cfg)
        p, lab = classify(engine, items, 8)
        return {"top1": float((p == lab).mean())}
    coco, ids = data.coco_val_ids(cfg)
    det = detect_frcnn if name == "frcnn_r50_fpn" else detect_yolo
    return coco_map(coco, det(cfg, engine, coco, ids), ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(VARIANTS))
    ap.add_argument("--probe", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config()
    work = results_dir(cfg, "ablation")
    table = results_dir(cfg, "tables") / "placement_ablation.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    for name in args.models:
        spec = SPECS[name]
        bs = 8 if spec.task == "cls" else 1
        variants = {"default": {}, **VARIANTS.get(name, {})}
        for tag, spec_v in variants.items():
            ops, method = spec_v.get("ops", []), spec_v.get("method", "entropy")
            if any(r["model"] == name and r["variant"] == tag for r in rows):
                continue
            q32 = work / f"{name}_{tag}_int8fp32.onnx"
            q16 = work / f"{name}_{tag}_int8.onnx"
            if tag == "default":
                q32, q16 = results_dir(cfg, "onnx") / f"{name}_int8fp32.onnx", results_dir(cfg, "onnx") / f"{name}_int8.onnx"
            elif not q16.exists():
                to_int8(results_dir(cfg, "onnx") / f"{name}_fp32.onnx", q32, calibration_inputs(cfg, name),
                        method=method, op_types_to_exclude=excluded_op_types(cfg, name) + ops,
                        nodes_to_exclude=excluded_nodes(cfg, name))
                int8_with_fp16(q32, q16)
            eng = work / f"{name}_{tag}_int8_bs{bs}.engine"
            if not eng.exists():
                shape = (bs, *spec.bench_shape)
                shapes = engine_shapes(spec) if spec.dynamic_hw else {spec.input_name: (shape,) * 3}
                build_engine(q16, eng, shapes)
            row = {"model": name, "variant": tag, "excluded_ops": ";".join(ops), "calibration": method,
                   **accuracy(cfg, name, eng),
                   "head_sqnr_db": probe_deviation(cfg, name, q32, args.probe, work)}
            rows.append(row)
            pd.DataFrame(rows).to_csv(table, index=False)
            print(row, flush=True)
