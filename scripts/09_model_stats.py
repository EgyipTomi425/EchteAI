"""Architecture table: parameters, FLOPs, conv layers, analysed tensors and weight bytes per precision."""
import onnx
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from pepai import models
from pepai.config import load_config, results_dir
from pepai.modelsize import weight_bytes
from pepai.models import SPECS


def torch_module(cfg, name):
    if name == "frcnn_r50_fpn":
        return models.FRCNNBackbone(models.load_frcnn())
    if name.startswith("yolov10"):
        return models.load_yolo(name, cfg["paths"]["data"] / "weights").model.float()
    return models.load_classifier(name)


if __name__ == "__main__":
    cfg = load_config()
    onnx_dir = results_dir(cfg, "onnx")
    rows = []
    for name in cfg["models"]:
        spec = SPECS[name]
        net = torch_module(cfg, name).eval()
        x = torch.randn(1, *spec.bench_shape)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            net(x)
        g = onnx.load(str(onnx_dir / f"{name}_fp32.onnx"), load_external_data=False).graph
        row = {
            "model": name, "task": spec.task, "input": "x".join(map(str, spec.bench_shape[1:])),
            "params_M": sum(p.numel() for p in net.parameters()) / 1e6,
            "gflops": fc.get_total_flops() / 1e9,
            "conv_layers": sum(n.op_type == "Conv" for n in g.node),
        }
        for precision in ("fp32", "fp16", "int8", "fp8"):
            row[f"weights_mb_{precision}"] = weight_bytes(onnx_dir / f"{name}_{precision}.onnx") / 1e6
        q = onnx.load(str(onnx_dir / f"{name}_int8.onnx"), load_external_data=False).graph
        row["qdq_pairs"] = sum(n.op_type == "QuantizeLinear" for n in q.node)
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "models.csv", index=False)
