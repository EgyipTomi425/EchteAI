"""GPU latency / jitter / energy of every engine (FP32, FP16, INT8, INT8+FP32) at each batch size."""
import argparse
import json

import pandas as pd
import torch

from pepai import data, models
from pepai.bench import GPUMonitor, energy, idle_power_w, latency, summarize_latency
from pepai.config import load_config, results_dir
from pepai.models import SPECS
from pepai.trt import TRTModel


def sample_input(cfg, name, batch):
    """A real preprocessed image of the benchmark shape, repeated to the batch size."""
    coco, ids = data.coco_val_ids(cfg)
    spec = SPECS[name]
    if name == "frcnn_r50_fpn":
        frcnn = models.load_frcnn().cuda()
        for i in ids:
            img = data.load_rgb(data.coco_val_path(cfg, coco, i))
            x = models.frcnn_preprocess(frcnn, [img]).tensors[0]
            if tuple(x.shape) == spec.bench_shape:
                break
    elif name.startswith("yolov10"):
        x = models.letterbox(data.load_rgb(data.coco_val_path(cfg, coco, ids[0])))[0]
    else:
        x = models.classifier_preprocess(data.load_rgb(data.coco_val_path(cfg, coco, ids[0])))
    assert tuple(x.shape) == spec.bench_shape, (name, x.shape)
    return x.unsqueeze(0).repeat(batch, 1, 1, 1).cuda().contiguous()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--precisions", nargs="*")
    ap.add_argument("--out", default="benchmark.csv")
    ap.add_argument("--batch-sizes", nargs="*", type=int, help="instead of the configured batch sizes")
    ap.add_argument("--quick", action="store_true",
                    help="smoke check: one repeat, 300 iterations, CUDA-graph latency only, no energy")
    args = ap.parse_args()
    cfg = load_config()
    b = cfg["benchmark"]
    if args.quick:
        b = {**b, "repeats": 1, "iters": 300}
    if args.batch_sizes:
        b = {**b, "batch_sizes": args.batch_sizes}
    monitor = GPUMonitor()
    engine_dir = results_dir(cfg, "engines")
    out_dir = results_dir(cfg, "tables")
    (out_dir / "gpu_info.json").write_text(json.dumps(monitor.info(), indent=2))

    rows = []
    for name in args.models or cfg["models"]:
        for bs in b["batch_sizes"]:
            x = sample_input(cfg, name, bs)
            for precision in args.precisions or cfg["precisions"]:
                path = engine_dir / f"{name}_{precision}_bs{bs}.engine"
                model = TRTModel(path)
                inputs = {SPECS[name].input_name: x.to(model.dtype(SPECS[name].input_name)).contiguous()}
                for rep in range(b["repeats"]):
                    row = {"model": name, "precision": precision, "batch": bs, "repeat": rep,
                           "engine_mb": path.stat().st_size / 1e6,
                           "activation_mem_mb": model.device_memory_bytes() / 1e6}
                    ms_graph = latency(model, inputs, b["warmup"], b["iters"], cuda_graph=True)
                    row.update({f"graph_{k}": v for k, v in summarize_latency(ms_graph, bs).items()})
                    if args.quick:
                        rows.append(row)
                        print(f"{name:16s} {precision:9s} bs{bs}: graph p50 {row['graph_p50_ms']:.3f} ms", flush=True)
                        continue
                    ms = latency(model, inputs, b["warmup"], b["iters"])
                    row.update(summarize_latency(ms, bs))
                    idle = idle_power_w(monitor, b["idle_seconds"])
                    row.update(energy(model, inputs, monitor, b["energy_seconds"], bs, idle))
                    rows.append(row)
                    print(f"{name:16s} {precision:9s} bs{bs} rep{rep}: p50 {row['p50_ms']:.3f} ms "
                          f"(graph {row['graph_p50_ms']:.3f}), p99 {row['p99_ms']:.3f} ms, "
                          f"{row['energy_net_j_per_img'] * 1000:.2f} mJ/img net", flush=True)
                del model
                torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out_dir / args.out, index=False)
