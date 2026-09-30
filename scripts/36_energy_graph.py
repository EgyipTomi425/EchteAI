"""Energy per image with CUDA graphs (GPU, one engine at a time).

04_benchmark.py measures latency with CUDA graphs but energy with direct enqueues (20 per synchronisation), so
that for small engines launch gaps lower the sustained throughput (to 69-100 % of the CUDA-graph throughput) and
add idle-time energy. This script repeats the energy measurement with CUDA-graph replays, i.e. in the execution
mode of the reported latencies and of a deployed real-time pipeline: idle power, 20 s of sustained replays,
median of three repeats, for the engines of the main results and of the fleet scenario.
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pepai.bench import GPUMonitor, idle_power_w
from pepai.config import load_config, results_dir
from pepai.models import SPECS
from pepai.trt import TRTModel

ORDER = ["frcnn_r50_fpn", "yolov10s", "yolov10x", "efficientnet_b0", "densenet121"]


def graph_energy(model, inputs, monitor, seconds, batch, idle_w, per_sync=20):
    model.bind(**inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(20):
            model.execute()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            model.execute()
        for _ in range(20):
            graph.replay()
        stream.synchronize()
        n, clocks = 0, []
        e0, t0 = monitor.energy_mj(), time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            for _ in range(per_sync):
                graph.replay()
            n += per_sync
            stream.synchronize()
            clocks.append(monitor.sm_clock())
        elapsed = time.perf_counter() - t0
        joules = (monitor.energy_mj() - e0) / 1000
    images = n * batch
    return {"energy_j_per_img": joules / images, "energy_net_j_per_img": (joules - idle_w * elapsed) / images,
            "avg_power_w": joules / elapsed, "idle_power_w": idle_w, "sustained_img_s": images / elapsed,
            "sm_clock_mhz": float(np.mean(clocks))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16", "int8", "fp8"])
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=[8, 6])
    args = ap.parse_args()
    cfg = load_config()
    b = cfg["benchmark"]
    spec_b = importlib.util.spec_from_file_location("bench04", Path(__file__).with_name("04_benchmark.py"))
    bench04 = importlib.util.module_from_spec(spec_b)
    spec_b.loader.exec_module(bench04)            # same real-image input as the latency benchmark
    out = results_dir(cfg, "tables") / "energy_graph.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["model"], r["precision"], r["batch"], r["repeat"]) for r in rows}
    monitor = GPUMonitor()
    engine_dir = results_dir(cfg, "engines")
    for name in args.models:
        spec = SPECS[name]
        for bs in args.batch_sizes:
            for precision in args.precisions:
                path = engine_dir / f"{name}_{precision}_bs{bs}.engine"
                if not path.exists():
                    print("missing", path.name, flush=True)
                    continue
                model = TRTModel(path)
                x = bench04.sample_input(cfg, name, bs)
                inputs = {spec.input_name: x.to(model.dtype(spec.input_name)).contiguous()}
                for rep in range(b["repeats"]):
                    if (name, precision, bs, rep) in done:
                        continue
                    idle = idle_power_w(monitor, b["idle_seconds"])
                    row = {"model": name, "precision": precision, "batch": bs, "repeat": rep,
                           **graph_energy(model, inputs, monitor, b["energy_seconds"], bs, idle)}
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(out, index=False)
                    print(f"{name:16s} {precision:5s} bs{bs} rep{rep}: {1000 * row['energy_j_per_img']:.2f} mJ/img, "
                          f"{row['sustained_img_s']:.0f} img/s, {row['avg_power_w']:.0f} W", flush=True)
                del model
                torch.cuda.empty_cache()
