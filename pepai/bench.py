"""Latency, jitter and energy measurement of TensorRT engines on the GPU."""
import time

import numpy as np
import pynvml
import torch


class GPUMonitor:
    def __init__(self, index=0):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)

    def energy_mj(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)

    def sm_clock(self):
        return pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)

    def temperature(self):
        return pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)

    def info(self):
        return {
            "name": pynvml.nvmlDeviceGetName(self.h),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(self.h) / 1000,
        }


def latency(model, inputs, warmup, iters, cuda_graph=False):
    """Per-inference latency in ms, measured with CUDA events around every execution."""
    model.bind(**inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            model.execute()
        stream.synchronize()

        run = model.execute
        if cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                model.execute()
            run = graph.replay

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for s, e in zip(starts, ends):
            s.record(stream)
            run()
            e.record(stream)
        stream.synchronize()
    return np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])


def summarize_latency(ms, batch):
    return {
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
        "std_ms": float(ms.std()),
        "jitter_p99_p50_ms": float(np.percentile(ms, 99) - np.percentile(ms, 50)),
        "throughput_img_s": float(1000.0 * batch / ms.mean()),
    }


def idle_power_w(monitor, seconds):
    torch.cuda.synchronize()
    e0, t0 = monitor.energy_mj(), time.perf_counter()
    time.sleep(seconds)
    return (monitor.energy_mj() - e0) / 1000 / (time.perf_counter() - t0)


def energy(model, inputs, monitor, seconds, batch, idle_w):
    """Energy per image from the NVML energy counter over a sustained back-to-back run."""
    model.bind(**inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(20):
            model.execute()
        stream.synchronize()
        n, clocks, temps = 0, [], []
        e0, t0 = monitor.energy_mj(), time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            for _ in range(20):
                model.execute()
            n += 20
            stream.synchronize()
            clocks.append(monitor.sm_clock())
            temps.append(monitor.temperature())
        elapsed = time.perf_counter() - t0
        joules = (monitor.energy_mj() - e0) / 1000
    images = n * batch
    return {
        "energy_j_per_img": joules / images,
        "energy_net_j_per_img": (joules - idle_w * elapsed) / images,
        "avg_power_w": joules / elapsed,
        "idle_power_w": idle_w,
        "sustained_img_s": images / elapsed,
        "sm_clock_mhz": float(np.mean(clocks)),
        "temp_c_max": float(np.max(temps)),
    }


ENERGY_COLUMNS = ["energy_j_per_img", "energy_net_j_per_img", "avg_power_w", "idle_power_w", "sustained_img_s",
                  "sm_clock_mhz"]


def benchmark_medians(tables_dir, files=("benchmark.csv", "benchmark_bs6.csv", "benchmark_extra.csv")):
    """Median over repeats per (model, precision, batch) of every benchmark table.

    Latency always comes from the CUDA-graph runs of 04_benchmark.py. Energy columns are taken from
    energy_graph.csv (36_energy_graph.py, CUDA-graph replays, i.e. the execution mode of the reported latencies)
    wherever it covers an engine, and from the direct-enqueue runs of 04_benchmark.py otherwise; the column
    energy_mode records which one was used."""
    import pandas as pd
    frames = [pd.read_csv(tables_dir / f) for f in files if (tables_dir / f).exists()]
    b = pd.concat(frames).groupby(["model", "precision", "batch"]).median(numeric_only=True)
    b["energy_mode"] = "enqueue"
    g_path = tables_dir / "energy_graph.csv"
    if g_path.exists():
        g = pd.read_csv(g_path).groupby(["model", "precision", "batch"]).median(numeric_only=True)
        common = b.index.intersection(g.index)
        for c in ENERGY_COLUMNS:
            if c in g.columns:
                b.loc[common, c] = g.loc[common, c]
        b.loc[common, "energy_mode"] = "graph"
    return b
