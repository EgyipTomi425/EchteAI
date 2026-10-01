"""PEP-AI-guided selective quantization: keep the k most error-injecting Conv nodes in FP16.

The base configuration (k = 0) is calibrated once; every variant is derived from it by removing the
quantizers of the excluded nodes (pepai.quant.exclude_from_quantized), so all variants share scales.

Strategies
  pepai       ranking from the 500-image activation analysis of the fully quantized model
  pepai_iter  greedy: exclude the top node(s), re-measure the new model, repeat
  random      seeded random subsets of the same sizes (control)
  depthwise   all depthwise convolutions (architecture heuristic, control)
  noise       convolutions ranked by the noise injected at their activation inputs, predicted from FP32
              statistics alone (26_quantizer_snr.py; Proposition 3 with unit propagation factors)
"""
import argparse
import hashlib
import random
import shutil

import pandas as pd
import torch

from pepai import data
from pepai.bench import GPUMonitor, energy, idle_power_w, latency, summarize_latency
from pepai.config import load_config, results_dir
from pepai.evaluate import classify, coco_map, detect_frcnn, detect_yolo
from pepai.models import SPECS, engine_shapes
from pepai.quant import (calibration_inputs, exclude_from_quantized, excluded_nodes, excluded_op_types,
                         int8_with_fp16, to_int8)
from pepai.sensitivity import conv_nodes, conv_sensitivity, measure_and_rank
from pepai.trt import TRTModel, build_engine


def fp_convs(path):
    """Number of convolutions that run in floating point (activation input not dequantized)."""
    import onnx
    g = onnx.load(str(path), load_external_data=False).graph
    producer = {o: n for n in g.node for o in n.output}
    return sum(n.op_type == "Conv" and producer.get(n.input[0]) is not None
               and producer[n.input[0]].op_type != "DequantizeLinear" for n in g.node)


def base_configuration(cfg, name):
    """Calibration method and extra excluded op types chosen label-free (20_label_free_selection.py,
    output-consistency criterion); the default configuration if no selection exists yet."""
    sel = results_dir(cfg, "tables") / "label_free_selection.csv"
    abl = results_dir(cfg, "tables") / "placement_ablation.csv"
    if not (sel.exists() and abl.exists()):
        return "default", "entropy", []
    g = pd.read_csv(sel)
    g = g[g.model == name]
    if g.empty:
        return "default", "entropy", []
    variant = g.loc[g.consistency.idxmax()].variant
    row = pd.read_csv(abl).query("model == @name and variant == @variant").iloc[0]
    ops = [] if pd.isna(row.excluded_ops) or not row.excluded_ops else str(row.excluded_ops).split(";")
    return variant, row.calibration, ops


class Variants:
    """Builds INT8 variants with a given set of excluded nodes and evaluates them."""

    def __init__(self, cfg, name, limit, probe_images=32):
        self.cfg, self.name, self.limit, self.spec = cfg, name, limit, SPECS[name]
        self.probe_images = probe_images
        self.monitor = GPUMonitor()
        self.base, self.method, self.base_ops = base_configuration(cfg, name)
        self.out = results_dir(cfg, "selective")
        self.fp32 = results_dir(cfg, "onnx") / f"{name}_fp32.onnx"
        self._calib = None

    def calib(self):
        if self._calib is None:
            self._calib = calibration_inputs(self.cfg, self.name)
        return self._calib

    def onnx(self, tag, excluded):
        # The excluded node list is part of the file name: a changed ranking never reuses old models.
        if not excluded:
            tag = "pepai_k0"        # one calibrated base model for every strategy
        digest = hashlib.sha1(";".join(sorted(excluded)).encode()).hexdigest()[:8]
        tag = f"{self.base}_{tag}_{digest}"
        q32 = self.out / f"{self.name}_{tag}_int8fp32.onnx"
        q16 = self.out / f"{self.name}_{tag}_int8.onnx"
        if not q16.exists():
            if excluded:    # same scales as the fully quantized base model; only the excluded nodes differ
                effective = exclude_from_quantized(self.base_onnx(), q32, list(excluded))
                if len(effective) > len(excluded):
                    print(f"closure: also excluded {effective[len(excluded):]}", flush=True)
            elif self.base == "default":    # the deployed INT8 model is the default base
                shutil.copy(results_dir(self.cfg, "onnx") / f"{self.name}_int8fp32.onnx", q32)
            else:
                to_int8(self.fp32, q32, self.calib(), method=self.method,
                        nodes_to_exclude=excluded_nodes(self.cfg, self.name),
                        op_types_to_exclude=excluded_op_types(self.cfg, self.name) + self.base_ops)
            int8_with_fp16(q32, q16)
        return q32, q16

    def base_onnx(self):
        """The fully quantized (k = 0) INT8+FP32 model of the base configuration, calibrated once."""
        return self.onnx("pepai_k0", [])[0]

    def evaluate(self, tag, excluded, probe=True):
        q32, q16 = self.onnx(tag, excluded)
        engines = {}
        for bs in (1, 8):
            e = q16.with_name(f"{q16.stem}_bs{bs}.engine")
            if not e.exists():
                shape = (bs, *self.spec.bench_shape)
                shapes = (engine_shapes(self.spec) if (bs == 1 and self.spec.dynamic_hw)
                          else {self.spec.input_name: (shape,) * 3})
                build_engine(q16, e, shapes, timing_cache=results_dir(self.cfg, "engines") / "timing.cache")
            engines[bs] = e
        row = {"model": self.name, "k": len(excluded), "k_effective": fp_convs(q32) - fp_convs(self.onnx("pepai_k0", [])[0]),
               "base": self.base, **self.accuracy(engines)}
        for bs in (1, 8):
            if self.spec.dynamic_hw and bs == 1:
                continue
            m = TRTModel(engines[bs])
            x = torch.rand(bs, *self.spec.bench_shape, device="cuda").to(m.dtype(self.spec.input_name))
            inputs = {self.spec.input_name: x}
            row[f"p50_ms_bs{bs}"] = summarize_latency(latency(m, inputs, 50, 500, cuda_graph=True), bs)["p50_ms"]
            if bs == 8:
                idle = idle_power_w(self.monitor, 3)
                row["energy_mj_per_img_bs8"] = 1000 * energy(m, inputs, self.monitor, 5, bs, idle)["energy_j_per_img"]
            del m
        for e in engines.values():
            e.unlink()
        if probe:   # per-layer deviation of this variant (depth x energy x deviation surface)
            _, layers = measure_and_rank(self.cfg, self.name, q32, self.probe_images, self.out, return_layers=True)
            layers.assign(tag=tag, k=len(excluded)).to_csv(self.out / f"{self.name}_{tag}_layers.csv", index=False)
        return row

    def accuracy(self, engines):
        if self.spec.task == "cls":
            _, items = data.imagenetv2_split(self.cfg)
            p, lab = classify(engines[8], items[:self.limit], 8)
            return {"top1": float((p == lab).mean())}
        coco, ids = data.coco_val_ids(self.cfg, self.limit)
        det = detect_frcnn if self.name == "frcnn_r50_fpn" else detect_yolo
        return coco_map(coco, det(self.cfg, engines[1], coco, ids), ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["efficientnet_b0"])
    ap.add_argument("--strategies", nargs="*", default=["pepai", "pepai_iter", "random", "depthwise"])
    ap.add_argument("--ks", nargs="*", type=int, default=[0, 1, 2, 3, 5, 8, 12, 20, 30])
    ap.add_argument("--iter-ks", nargs="*", type=int, default=[1, 2, 3, 4, 5, 8, 11, 14, 17, 20])
    ap.add_argument("--random-ks", nargs="*", type=int, default=[1, 5, 12, 20])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--probe-images", type=int, default=64)
    ap.add_argument("--limit", type=int, help="evaluation images (default: all)")
    ap.add_argument("--min-drop", type=float, default=0.5,
                    help="skip models whose INT8 accuracy is within this many points of FP32")
    args = ap.parse_args()
    cfg = load_config()
    act_dir = results_dir(cfg, "activations")
    table = results_dir(cfg, "tables") / "selective.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    if rows and "strategy" not in rows[0]:
        for r in rows:
            r["strategy"], r["seed"] = "pepai", 0
    done = {(r["model"], r["strategy"], r["k"], r.get("seed", 0)) for r in rows}

    def record(row, strategy, seed=0, excluded=()):
        row.update({"strategy": strategy, "seed": seed, "excluded": ";".join(excluded)})
        rows.append(row)
        pd.DataFrame(rows).to_csv(table, index=False)
        print({k: v for k, v in row.items() if k != "excluded"}, flush=True)

    acc_path = results_dir(cfg, "tables") / "accuracy.csv"
    acc = pd.read_csv(acc_path).set_index(["model", "precision"]) if acc_path.exists() else None
    abl_path = results_dir(cfg, "tables") / "placement_ablation.csv"
    for name in args.models:
        v = Variants(cfg, name, args.limit)
        if acc is not None and (name, "int8") in acc.index:
            # Remaining loss of the base configuration: the deployed INT8 model, or the label-free
            # selected placement/calibration variant (evaluated on the same images by 17_placement_ablation).
            metric = "top1" if SPECS[name].task == "cls" else "mAP"
            base_acc = acc.loc[(name, "int8"), metric]
            if v.base != "default":
                base_acc = pd.read_csv(abl_path).query("model == @name and variant == @v.base").iloc[0][metric]
            drop = 100 * (acc.loc[(name, "fp32"), metric] - base_acc)
            if drop < args.min_drop:
                print(f"{name}: base '{v.base}' INT8 drop {drop:.2f} points < {args.min_drop}, "
                      "selective quantization skipped")
                continue
        convs = conv_nodes(v.fp32)
        if "pepai" in args.strategies:
            rank = conv_sensitivity(act_dir / f"{name}_int8fp32.csv.gz", act_dir / f"{name}_int8fp32_tensors.json",
                                    v.fp32, results_dir(cfg, "onnx") / f"{name}_int8fp32.onnx")
            rank.to_csv(v.out / f"{name}_sensitivity.csv", index=False)
            for k in args.ks:
                if (name, "pepai", k, 0) not in done:
                    excl = list(rank.node[:k])
                    record(v.evaluate(f"pepai_k{k}", excl), "pepai", 0, excl)
        if "noise" in args.strategies:
            q = pd.read_csv(results_dir(cfg, "tables") / "quantizer_snr.csv")
            q = q[(q.model == name) & q.upstream_of_head.astype(bool)]
            share = {}
            for _, r in q.iterrows():
                for u in str(r.consumers).split(";"):
                    share[u] = share.get(u, 0.0) + 10 ** (-r.sqnr_inj_db / 10)
            candidates = {c for c, _ in convs}
            order = [n for n, _ in sorted(share.items(), key=lambda kv: -kv[1]) if n in candidates]
            for k in args.ks:
                if (name, "noise", k, 0) not in done:
                    excl = order[:k]
                    record(v.evaluate(f"noise_k{k}", excl, probe=False), "noise", 0, excl)
        if "pepai_iter" in args.strategies:
            excluded = []
            for k in args.iter_ks:
                while len(excluded) < k:
                    q32, _ = v.onnx(f"iter_k{len(excluded)}", excluded)
                    rank = measure_and_rank(cfg, name, q32, args.probe_images, v.out)
                    rank = rank[~rank.node.isin(excluded)]
                    excluded.append(rank.node.iloc[0])
                if (name, "pepai_iter", k, 0) not in done:
                    record(v.evaluate(f"iter_k{k}", excluded), "pepai_iter", 0, excluded)
        if "random" in args.strategies:
            for seed in range(args.seeds):
                for k in args.random_ks:
                    if (name, "random", k, seed) not in done:
                        excl = random.Random(1000 + seed).sample([c for c, _ in convs], k)
                        record(v.evaluate(f"random_s{seed}_k{k}", excl, probe=False), "random", seed, excl)
        if "depthwise" in args.strategies:
            excl = [c for c, dw in convs if dw]
            if excl and (name, "depthwise", len(excl), 0) not in done:
                record(v.evaluate("depthwise", excl, probe=False), "depthwise", 0, excl)
