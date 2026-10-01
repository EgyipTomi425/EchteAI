"""Check the static analysis (43_static_analysis.py) and the INT8 prediction (42_predict_int8.py) on a model and a
small labelled image folder, with simulated quantization on the CPU (no GPU, no TensorRT).

Quantizers are placed where the static analysis models them: on every batch-normalised (and activated) tensor,
one symmetric per-tensor scale each, calibrated on --n-calib images with the MSE-optimal range of Eq. (5) (INT8) or
the maximum (FP8 E4M3); weights are quantized per output channel. On --n-eval further images the script measures

  * the exact injected SQNR of every quantizer (FP32 activations, calibrated scale), compared with the static
    prediction from the BN parameters alone
  * SQNR_add from these measured values and from the static analysis (Eq. (16), unit propagation factors)
  * the head-input SQNR (input of the last linear layer) of the simulated INT8 and FP8 model, and Gamma_bar
  * top-1 accuracy of FP32, INT8 and FP8 and the relative loss, compared with the relative loss predicted from
    the head-input SQNR by the calibration of the article (tables/prediction.csv, five TensorRT networks)

The image folder holds one sub-folder per class named by the ImageNet class index (0 ... 999), as ImageNetV2;
without labels (--no-labels) only the SQNR quantities are reported.

  python scripts/44_validate_static.py --torchvision mobilenet_v2 --images data/imagenetv2/imagenetv2-matched-frequency-format-val
  python scripts/44_validate_static.py --torchvision resnet50 --images <folder> --n-calib 128 --n-eval 2000

The simulated placement (BN outputs only) and calibration differ from the TensorRT toolchain of the article, so the
check tests the method, not the deployment numbers of a particular runtime.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from pepai.config import CODE_ROOT, load_config, results_dir

_spec = importlib.util.spec_from_file_location("static", Path(__file__).with_name("43_static_analysis.py"))
static = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(static)

IDEMPOTENT = {"relu", "relu6", "identity"}
TORCH_ACT = {"relu": torch.relu, "relu6": lambda z: z.clamp(0, 6), "identity": lambda z: z}


def t_int8(x, s):
    return torch.clamp(torch.round(x / s), -static.INT8_LEVELS, static.INT8_LEVELS) * s


def t_e4m3(x, s):
    v = x / s
    a = v.abs().clamp(max=static.E4M3_MAX)
    e = torch.floor(torch.log2(a.clamp(min=2.0 ** -6)))
    step = torch.pow(2.0, e - 3)
    return torch.sign(v) * torch.clamp(torch.round(a / step) * step, max=static.E4M3_MAX) * s


def load_images(root, n, seed, labels=True):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))[:n]]
    y = [int(p.parent.name) for p in files] if labels else [-1] * len(files)
    return files, y


def batches(files, preprocess, size=32):
    for i in range(0, len(files), size):
        yield torch.stack([preprocess(Image.open(p).convert("RGB")) for p in files[i:i + size]])


def quant_sites(model):
    """(name, module to hook, activation) for every BN-modelled tensor, as in 43_static_analysis.py."""
    leaves = static.leaf_modules(model)
    sites = []
    for i, (name, m) in enumerate(leaves):
        if not static.is_bn(m):
            continue
        act = static.following_activation(leaves, i, name, model)
        if act in IDEMPOTENT or i + 1 >= len(leaves):
            sites.append((name, m, act, "bn"))           # quantize act(BN output); a later ReLU is then a no-op
        else:
            sites.append((name, leaves[i + 1][1], act, "act"))
    return sites


class Simulator:
    def __init__(self, model, sites):
        self.model, self.sites = model, sites
        self.mode, self.scales, self.fmt = "fp32", {}, "int8"
        self.samples, self.energy = {}, {}
        self.head = None
        for name, mod, act, where in sites:
            mod.register_forward_hook(self._hook(name, act, where))
        last_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
        last_linear.register_forward_hook(lambda m, i, o: setattr(self, "head", i[0].detach().flatten(1)))

    def _hook(self, name, act, where):
        fn = TORCH_ACT.get(act, lambda z: z)

        def hook(module, inp, out):
            x = fn(out) if where == "bn" else out
            ret = None if where == "act" or act == "identity" else x
            if self.mode == "collect":
                v = x.detach().flatten()
                idx = torch.randint(0, v.numel(), (min(v.numel(), 20000),),
                                    generator=torch.Generator().manual_seed(len(self.samples.get(name, []))))
                self.samples.setdefault(name, []).append(v[idx].double().numpy())
                self.energy.setdefault(name, [0.0, 0.0])
                self.energy[name][0] = max(self.energy[name][0], float(v.abs().max()))
                return ret
            if self.mode in ("measure", "quant"):
                s = self.scales[name]
                q = t_int8(x, s) if self.fmt == "int8" else t_e4m3(x, s)
                if self.mode == "quant":
                    return q
                e = self.energy.setdefault(name, [0.0, 0.0])
                e[0] += float((x.double() ** 2).sum())
                e[1] += float(((q - x).double() ** 2).sum())
            return ret
        return hook

    def calibrate(self, fmt):
        scales = {}
        for name, _, _, _ in self.sites:
            v = np.concatenate(self.samples[name]).astype(np.float64)
            amax = self.energy[name][0]
            if fmt == "fp8":
                scales[name] = amax / static.E4M3_MAX
                continue
            cands = amax * np.geomspace(0.05, 1.0, 60)
            errs = [np.mean((static.int8_quant(v, a / static.INT8_LEVELS) - v) ** 2) for a in cands]
            scales[name] = float(cands[int(np.argmin(errs))]) / static.INT8_LEVELS
        return scales


def quantize_weights(model, fmt):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data.double().numpy()
            flat = w.reshape(w.shape[0], -1)
            amax = np.maximum(np.abs(flat).max(axis=1, keepdims=True), 1e-12)
            q = static.int8_quant(flat, amax / static.INT8_LEVELS) if fmt == "int8" else \
                static.e4m3_quant(flat, amax / static.E4M3_MAX)
            m.weight.data = torch.from_numpy(q.reshape(w.shape)).float()


def head_sqnr(f, q):
    """Per-image SQNR of the head input, median over images (dB)."""
    num = (f.double() ** 2).sum(1)
    den = ((q.double() - f.double()) ** 2).sum(1).clamp_min(1e-30)
    return float(np.median(10 * np.log10((num / den).numpy())))


def run(model_fn, name, images, n_calib, n_eval, seed, labels, preprocess):
    calib, _ = load_images(images, n_calib, seed + 1, labels=False)
    evalf, y = load_images(images, n_calib + n_eval, seed + 1, labels)
    evalf, y = evalf[n_calib:], np.array(y[n_calib:])
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(seed)

    base = model_fn().eval()
    layers, acts, bns, extra = static.analyse(base, rng)
    s_static = static.summarise(name, layers, acts, bns, extra)

    results = {"model": name, "n_calib": n_calib, "n_eval": len(evalf)}
    sim = Simulator(base, quant_sites(base))
    logits32, heads32 = [], []
    for xb in batches(evalf, preprocess):
        logits32.append(base(xb)); heads32.append(sim.head)
    logits32, heads32 = torch.cat(logits32), torch.cat(heads32)
    sim.mode = "collect"
    for xb in batches(calib, preprocess):
        base(xb)
    per_site = []
    for fmt in ("int8", "fp8"):
        m = model_fn().eval()
        s = Simulator(m, quant_sites(m))
        s.samples, s.energy = sim.samples, {k: [v[0], 0.0] for k, v in sim.energy.items()}
        s.scales, s.fmt = s.calibrate(fmt), fmt
        # exact injected SQNR of every quantizer on FP32 activations (first 64 evaluation images)
        s.mode, s.energy = "measure", {}
        for xb in batches(evalf[:64], preprocess):
            m(xb)
        inj = {k: 10 * np.log10(v[0] / v[1]) for k, v in s.energy.items() if v[1] > 0}
        s.mode = "quant"
        quantize_weights(m, fmt)
        logits, heads = [], []
        for xb in batches(evalf, preprocess):
            logits.append(m(xb)); heads.append(s.head)
        logits, heads = torch.cat(logits), torch.cat(heads)
        wcol = "w_int8_per_channel_db" if fmt == "int8" else "w_fp8_db"
        w_terms = [10 ** (-v / 10) for v in layers[wcol].values]
        a_terms = [10 ** (-v / 10) for v in inj.values()]
        add = -10 * np.log10(np.sum(a_terms) + np.sum(w_terms))
        results[f"{fmt}_sqnr_add_activations_db"] = -10 * np.log10(np.sum(a_terms))
        results[f"{fmt}_sqnr_add_weights_db"] = -10 * np.log10(np.sum(w_terms))
        results[f"{fmt}_sqnr_add_measured_db"] = add
        results[f"{fmt}_sqnr_head_db"] = head_sqnr(heads32, heads)
        results[f"{fmt}_gamma_bar"] = 10 ** ((add - results[f"{fmt}_sqnr_head_db"]) / 10)
        results[f"{fmt}_injected_median_db"] = float(np.median(list(inj.values())))
        if labels:
            results["fp32_top1"] = float((logits32.argmax(1).numpy() == y).mean())
            results[f"{fmt}_top1"] = float((logits.argmax(1).numpy() == y).mean())
            results[f"{fmt}_rel_loss"] = (results["fp32_top1"] - results[f"{fmt}_top1"]) / results["fp32_top1"]
            results[f"{fmt}_top1_agreement"] = float((logits.argmax(1) == logits32.argmax(1)).float().mean())
            c32, cq = logits32.argmax(1).numpy() == y, logits.argmax(1).numpy() == y
            brng = np.random.default_rng(seed)
            boots = []
            for _ in range(1000):     # paired bootstrap over the evaluation images
                idx = brng.integers(0, len(y), len(y))
                boots.append((c32[idx].mean() - cq[idx].mean()) / c32[idx].mean())
            results[f"{fmt}_rel_loss_lo"], results[f"{fmt}_rel_loss_hi"] = map(float, np.percentile(boots, [2.5, 97.5]))
        col = "sqnr_int8_db" if fmt == "int8" else "sqnr_fp8_db"
        for _, r in acts.iterrows():
            if r.tensor in inj:
                per_site.append({"model": name, "format": fmt, "tensor": r.tensor, "static_db": r[col],
                                 "measured_db": inj[r.tensor], "static_kappa": r.kappa})
    ps = pd.DataFrame(per_site)
    for fmt in ("int8", "fp8"):
        d = ps[ps.format == fmt]
        results[f"{fmt}_static_vs_measured_median_diff_db"] = float((d.static_db - d.measured_db).median())
        results[f"{fmt}_static_vs_measured_mae_db"] = float((d.static_db - d.measured_db).abs().mean())
        results[f"{fmt}_static_vs_measured_spearman"] = float(d.static_db.rank().corr(d.measured_db.rank()))
    for fmt, wcol in (("int8", "w_int8_per_channel_db"), ("fp8", "w_fp8_db")):
        a_db = s_static[f"static_sqnr_add_{fmt}_db"]
        w_terms = np.sum(10 ** (-layers[wcol].values / 10))
        results[f"static_sqnr_add_{fmt}_activations_db"] = a_db
        results[f"static_sqnr_add_{fmt}_db"] = float(-10 * np.log10(10 ** (-a_db / 10) + w_terms))
    # prediction of the relative INT8 loss from the head-input SQNR (calibration of the article)
    pred_p = results_dir(load_config(), "tables") / "prediction.csv"
    if not pred_p.exists():
        pred_p = CODE_ROOT / "reference_results" / "tables" / "prediction.csv"
    cal = pd.read_csv(pred_p)
    b, a = np.polyfit(cal.sqnr_head_db, np.log10(cal.int8_rel_loss), 1)
    for fmt in ("int8", "fp8"):
        results[f"{fmt}_rel_loss_predicted"] = float(10 ** (a + b * results[f"{fmt}_sqnr_head_db"]))
        if labels:
            lo, hi = results[f"{fmt}_rel_loss_lo"], results[f"{fmt}_rel_loss_hi"]
            results[f"{fmt}_prediction_within_ci"] = bool(lo <= results[f"{fmt}_rel_loss_predicted"] <= hi)
    return results, ps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--torchvision", help="torchvision classification model (default weights)")
    g.add_argument("--module", help="a whole PyTorch model saved with torch.save(model, path); 224x224 RGB input "
                                    "with ImageNet normalisation; its last nn.Linear is taken as the head")
    ap.add_argument("--images", help="folder with one sub-folder per ImageNet class index; without it only the "
                                     "static analysis is run")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--no-labels", action="store_true")
    args = ap.parse_args()
    import copy
    import torchvision
    cfg = load_config()
    if args.module:
        name = Path(args.module).stem
        loaded = torch.load(args.module, map_location="cpu", weights_only=False).float().eval()
        from torchvision import transforms as T
        preprocess = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        fn = lambda: copy.deepcopy(loaded)
    else:
        name = args.torchvision
        weights = torchvision.models.get_model_weights(args.torchvision).DEFAULT
        preprocess = weights.transforms()
        fn = lambda: torchvision.models.get_model(args.torchvision, weights=weights)
    if not args.images:
        torch.set_grad_enabled(False)
        layers, acts, bns, extra = static.analyse(fn().eval(), np.random.default_rng(cfg["seed"]))
        summary = static.summarise(name, layers, acts, bns, extra)
        static.report(summary, layers, acts, bns)
        static.report_scenarios(summary, layers, acts)
        print("\nno --images: static analysis only (nothing executed); pass an image folder to measure the error")
        return
    res, ps = run(fn, name, args.images, args.n_calib, args.n_eval, cfg["seed"], not args.no_labels,
                  preprocess)
    out = results_dir(cfg, "tables", "static")
    ps.to_csv(out / f"{name}_validation_sites.csv", index=False)
    p = results_dir(cfg, "tables") / "static_validation.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if len(old):
        old = old[old.model != name]
    pd.concat([old, pd.DataFrame([res])]).to_csv(p, index=False)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
