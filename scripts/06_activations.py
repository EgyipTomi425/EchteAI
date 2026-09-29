"""Layer-wise activation deviation FP32 vs INT8 (and FP16) on 500 evaluation images per model."""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pepai import debug
from pepai.activations import (analysed_tensors, final_tensors, region_errors, sample_pairs,
                               tensor_metrics)
from pepai.config import load_config, results_dir
from pepai.inputs import analysis_inputs
from pepai.trt import TRTModel

N_PAIRS = 128        # hexbin samples per tensor per image
N_MAPS = 8           # images for which spatial maps are stored


def spatial_maps(outs, infos):
    """Channel-max projections of all Conv outputs averaged at the largest resolution (CITDS style)."""
    convs = [t.name for t in infos if t.op == "Conv" and outs[t.name].dim() == 4]
    size = max((outs[n].shape[-2:] for n in convs), key=lambda s: s[0] * s[1])
    acc = torch.zeros(size, device="cuda")
    for n in convs:
        a = outs[n][0].float().amax(0)[None, None]
        acc += torch.nn.functional.interpolate(a, size=size, mode="bicubic", align_corners=False)[0, 0]
    return (acc / len(convs)).cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--quant", nargs="*", default=["int8fp32", "fp16"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--condition", help="analyse corrupted images (see 08_robustness.py)")
    ap.add_argument("--severity", type=int)
    args = ap.parse_args()
    cfg = load_config()
    n = args.n or cfg["coco"]["n_activation"]
    out_dir = results_dir(cfg, "activations")
    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])

    for name in args.models or cfg["models"]:
        onnx_dir = results_dir(cfg, "onnx")
        for quant in args.quant:
            infos = analysed_tensors(onnx_dir / f"{name}_fp32.onnx", onnx_dir / f"{name}_{quant}.onnx")
            final = set(final_tensors(cfg, name, onnx_dir / f"{name}_fp32.onnx", infos))
            ref = TRTModel(debug.ensure_engine(cfg, name, "fp32", "full"))
            qnt = TRTModel(debug.ensure_engine(cfg, name, quant, "full"))
            (out_dir / f"{name}_{quant}_tensors.json").write_text(json.dumps(
                [{"name": t.name, "op": t.op, "node": t.node, "order": t.order, "inputs": t.inputs,
                  "final": t.name in final} for t in infos], indent=1))

            rows, pairs, maps = [], [], {}
            image_dir, tag = None, f"{name}_{quant}"
            if args.condition:
                image_dir = cfg["coco"]["root"] / "coco_c" / args.condition / str(args.severity)
                tag += f"_{args.condition}{args.severity}"
            for k, (key, x, mask) in enumerate(tqdm(analysis_inputs(cfg, name, n, image_dir), total=n,
                                                    desc=tag)):
                fo = {t: v.clone() for t, v in ref(images=x.to(ref.dtype("images")).contiguous()).items()}
                qo = qnt(images=x.to(qnt.dtype("images")).contiguous())
                for t in infos:
                    f, q = fo[t.name], qo[t.name]
                    row = {"image": key, "tensor": t.name, "op": t.op, "order": t.order,
                           "final": t.name in final, "numel": f.numel()}
                    row.update(tensor_metrics(f, q))
                    if mask is not None:
                        row.update(region_errors(f, q, mask) or {})
                    rows.append(row)
                    if quant.startswith("int8"):
                        pairs.append(sample_pairs(f, q, N_PAIRS, gen))
                if k < N_MAPS and quant.startswith("int8"):
                    maps[f"{key}_fp32"] = spatial_maps(fo, infos)
                    maps[f"{key}_{quant}"] = spatial_maps(qo, infos)
                    maps[f"{key}_input"] = x[0].float().cpu().numpy()
                    if mask is not None:
                        maps[f"{key}_mask"] = mask.cpu().numpy()
            pd.DataFrame(rows).to_csv(out_dir / f"{tag}.csv.gz", index=False)
            if pairs and not args.condition:
                np.save(out_dir / f"{tag}_pairs.npy", torch.cat(pairs).numpy())
                np.savez_compressed(out_dir / f"{tag}_maps.npz", **maps)
            del ref, qnt
            torch.cuda.empty_cache()
