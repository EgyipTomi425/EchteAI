"""Activation-level comparison of FP32 and quantized engines (PEP-AI 'precise' component).

Intermediate tensors are captured from TensorRT debug engines in which the analysed tensors are
marked as outputs. The reference is a strict-FP32 engine, the quantized side is the INT8 QDQ model
with FP32 high-precision ops, so the measured deviation is caused by quantization alone.
"""
from dataclasses import dataclass

import onnx
import torch

from pepai import models

# Op types whose outputs are analysed. BatchNormalization only appears where it cannot be folded
# into a preceding convolution (e.g. DenseNet transition inputs).
ANALYSED_OPS = {"Conv", "Relu", "Add", "MaxPool", "AveragePool", "Mul", "Sigmoid", "Concat",
                "Resize", "BatchNormalization", "GlobalAveragePool", "Split", "Clip"}


@dataclass
class TensorInfo:
    name: str
    op: str           # producing op type
    node: str         # producing node name
    order: int        # topological position of the producing node
    inputs: list      # analysed tensors consumed by the producing node


def analysed_tensors(fp32_onnx, quant_onnx):
    """Feature tensors present (same name, same producer op) in both graphs, in topological order.

    Graph outputs that are not 4D feature maps (e.g. YOLOv10's list of 300 detections, whose order
    differs between precisions) are not activations and are excluded.
    """
    fp = onnx.load(str(fp32_onnx), load_external_data=False).graph
    q = onnx.load(str(quant_onnx), load_external_data=False).graph
    non_feature_outputs = {o.name for o in fp.output if len(o.type.tensor_type.shape.dim) != 4}
    q_nodes = {o: n for n in q.node for o in n.output}

    def q_producer(tensor):
        """Producer op type; a Cast inserted by FP16 conversion (keep_io_types) is looked through."""
        n = q_nodes.get(tensor)
        if n is not None and n.op_type == "Cast" and n.input[0] in q_nodes:
            return q_nodes[n.input[0]].op_type
        return n.op_type if n is not None else None

    q_producers = {o: q_producer(o) for o in q_nodes}
    infos, known = [], set()
    for order, node in enumerate(fp.node):
        if node.op_type not in ANALYSED_OPS:
            continue
        for out in node.output:
            if q_producers.get(out) != node.op_type or out in non_feature_outputs:
                continue
            infos.append(TensorInfo(out, node.op_type, node.name, order,
                                    [i for i in node.input if i in known]))
            known.add(out)
    return infos


def channel_max_projection(t):
    """A(x, y) = max_c F(c, x, y) for a (1, C, H, W) tensor (CITDS Eq. 1)."""
    return t[0].amax(dim=0)


@torch.no_grad()
def tensor_metrics(f, q, eta=1e-6):
    """Deviation statistics of a quantized tensor q against its FP32 reference f."""
    f = f.float()
    q = q.float()
    if f.abs().max() == 0:          # identically zero reference: relative metrics are undefined
        return {k: float("nan") for k in ("abs_mean", "abs_median", "mre_elem", "rel_l2", "sqnr_db", "cosine",
                                          "mre_proj")}
    e = q - f
    abs_e = e.abs()
    rel = abs_e / (f.abs() + eta)
    energy_f = f.pow(2).sum()
    energy_e = e.pow(2).sum().clamp_min(1e-30)
    m = {
        "abs_mean": abs_e.mean().item(),
        "abs_median": median(abs_e),
        "mre_elem": median(rel),
        "rel_l2": (energy_e.sqrt() / energy_f.sqrt().clamp_min(1e-30)).item(),
        "sqnr_db": (10 * torch.log10(energy_f.clamp_min(1e-30) / energy_e)).item(),
        "cosine": torch.nn.functional.cosine_similarity(f.flatten(), q.flatten(), dim=0).item(),
    }
    if f.dim() == 4:
        af, aq = channel_max_projection(f), channel_max_projection(q)
        m["mre_proj"] = median((aq - af).abs() / (af.abs() + eta))
    return m


def median(t):
    """Exact median of a large tensor (torch.quantile is limited to 16M elements)."""
    flat = t.flatten()
    k = (flat.numel() - 1) // 2
    return flat.kthvalue(k + 1).values.item()


@torch.no_grad()
def region_errors(f, q, mask, eta=1e-6):
    """Median relative error of the channel-max projection inside / outside an object mask.

    mask: (H0, W0) bool tensor at input resolution; resized to the tensor resolution.
    """
    if f.dim() != 4:
        return None
    af, aq = channel_max_projection(f.float()), channel_max_projection(q.float())
    m = torch.nn.functional.interpolate(mask[None, None].float(), size=af.shape, mode="nearest")[0, 0] > 0.5
    rel = (aq - af).abs() / (af.abs() + eta)
    if m.sum() == 0 or (~m).sum() == 0:
        return None
    return {"mre_proj_object": median(rel[m]), "mre_proj_background": median(rel[~m])}


@torch.no_grad()
def sample_pairs(f, q, n, generator):
    """Random (reference value, deviation) pairs for the aggregated hexbin plots."""
    f = f.float().flatten()
    idx = torch.randint(0, f.numel(), (n,), device=f.device, generator=generator)
    return torch.stack([f[idx], (q.float().flatten()[idx] - f[idx])], dim=1).cpu()


def final_tensors(cfg, name, fp32_onnx, infos):
    """Feature tensors handed to the task head: FPN outputs, YOLO Detect inputs, pre-pooling features."""
    names = {t.name for t in infos}
    graph = onnx.load(str(fp32_onnx), load_external_data=False).graph
    if name == "frcnn_r50_fpn":
        out = list(models.FRCNN_OUTPUTS)
    elif name.startswith("yolov10"):
        detect_from = models.load_yolo(name, cfg["paths"]["data"] / "weights").model.model[-1].f
        out = []
        for i in detect_from:
            nodes = [n for n in graph.node if n.name.startswith(f"/model.{i}/")]
            out.append(nodes[-1].output[0])
    else:
        # The last global pooling feeds the classifier (earlier ones sit in squeeze-excitation blocks).
        pool = [n for n in graph.node if n.op_type in ("GlobalAveragePool", "ReduceMean")][-1]
        out = [pool.input[0]]
    missing = [o for o in out if o not in names]
    assert not missing, f"final tensors not analysed: {missing}"
    return out


def load_activation_table(path, usecols=None):
    """Activation CSV without undefined rows (identically zero references) and non-feature outputs."""
    import pandas as pd
    if usecols is not None:         # the filter columns are always needed
        usecols = sorted(set(usecols) | {"tensor", "sqnr_db", "cosine", "abs_mean", "mre_proj"})
    df = pd.read_csv(path, usecols=usecols)
    if "tensor" in df:
        df = df[df.tensor != "output0"]
    if "sqnr_db" in df and "cosine" in df:
        df = df[~((df.cosine == 0) & (df.get("abs_mean", 0) == 0))]
    if "mre_proj" in df:            # only 4D feature maps have a projection; drops 1D shape tensors
        df = df[df.mre_proj.notna()]
    return df.dropna(subset=[c for c in ("sqnr_db",) if c in df])


def resolved_inputs(fp32_onnx, analysed_names):
    """For every node, the nearest analysed ancestor tensors of each input.

    Nodes such as the PSA positional-encoding conv or the DFL conv read their input through
    Reshape / Transpose / Softmax, which are not analysed; resolving through them gives the true
    upstream deviation. Returns {node name: [analysed tensor names]}; empty only for graph inputs.
    """
    graph = onnx.load(str(fp32_onnx), load_external_data=False).graph
    analysed = set(analysed_names)
    ancestors = {}          # tensor -> set of nearest analysed ancestors
    for node in graph.node:
        for out in node.output:
            if out in analysed:
                ancestors[out] = {out}
            else:
                ancestors[out] = set().union(*[ancestors.get(i, set()) for i in node.input]) if node.input else set()
    return {node.name: sorted(set().union(*[ancestors.get(i, set()) for i in node.input]))
            for node in graph.node}
