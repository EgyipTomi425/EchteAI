import torch
import logging
import time
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList
import torch.nn as nn
import onnxruntime as ort

class FeatureExtractor(nn.Module):
    def __init__(self, transform, backbone):
        super().__init__()
        self.transform = transform
        self.backbone = backbone

    def forward(self, images):
        original_image_sizes = torch.tensor([list(img.shape[-2:]) for img in images], dtype=torch.int64)
        img_list, _ = self.transform(images)
        features = self.backbone(img_list.tensors)
        feats = list(features.values())
        image_sizes = torch.tensor([list(s) for s in img_list.image_sizes], dtype=torch.int64)
        return (
            img_list.tensors,
            image_sizes,
            original_image_sizes,
            feats[0], feats[1], feats[2], feats[3], feats[4],
        )

class DetectorHead(nn.Module):
    def __init__(self, rpn, roi_heads, postprocess):
        super().__init__()
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.postprocess = postprocess

    def forward(self, tensors, image_sizes, original_image_sizes, feat0, feat1, feat2, feat3, feat4):
        feats = {"0": feat0, "1": feat1, "2": feat2, "3": feat3, "4": feat4}
        img_list = ImageList(tensors, image_sizes)
        proposals, _ = self.rpn(img_list, feats, targets=None)
        detections, _ = self.roi_heads(feats, proposals, image_sizes, targets=None)
        results = self.postprocess(detections, image_sizes, original_image_sizes)
        boxes = [r["boxes"] for r in results]
        labels = [r["labels"] for r in results]
        scores = [r["scores"] for r in results]
        return boxes, labels, scores

def split_frcnn_pipeline(model, images, device):
    logging.info("=== Comparison: full forward vs. manual pipeline (batch) ===")
    model.eval()
    images = [img.to(device) for img in images]
    fe = FeatureExtractor(model.transform, model.backbone).to(device)
    dh = DetectorHead(model.rpn, model.roi_heads, model.transform.postprocess).to(device)
    with torch.no_grad():
        t0 = time.time()
        full_output = model(images)
        t1 = time.time()
        t2 = time.time()
        tensors, image_sizes, orig_sizes, f0, f1, f2, f3, f4 = fe(images)
        manual_boxes, manual_labels, manual_scores = dh(tensors, image_sizes, orig_sizes, f0, f1, f2, f3, f4)
        t3 = time.time()
    logging.info(f"⏱️ Full forward time:     {(t1 - t0):.4f} s")
    logging.info(f"⏱️ Manual pipeline time:  {(t3 - t2):.4f} s")
    for i, full in enumerate(full_output):
        logging.info(f"\n📷 Image {i}:")
        logging.info(f"📦 Box count - Full: {len(full['boxes'])}, Manual: {len(manual_boxes[i])}")
        if len(full['boxes']) == len(manual_boxes[i]) and len(full['boxes']) > 0:
            box_diff = torch.abs(full['boxes'] - manual_boxes[i]).mean().item()
            score_diff = torch.abs(full['scores'] - manual_scores[i]).mean().item()
            label_diff = int((full['labels'] != manual_labels[i]).sum().item())
            logging.info(f"🔢 Avg. box difference:   {box_diff:.6f}")
            logging.info(f"🔢 Avg. score difference: {score_diff:.6f}")
            logging.info(f"❗ Label mismatches:      {label_diff}")
        else:
            logging.warning("⚠️ Box count mismatch or empty detections.")
            logging.warning(f"  Full boxes: {len(full['boxes'])}, Manual boxes: {len(manual_boxes[i])}")
    return fe, dh

def split_save_frcnn(model, images, device):
    logging.info("🔧 Preparing model and inputs")
    fe, dh = split_frcnn_pipeline(model, images, device)
    fe.eval(); dh.eval()

    torch.onnx.export(
        fe,
        (images,),
        "feature_extractor.onnx",
        input_names=["images"],
        output_names=["tensors", "image_sizes", "orig_sizes", "feat0", "feat1", "feat2", "feat3", "feat4"],
        opset_version=16
    )

    tensors, image_sizes, orig_sizes, f0, f1, f2, f3, f4 = fe(images)

    torch.onnx.export(
        dh,
        (tensors, image_sizes, orig_sizes, f0, f1, f2, f3, f4),
        "detector_head.onnx",
        input_names=["tensors", "image_sizes", "orig_sizes", "feat0", "feat1", "feat2", "feat3", "feat4"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={"feat3": {0: "batch", 2: "h", 3: "w"}},
        opset_version=16
    )

    logging.info("▶️ Running ONNX models")
    fe_sess = ort.InferenceSession("feature_extractor.onnx")
    dh_sess = ort.InferenceSession("detector_head.onnx")

    logging.info("🧠 FeatureExtractor (fe) model:")
    for inp in fe_sess.get_inputs():
        logging.info(f"🟩 FE input:  {inp.name:15} shape={inp.shape}")
    for out in fe_sess.get_outputs():
        logging.info(f"🟦 FE output: {out.name:15} shape={out.shape}")

    logging.info("🧠 DetectorHead (dh) model:")
    for inp in dh_sess.get_inputs():
        logging.info(f"🟩 DH input:  {inp.name:15} shape={inp.shape}")
    for out in dh_sess.get_outputs():
        logging.info(f"🟦 DH output: {out.name:15} shape={out.shape}")

    img_batch = np.stack([img.cpu().numpy() for img in images], axis=0)
    fe_outs = fe_sess.run(None, {"images": img_batch})
    fe_out_dict = {out.name: fe_outs[i] for i, out in enumerate(fe_sess.get_outputs())}

    dh_inputs = {}
    for inp in dh_sess.get_inputs():
        arr = fe_out_dict[inp.name]
        if inp.name in ("image_sizes", "orig_sizes"):
            arr = arr.reshape(-1, 2).astype(np.int64)
        dh_inputs[inp.name] = arr

    dh_outs = dh_sess.run(None, dh_inputs)
    bs = len(images)
    boxes  = dh_outs[0:bs]
    labels = dh_outs[bs:2*bs]
    scores = dh_outs[2*bs:3*bs]

    logging.info("🖼️ Detection results per image:")
    for i in range(bs):
        logging.info(f"📷 Image {i}: boxes={boxes[i].shape[0]} labels={labels[i].tolist()}")

    for out, meta in zip(dh_outs, dh_sess.get_outputs()):
        logging.info(f"📤 {meta.name}: {np.array(out).shape}")

class ONNXFasterRCNNWrapper(torch.nn.Module):
    def __init__(self, fe_onnx_path="feature_extractor.onnx", dh_onnx_path="detector_head.onnx", device='cpu'):
        super().__init__()
        self.device = device
        self.fe_session = ort.InferenceSession(fe_onnx_path, providers=['CPUExecutionProvider'])
        self.dh_session = ort.InferenceSession(dh_onnx_path, providers=['CPUExecutionProvider'])

    def forward(self, images):
        resized = [F.interpolate(img.unsqueeze(0), size=(375, 1242), mode="bilinear", align_corners=False).squeeze(0) for img in images]
        batch = torch.stack(resized).to(self.device)

        fe_outputs = self.fe_session.run(None, {"images": batch.cpu().numpy()})
        output_names = [o.name for o in self.fe_session.get_outputs()]
        fe_dict = dict(zip(output_names, fe_outputs))

        for k in ("image_sizes", "orig_sizes"):
            fe_dict[k] = fe_dict[k].reshape(-1, 2).astype(np.int64)

        dh_inputs = {inp.name: fe_dict[inp.name] for inp in self.dh_session.get_inputs()}
        dh_outputs = self.dh_session.run(None, dh_inputs)

        bs = len(images)
        boxes  = dh_outputs[0:bs]
        labels = dh_outputs[bs:2*bs]
        scores = dh_outputs[2*bs:3*bs]

        results = []
        for i in range(bs):
            result = {
                "boxes": torch.tensor(boxes[i], device=self.device),
                "labels": torch.tensor(labels[i], device=self.device),
                "scores": torch.tensor(scores[i], device=self.device),
            }
            results.append(result)
        return results
    
def compare_direct_vs_manual_pipeline(model, images, device):
    logging.info("=== Comparison: full forward vs. manual pipeline (batch) ===")
    model.eval()
    images = [img.to(device) for img in images]

    with torch.no_grad():
        t0 = time.time()
        full_output = model(images)
        t1 = time.time()

    with torch.no_grad():
        t2 = time.time()
        img_list, _ = model.transform(images)
        feats = model.backbone(img_list.tensors)
        if isinstance(feats, torch.Tensor):
            feats = OrderedDict([("0", feats)])
        elif isinstance(feats, (list, tuple)):
            feats = OrderedDict((str(i), f) for i, f in enumerate(feats))
        elif isinstance(feats, dict):
            feats = OrderedDict(feats)
        else:
            raise TypeError(f"Unsupported feature type: {type(feats)}")
        proposals, _ = model.rpn(img_list, feats, targets=None)
        detections, _ = model.roi_heads(feats, proposals, img_list.image_sizes, targets=None)
        orig_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        detections = model.transform.postprocess(detections, img_list.image_sizes, orig_sizes)
        t3 = time.time()

    logging.info(f"⏱️ Full forward time:  {(t1 - t0):.4f} s")
    logging.info(f"⏱️ Manual pipeline time: {(t3 - t2):.4f} s")

    for i, (full, manual) in enumerate(zip(full_output, detections)):
        logging.info(f"\n📷 Image {i}:")
        logging.info(f"📦 Box count - Full: {len(full['boxes'])}, Manual: {len(manual['boxes'])}")
        if len(full['boxes']) and len(manual['boxes']):
            box_diff = torch.abs(full['boxes'] - manual['boxes']).mean().item()
            score_diff = torch.abs(full['scores'] - manual['scores']).mean().item()
            label_diff = int((full['labels'] != manual['labels']).sum().item())
            logging.info(f"🔢 Avg. box difference:   {box_diff:.6f}")
            logging.info(f"🔢 Avg. score difference: {score_diff:.6f}")
            logging.info(f"❗ Label mismatches:      {label_diff}")
        else:
            logging.warning("No detected boxes in either result.")