import EchteAI.data.preprocessing as pp
import EchteAI.data.dataloaders as dl
import EchteAI.models.vision.fasterrcnn as frcnn
import logging
import torchvision.transforms as T
import torch
import os
import onnx
import torch.nn.functional as F
from torch.fx import symbolic_trace
from torch.fx import symbolic_trace
from torch.quantization import QConfig, MinMaxObserver
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from datetime import datetime


from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger('matplotlib').setLevel(logging.WARNING)

file_handler = logging.FileHandler('output.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Azonnali flush minden logolás után
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
file_handler.flush = lambda: None  # Kikapcsolja a pufferelést

logger = logging.getLogger()
logger.addHandler(file_handler)


import PIL
logging.getLogger("PIL").setLevel(logging.INFO)

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    root = "./downloads"
    image_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    label_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    pp.download_data([image_zip_url, label_zip_url], download_dir=root)
    transform = T.Compose([T.ToTensor()])

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, class_to_idx, idx_to_class = dl.get_dataloaders(dl.KittiDataset,root,transform)

    num_epochs = 16
    model = frcnn.setup_fasterrcnn(train_dataset)
    model.to(device)
    model = frcnn.train_fasterrcnn(model, train_loader, val_loader, device, num_epochs)
    model.eval()

    #dl.convert_kitti_to_yolo_structure() ## Első futtattáskor ne legyen kikommentezve
    model_yolo = frcnn.setup_yolo(model_name="yolo11s.pt")
    print(model_yolo)
    model_yolo = frcnn.train_yolo(model_yolo, data_yaml_path="downloads/yolo_dataset/kitti.yaml", device=device, model_name="yolo11s.pt")
    metrics = frcnn.compute_metrics_yolo(model_yolo, data_yaml_path="downloads/yolo_dataset/kitti.yaml", device=device)
    logging.info(f"YOLOv11 validation metrics: {metrics}")
    frcnn.run_predictions_yolo(model_yolo, image_folder="downloads/yolo_dataset/images/val", output_folder="outputs/yolo", num_images=45)

    #model_yolo.export(format="onnx")
    model_yolo.export(format="onnx", imgsz=(640, 640), dynamic=False)
    model_yolo_fp32 = frcnn.setup_yolo(model_name="yolo11s.onnx")
    frcnn.run_predictions_yolo(model_yolo_fp32, image_folder="downloads/yolo_dataset/images/val", output_folder="outputs/yolo/fp32", num_images=40)
    metrics = frcnn.compute_metrics_yolo(model_yolo_fp32, data_yaml_path="downloads/yolo_dataset/kitti.yaml", device="cpu")
    logging.info(f"YOLOv11 onnx validation metrics: {metrics}")

    # loader = frcnn.YoloCalibrationDataLoader("downloads/yolo_dataset/images/train", "./self_yolo11s.onnx", batch_size=1, num_batches=32)
    # for _ in range(len(loader)):
    #     batch = loader.get_next()
    #     if batch is None:
    #         break
    #     frcnn.predict_yolo_onnx_tensor(torch.from_numpy(list(batch.values())[0]))

    device="cpu"
    loader = frcnn.YoloCalibrationDataLoader("downloads/yolo_dataset/images/train", "./self_yolo11s.onnx", batch_size=1, num_batches=32)
    frcnn.quantize_onnx_model_calibdl("./self_yolo11s.onnx", loader, "./self_yolo11s_int16.onnx")
    model_yolo_quantized = frcnn.setup_yolo(model_name="yolo11s_int16.onnx")
    frcnn.run_predictions_yolo(model_yolo_quantized, image_folder="downloads/yolo_dataset/images/val", output_folder="outputs/yolo/int16", num_images=40, batch_size=1)
    metrics = frcnn.compute_metrics_yolo(model_yolo_quantized, data_yaml_path="downloads/yolo_dataset/kitti.yaml", device=device)
    logging.info(f"YOLOv11 int16 validation metrics: {metrics}")

    return 0

    quant_presets = [
        "INT8_CNN_DEFAULT", "INT16_CNN_DEFAULT", "INT8_CNN_ACCURATE", "INT16_CNN_ACCURATE",
        "XINT8", "XINT8_ADAROUND", "XINT8_ADAQUANT", "S8S8_AAWS", "S8S8_AAWS_ADAROUND",
        "S8S8_AAWS_ADAQUANT", "U8S8_AAWS", "U8S8_AAWS_ADAROUND", "U8S8_AAWS_ADAQUANT",
        "S16S8_ASWS", "S16S8_ASWS_ADAROUND", "S16S8_ASWS_ADAQUANT", "A8W8", "A16W8",
        "U16S8_AAWS", "U16S8_AAWS_ADAROUND", "U16S8_AAWS_ADAQUANT"
    ]

    for preset in quant_presets:
        try:
            safe_name = preset.lower()
            output_model_path = f"self_yolo_{safe_name}.onnx"
            output_folder = f"outputs/yolo/{safe_name}"

            logging.info(f"[{preset}] Quantization started...")

            frcnn.quantize_yolo_model_with_quark(
                model_path="self_yolo11s.onnx",
                image_dir="downloads/yolo_dataset/images/train",
                output_path=output_model_path,
                batch_size=1,
                num_batches=32,
                image_size=(640, 640),
                quant_preset=preset
            )

            model = frcnn.setup_yolo(output_model_path)

            metrics = frcnn.compute_metrics_yolo(
                model,
                data_yaml_path="downloads/yolo_dataset/kitti.yaml",
                device=device
            )
            logging.info(f"[{preset}] validation metrics: {metrics}")

            frcnn.run_predictions_yolo(
                model,
                image_folder="downloads/yolo_dataset/images/val",
                output_folder=output_folder,
                num_images=45
            )

            logging.info(f"[{preset}] ✅ Done")

        except Exception as e:
            logging.error(f"[{preset}] ❌ Failed: {e}")

    return 0

    
    # traced = symbolic_trace(model.backbone)

    # qconfig = QConfig(
    #     activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    #     weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    # )[]
    # qconfig_dict = {"": qconfig}

    # images, _ = next(iter(val_loader))
    # images = torch.stack([
    #     F.interpolate(img.unsqueeze(0), size=(375, 1242), mode="bilinear", align_corners=False).squeeze(0)
    #     for img in images
    # ], dim=0).to(device)
    # images = images[:32]

    # prepared = prepare_fx(traced, qconfig_dict, example_inputs=(images,))

    # prepared(images)

    # quantized = convert_fx(prepared)

    # torch.save(quantized.state_dict(), "quantized_backbone.pth")

    # model.backbone = quantized



    device = "cpu"
    val_out = os.path.join("outputs", "predictions", "validation")
    test_out = os.path.join("outputs", "predictions", "test")
    #frcnn.run_predictions_fasterrcnn(model, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out, evaluate=True, num_batches=3)
    #frcnn.run_predictions_fasterrcnn(model, test_loader, device, test_dataset, test_out, evaluate=False, num_batches=2)

    device = "cpu"
    #model_quantized = frcnn.quantize_fasterrcnn(model, train_loader)
    #model_quantized = frcnn.quantize_dynamic(model_quantized)
    #print(model_quantized)
    val_out_qint8_static = os.path.join("outputs", "qint8_static", "validation")
    #frcnn.run_predictions_fasterrcnn(model_quantized, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out_qint8_static, evaluate=True, num_batches=3)

    #video_set, video_loader = dl.video_to_dataloader("car_video.mp4", class_to_idx, idx_to_class)
    #video_out_qint8_static = os.path.join("outputs", "qint8_static", "video")
    #frcnn.run_predictions_fasterrcnn(model_quantized, video_loader, device, video_set.dataset if hasattr(video_set, "dataset") else video_set, video_out_qint8_static, evaluate=False, num_batches=3)
    #frcnn.compare_models_visual(model, model_quantized, video_loader, device, video_set.dataset if hasattr(video_set, "dataset") else video_set, video_out_qint8_static, num_batches=-1)
    #dl.create_video_from_images(video_out_qint8_static)

    #torch.save(model_quantized, "quantized_model.pth")



    images, _ = next(iter(val_loader))
    images = torch.stack([F.interpolate(img.unsqueeze(0), size=(375, 1242), mode="bilinear", align_corners=False).squeeze(0) for img in images], dim=0).to(device)
    images = images[:32]
    #torch.onnx.export(model.backbone, images[:15], "./outputs/model.onnx")
    first_batch = next(iter(val_loader))
    first_image = first_batch[0][13].to(device)
    dl.save_image(first_image)


    frcnn.split_save_frcnn(model, images[:2], device)
    model_onnx_fp32 = frcnn.ONNXFasterRCNNWrapper("feature_extractor.onnx", "detector_head.onnx", "cpu")
    val_out_onnx_fp32 = os.path.join("outputs", "onnx", "fp32")
    #frcnn.run_predictions_fasterrcnn(model_onnx_fp32, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out_onnx_fp32, evaluate=False, num_batches=3, batch_size=2)

    frcnn.quantize_onnx_static(
        onnx_model_path="feature_extractor.onnx",
        data_loader=train_loader,
        input_shape=(2, 3, 375, 1242),
        num_batches=8
    )
    model_onnx_int8 = frcnn.ONNXFasterRCNNWrapper("feature_extractor_int8.onnx", "detector_head.onnx", "cpu")
    val_out_onnx_int8 = os.path.join("outputs", "onnx", "int8")
    #frcnn.run_predictions_fasterrcnn(model_onnx_int8, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out_onnx_int8, evaluate=False, num_batches=3, batch_size=2)

    #outputs1 = frcnn.backbone_cnn_layers_outputs(model_quantized, first_image)
    #outputs2 = frcnn.backbone_cnn_layers_outputs(model, first_image)
    #outputs_diffs = frcnn.absolute_differences(outputs1, outputs2)
    #outputs_percentage_diffs = frcnn.percentage_differences(outputs1, outputs2)
    #frcnn.visualize_cnn_outputs(outputs1)
    #frcnn.visualize_cnn_outputs(outputs_diffs, filename="activation_difference_heatmap")
    #frcnn.visualize_cnn_outputs(outputs_diffs, filename="activation_difference_heatmap_layer1", layer=1)
    #frcnn.visualize_cnn_outputs(outputs_percentage_diffs, filename="activation_difference_heatmap_percentage")
    #frcnn.visualize_cnn_outputs(outputs_percentage_diffs, filename="activation_difference_heatmap_percentage_layer1")
    #frcnn.fit_and_plot_distribution(outputs1, outputs_diffs, layer=1)
    #frcnn.fit_and_plot_distribution(outputs1, outputs_percentage_diffs, layer=1, filename="distribution_fit_percentage")


    tensor = images[:2]
    outputs_fp32 = frcnn.onnx_conv_outputs_from_batch("feature_extractor.onnx", tensor, pattern=r".*conv.*")
    outputs_int8 = frcnn.onnx_conv_outputs_from_batch("feature_extractor_int8.onnx", tensor, pattern=r".*conv.*")
    onnx_abs_diffs = frcnn.absolute_differences(outputs_fp32, outputs_int8)
    frcnn.fit_and_plot_distribution(outputs_fp32, onnx_abs_diffs, filename="distribution_onnx_fit_abs")
    print("a")


if __name__ == "__main__":
    main()
