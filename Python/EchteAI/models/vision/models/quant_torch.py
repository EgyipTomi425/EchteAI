import torch
import torch.ao.quantization as quant
import logging
import copy
import EchteAI.models.quantized.quantized_resnet50 as qresnet

def quantize_dynamic(model):
    model.to("cpu")
    model.eval()

    model_quantized = torch.ao.quantization.quantize_dynamic(model)
    logging.info("Dynamic quantization applied (qint8_dynamic).")
    
    model_quantized.eval()
    return model_quantized.to("cpu")

def quantize_fasterrcnn(model_fp32, data_loader, number_of_batches=2):
    model_fp32.eval().to("cpu")
    logging.info("Loading quantized backbone...")

    quantized_resnet = qresnet.quantized_resnet50()
    resnet_state_dict = model_fp32.backbone.body.state_dict()
    quantized_resnet.load_state_dict(resnet_state_dict, strict=False)
    quantized_resnet.eval()

    activation_observer = quant.HistogramObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255)
    weight_observer = quant.default_per_channel_weight_observer
    quantized_resnet.qconfig = quant.QConfig(
        activation=activation_observer,
        weight=weight_observer
    )

    logging.info("Calibrate quantized backbone...")

    quant.prepare(quantized_resnet, inplace=True)

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= number_of_batches and number_of_batches >= 0:
                break

            transformed_batch, _ = model_fp32.transform(images)

            if isinstance(transformed_batch, torchvision.models.detection.image_list.ImageList):
                images_tensor = transformed_batch.tensors
            else:
                raise TypeError(f"Nem megfelelő transzformált batch formátum: {type(transformed_batch)}")

            quantized_resnet(images_tensor)

    quant.convert(quantized_resnet, inplace=True)
    quantized_resnet.eval()

    import copy
    model_quantized = copy.deepcopy(model_fp32)
    model_quantized.backbone.body = quantized_resnet
    model_quantized.eval()

    logging.info("Loaded quantized backbone.")

    return model_quantized
