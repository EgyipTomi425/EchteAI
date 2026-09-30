# EchteAI — PEP-AI validation of quantized perception models

PEP-AI (*Precise, Explainable and Provable AI*) validates post-training quantized convolutional
networks at the activation level instead of on output accuracy alone. This repository contains the
complete, reproducible pipeline of the journal extension of

> T. Menyhárt, A. Hajdu, R. Lakatos: *Quantization-Induced Error Propagation: Activation Analysis within
> the PEP-AI Framework for Sustainable High-Efficiency and Explainable AI*, CITDS 2026.

The code of the conference paper (simulated INT8 on a Faster R-CNN backbone) is preserved under the
tag [`citds-2026`](https://github.com/EgyipTomi425/EchteAI/tree/citds-2026).

## What the pipeline does

| | |
|---|---|
| Models | Faster R-CNN R50-FPN (backbone + FPN in TensorRT), YOLOv10-S, YOLOv10-X, EfficientNet-B0, DenseNet-121 |
| Precisions | strict FP32 (TF32 off), FP16, INT8 (explicit Q/DQ, FP16 fallback), INT8 with FP32 fallback (analysis), FP8 E4M3 |
| Runtime | TensorRT 11 strongly typed engines on an NVIDIA H200; quantization with NVIDIA ModelOpt |
| Data | COCO 2017 (512 train images for calibration, 5 000 val images for evaluation), ImageNetV2 matched-frequency |

Measurements: task accuracy with paired bootstrap CIs; latency (p50, p99, jitter) and NVML energy per
image; memory traffic and executed kernel precision from the engine inspector; layer-wise activation
deviation (MRE, SQNR) on 500 images; deviation versus detection errors; robustness under eight adverse
conditions (COCO-C and low light); label-free diagnosis and repair (placement, calibration, selective
precision); quantization-noise and propagation model checks; monocular distance error; AIBO fleet
energy, cost and CO₂ scenario.

## Setup

```bash
conda create -n pepai python=3.12
conda activate pepai
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

Datasets and weights live in `data/` (not tracked): COCO 2017 `val2017/` and `annotations/` under
`data/coco/` (<https://cocodataset.org/#download>); `scripts/00_download.py` fetches the calibration
subset and ImageNetV2; the YOLOv10 weights are downloaded by Ultralytics on first use. All paths are set in `configs/default.yaml`, and all
randomness is seeded from it.

## Pipeline

| Step | Script | Output (`results/`) |
|---|---|---|
| Calibration subset, ImageNetV2 | `00_download.py` | `data/` |
| FP32 ONNX export (Faster R-CNN with folded BN) | `01_export.py` | `onnx/*_fp32.onnx` |
| FP16, INT8 and (`--fp8`) FP8 variants | `02_quantize.py` | `onnx/*_{fp16,int8,int8fp32,fp8}.onnx` |
| TensorRT engines (`--debug`: analysis engines with marked outputs) | `03_build_engines.py` | `engines/` |
| Latency, jitter, energy | `04_benchmark.py` | `tables/benchmark.csv` |
| mAP / top-1 | `05_accuracy.py` | `tables/accuracy.csv`, `detections/` |
| Layer-wise activation deviation (`--condition`: corrupted images) | `06_activations.py` | `activations/` |
| Deviation versus detection errors | `07_risk.py` | `risk/` |
| Adverse conditions | `08_robustness.py` | `tables/robustness.csv` |
| Architecture table, weight bytes | `09_model_stats.py` | `tables/models.csv` |
| PEP-AI-guided selective precision | `10_selective.py` | `tables/selective.csv` |
| AIBO fleet scenario | `11_aibo.py` | `tables/aibo.csv` |
| Figures / LaTeX tables | `12_figures.py`, `13_tables.py` | `figures/`, `tables/tex/` |
| Dataset statistics | `14_dataset_stats.py` | `tables/datasets.csv` |
| Splice tables and figures into the article (`PEPAI_PAPER`) | `15_assemble_paper.py` | |
| Monocular distance error of box shifts | `16_localization.py` | `tables/localization.csv` |
| Placement and calibration ablation | `17_placement_ablation.py` | `tables/placement_ablation.csv` |
| Memory traffic and executed kernel precision | `18_memory_traffic.py` | `tables/memory_traffic.csv` |
| Upper bound of spatial early exit | `19_early_exit.py` | `tables/early_exit.csv` |
| Label-free selection criteria | `20_label_free_selection.py` | `tables/label_free_selection.csv` |
| Robust calibration and input precision | `21_robust_calibration.py` | `tables/robust_calibration.csv` |
| Paired bootstrap CIs | `22_bootstrap_ci.py` | `tables/accuracy_ci.csv` |
| FP32 agreement of deployed engines (critical errors) | `23_deployed_agreement.py` | `tables/deployed_agreement.csv` |
| Odds ratio of local deviation for vanishing detections | `24_risk_odds.py` | `tables/risk_odds.csv` |
| Closed-form batch-normalisation gain vs measurement | `25_bn_amplification.py` | `tables/bn_amplification.csv` |
| Injected noise per quantizer from FP32 statistics (INT8/FP8, optionally corrupted inputs) | `26_quantizer_snr.py` | `tables/quantizer_snr*.csv` |
| Entropy vs max calibration under reduced contrast | `27_calibration_robustness.py` | `tables/calibration_robustness.csv` |
| Qualitative example (detections and deviation map) | `28_qualitative.py` | `qualitative/` |
| Batch-6 benchmark for the fleet scenario | `03_build_engines.py --batch-sizes 6`, `04_benchmark.py --batch-sizes 6` | `tables/benchmark_bs6.csv` |
| INT8 accuracy for calibration subsets (sample and order) | `29_calibration_variability.py` | `tables/calibration_variability.csv` |
| Activation scales of two calibrations compared | `30_calibration_scales.py` | `tables/calibration_scales_<model>.csv` |
| Causal test: swapping activation scales between calibrations | `31_scale_swap.py` | `tables/scale_swap.csv` |
| Road-user AP and recall/precision at the operating threshold | `32_operating_point.py` | `tables/operating_point.csv` |
| Duplicate detections of the NMS-free head, mAP with NMS | `33_duplicates.py` | `tables/duplicates.csv` |
| Order-independent entropy calibration (NVIDIA reference KL search) | `34_global_entropy.py` | `tables/global_entropy.csv` |
| EfficientNet-B0 with quantizers on convolution inputs only | `35_conv_only_placement.py` | `tables/conv_only_placement.csv` |
| Energy per image with CUDA graphs (execution mode of the latencies) | `36_energy_graph.py` | `tables/energy_graph.csv` |
| GPU cost of class-wise NMS on the YOLOv10 output | `37_nms_cost.py` | `tables/nms_cost.csv` |
| Duplicate suppression under selective precision (YOLOv10-X) | `38_selective_duplicates.py` | `tables/selective_duplicates.csv` |
| Engines of the DenseNet-121 variant with batch normalisation in FP16 | `39_build_extra.py` (then `04_benchmark.py`, `36_energy_graph.py`) | `tables/benchmark_extra.csv` |

### Running everything

Engine builds may run in parallel; measurements must not. `scripts/build_queue.sh` builds engines as
soon as their quantized models exist. `scripts/run_tests.py` runs every measurement one at a time as
soon as its inputs exist, never while another measurement is running, and records completed tasks in
`results/logs/done.txt`. The last task, `scripts/final_benchmark.sh`, rebuilds all timed engines on an
idle GPU, benchmarks them, re-measures accuracy on exactly these engines and regenerates every table
and figure.

## License

GNU Lesser General Public License v2.1 (see `LICENSE`).
