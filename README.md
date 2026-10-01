# EchteAI — PEP-AI validation of quantized perception models

> T. Menyhárt, A. Hajdu, R. Lakatos: *PEP-AI Validated Quantization for Energy-Efficient Perception in
> Intelligent Transportation Systems*. Journal manuscript (in preparation).

**Manuscript (PDF): [`paper/PEP-AI_manuscript.pdf`](paper/PEP-AI_manuscript.pdf)**

The LaTeX source of the article is in [`paper/`](paper/) (Springer Nature template, single `main.tex`,
figures in `paper/figures/`). The folder compiles as is with pdfLaTeX and BibTeX, e.g. after uploading
it to Overleaf or with `latexmk -pdf main.tex`. The TikZ diagrams of the Methods section are rebuilt
from `paper/figures/src/` with `build.sh`; all other figures and the generated tables are written by
the pipeline (`scripts/15_assemble_paper.py`).

PEP-AI (*Precise, Explainable and Provable AI*) validates post-training quantized convolutional
networks at the activation level instead of on output accuracy alone. This repository contains the
article and the complete, reproducible pipeline of the journal extension of

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

## Main result: predicting INT8 tolerance (Section 3.8 of the article)

The quantization error of a network splits into the noise injected by each quantizer, which FP32
statistics predict, and its propagation to the task head, which one measurement of the quantized
network reveals. This gives a procedure that tells, before deployment, whether INT8 is safe:

| Step | What is computed | Formula (INT8 / FP8) | Accuracy on the five networks |
|---|---|---|---|
| 1 | noise of each quantizer from FP32 activations and calibrated scales | SQNR ≈ 52.9 dB − 20 log₁₀ κ / 31.5 dB | median within 0.6–2.9 dB (INT8), 0.02 dB (FP8) |
| 2 | head-input SQNR with unit propagation factors (no quantized model) | SQNR_add = −10 log₁₀ Σ ρₙ² | ranks the INT8 loss with ρ = 0.8: screening |
| 3 | one measurement of the quantized head input → propagation factor | Γ̄ = 10^((SQNR_add − SQNR_h)/10) | ranks the INT8 loss exactly (ρ = 1.0); loss ×10 per 12 dB, leave-one-out within ×1.3–2.2 |
| 4 | plateau over depth; effect of a signal change after calibration | SQNR∞ = −20 log₁₀ ρ + 10 log₁₀(1 − g²); ΔSQNR = 20 log₁₀ c / 0 | within 0.3 dB; full shift underestimated by 2.9–4.5 dB |
| 5 | format and layers | the lowest precision within the accepted loss; FP16 layers by Γₙ→ₕ ρₙ² | measured ranking needed for single layers |

How to read the numbers: an SQNR of 40, 20 and 0 dB is a relative error of 1 %, 10 % and 100 %;
6 dB is a factor of two. Example: Faster R-CNN (Γ̄ = 0.13, head-input SQNR 21.5 dB) loses 0.7 % in
INT8; EfficientNet-B0 injects no more noise than YOLOv10-X but amplifies it (Γ̄ = 2.89) and loses
62 %, whereas FP8 keeps every network within 1 % of FP32. The calibration rests on five networks.

```bash
python scripts/26_quantizer_snr.py                                       # quantizer table (FP32 + scales only)
python scripts/42_predict_int8.py --model yolov10x                       # step 2: screening
python scripts/42_predict_int8.py --model yolov10x --head-sqnr 4.25      # step 3: after one measurement
python scripts/42_predict_int8.py --model yolov10x --head-sqnr 4.25 --leave-out   # without this model in the fit
```

Without arguments for `--tables`, the script uses `results/tables` if present and otherwise the
reference tables, so it runs on a fresh clone. Tables C7–C9 and Fig. C3 of the article contain the
formulas, the worked example and the predictions for all five networks.

## Static analysis of any model: no images, no execution (Sections 2.13 and 3.9)

`scripts/43_static_analysis.py` reads only the weights and the batch-normalisation buffers of an FP32
PyTorch model and reports, in 20–35 s on a CPU, how much noise each number format injects:

```bash
python scripts/43_static_analysis.py --model all                  # the five networks of the article
python scripts/43_static_analysis.py --torchvision resnet50       # any torchvision classification model
python scripts/43_static_analysis.py --module my_model.pt         # any model saved with torch.save(model, path)
```

What it computes: exact INT8 (per channel, per tensor) and FP8 SQNR of every weight tensor; for every
batch-normalised tensor a Gaussian model per channel, N(β, γ²v/(v+ε)), passed through the following
activation, with the channel imbalance, the MSE-optimal range κ and the injected INT8/FP8 SQNR (Lemma 1);
the network-level SQNR_add for INT8 with per-channel or per-tensor weights, FP8 and FP16; the gain of
non-foldable batch normalisations (Eq. A3); structural flags; and where the network falls among the five
measured networks of the article. Output: a report on the screen, `results/tables/static_analysis.csv` and
per-layer tables in `results/tables/static/`.

How network properties enter (Table C12 of the article): SQNR_add = −10 log₁₀ N − 10 log₁₀⟨ρ²⟩, i.e. −3 dB per
doubling of the number of quantized tensors; the mean is dominated by the weakest tensors (outlier channels,
depthwise inputs, gates, attention), −20 dB per decade of κ; width matters only through channel imbalance;
parameter count and spatial size do not matter.

**Validation.** `scripts/44_validate_static.py --torchvision <name> --images <folder>` executes a model on a
labelled image folder (ImageNet class-index sub-folders, e.g. ImageNetV2) with simulated quantization at the
modelled tensors, only to check the static prediction. Results on two networks not used elsewhere in the
article and on the two classifiers of the article (ImageNetV2, 128 calibration and 2 000 evaluation images):

| Network | FP8 noise: static − measured | INT8 noise: static − measured (median, MAE) | INT8 loss measured [95 % CI] / predicted* | FP8 loss measured [95 % CI] / predicted* |
|---|---|---|---|---|
| MobileNetV2 (new) | 0.14 dB | +0.4 dB, 1.1 dB | 0.2 % [−1.0, 1.3] / 1.4 % | 10.3 % [7.8, 12.6] / 9.8 % |
| ResNet-50 (new) | 0.02 dB | +5.4 dB, 6.3 dB | 1.1 % [0.0, 2.0] / 1.5 % | 2.8 % [1.3, 4.3] / 3.7 % |
| EfficientNet-B0 | 0.11 dB | +4.0 dB, 5.0 dB | 36.8 % [33.9, 39.6] / 29.7 % | 7.1 % [5.0, 9.0] / 8.6 % |
| DenseNet-121 | 0.04 dB | +6.2 dB, 6.1 dB | 2.5 % [0.8, 4.1] / 2.5 % | 0.6 % [−0.9, 1.9] / 1.5 % |

\* predicted from one measurement of the quantized head input (Section 3.8), not from the static analysis.

**What it is good for, and how far it can be trusted**

| Question | Reliability |
|---|---|
| Per-channel or per-tensor weight scales? | exact (no model involved); e.g. ResNet-50: 15.8 dB vs 7.4 dB at the network level |
| How much noise does FP8 inject? | within 0.02–0.14 dB per quantizer and 0.05 dB for the whole network on all four checked networks: floating-point noise does not depend on the distribution (Lemma 1) |
| How much noise does INT8 inject? | within about 1 dB for bounded or unrectified activations (MobileNetV2); 4–6 dB too optimistic for SiLU and unbounded ReLU outputs (error amplitude underestimated by a factor of about 2); 8–12 dB more optimistic than the entropy calibration of the TensorRT toolchain |
| Which batch normalisations amplify noise, which structures are suspicious? | BN gain ranks the measured amplification with ρ = 0.83 (DenseNet-121) |
| How does a network rank? | static SQNR_add orders the INT8 loss of the five article networks with ρ = 0.9 (one exchange) |
| Accuracy loss in %? | **not from the static analysis**, because the propagation of the noise is not visible in the parameters (the same ~11 dB of FP8 noise ends at 6.8 dB on the head of MobileNetV2 and 17.0 dB on DenseNet-121). One measurement of the head input on ~100 images predicts it within a factor of 1.4 wherever the loss exceeds 1 % (`42_predict_int8.py`, `44_validate_static.py`) |

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
| Comparison of a new run with the reference results | `40_compare_results.py` | report (exit status 1 on accuracy deviations) |
| TopK / DFL placement ablation of the YOLOv10 head | `41_topk_ablation.py` | `tables/ablation_topk.csv` |
| INT8 tolerance of a (new) network from its quantizer table and, optionally, one head-input measurement (Section 3.8) | `42_predict_int8.py` | `tables/predict_<model>.csv` |

## Reproducing the results

**One command.** `scripts/reproduce.sh` runs every stage in order and can be restarted at any time:

```bash
scripts/reproduce.sh                          # check data models engines measure analyze compare
scripts/reproduce.sh measure analyze compare  # or selected stages
```

| Stage | What it does |
|---|---|
| `check` | GPU, driver, CUDA, TensorRT and ModelOpt versions against `reference_results/environment.json` |
| `data` | calibration subset and ImageNetV2 (COCO val2017 must already be in `data/coco/`) |
| `models` | FP32 ONNX export, FP16/INT8 and FP8 quantization |
| `engines` | static- and dynamic-shape TensorRT engines |
| `measure` | every measurement via `scripts/run_tests.py`, then `scripts/final_benchmark.sh` |
| `analyze` | fleet scenario, figures, LaTeX tables, spliced into `paper/main.tex` (or `$PEPAI_PAPER/main.tex`) |
| `compare` | `scripts/40_compare_results.py`: new tables against `reference_results/tables/` |

**Scheduling.** Engine builds may run in parallel; measurements never do. `scripts/run_tests.py` starts
a measurement only when its inputs exist and no other measurement is running, and records completed tasks
in `results/logs/done.txt`, so an interrupted run resumes where it stopped (delete a line to repeat a
task). The last task, `scripts/final_benchmark.sh`, rebuilds all timed engines on an idle GPU, measures
latency at batch sizes 1, 6 and 8 and energy with CUDA graphs, re-measures accuracy on exactly these
engines and derives every remaining table and figure. On an NVIDIA H200 the measurements take about
35 GPU hours, quantization and engine builds a few hours more.

**What reproduces.** With the versions in `reference_results/environment.json`, accuracies, deviation
statistics and all derived quantities are deterministic up to TensorRT tactic selection and should agree
with the reference within the tolerances of `40_compare_results.py` (0.3 points for accuracies, 5 % for
other statistics). As a check, re-running the TopK/DFL ablation from scratch (quantization, engine
build and evaluation on 5 000 images, `41_topk_ablation.py`) reproduced all four mAP values of the
reference to five decimals. Latency, energy and the fleet figures derived from them depend on the GPU, driver and
TensorRT version; they are compared with a 10 % tolerance and reported as warnings. INT8 accuracies of
the fragile networks also depend on the order of the calibration images (see the article); the
calibration set is loaded in a fixed, seeded order.

**Reference results.** `reference_results/` contains the tables, figures and small intermediate results
of the run reported in the article (see `reference_results/README.md`). Large artefacts (ONNX models,
TensorRT engines, stored detections and activations, about 21 GB) are not tracked; the pipeline
regenerates them.

## License

GNU Lesser General Public License v2.1 (see `LICENSE`).
