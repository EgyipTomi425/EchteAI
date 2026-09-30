#!/usr/bin/env bash
# Final measurement phase, run on an otherwise idle GPU after every other test:
# rebuild all benchmark engines from scratch (fresh timing cache, detailed metadata, autotuning on an idle
# GPU), benchmark every precision at batch sizes 1, 8 and 6 (fleet scenario), measure energy with CUDA graphs,
# re-measure accuracy on exactly the timed engines, then derive the remaining tables, figures and the
# assembled manuscript.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/engines/previous
mv results/engines/*_bs[168].engine results/engines/previous/ 2>/dev/null || true
mv results/engines/timing.cache results/engines/previous/ 2>/dev/null || true

# Latency (CUDA graphs) and throughput at batch sizes 1 and 8, then 6 (synchronised frames of six cameras).
$PY scripts/03_build_engines.py --precisions fp32 fp16 int8 int8fp32 fp8
$PY scripts/04_benchmark.py --precisions fp32 fp16 int8 int8fp32 fp8
$PY scripts/03_build_engines.py --batch-sizes 6 --precisions fp32 fp16 int8 fp8
$PY scripts/04_benchmark.py --batch-sizes 6 --precisions fp32 fp16 int8 fp8 --out benchmark_bs6.csv
# Recommended DenseNet-121 INT8 variant (batch normalisation in FP16), timed like every other engine.
$PY scripts/39_build_extra.py
$PY scripts/04_benchmark.py --models densenet121 --precisions int8bnfp16 --batch-sizes 8 6 --out benchmark_extra.csv
# Energy per image with CUDA graphs, the execution mode of the reported latencies.
$PY scripts/36_energy_graph.py
$PY scripts/36_energy_graph.py --models densenet121 --precisions int8bnfp16

# Accuracy on the very engines that were timed (Faster R-CNN is evaluated with its dynamic-shape
# engines, which are not part of the timing), then the statistics that depend on it.
$PY scripts/05_accuracy.py --models yolov10s yolov10x efficientnet_b0 densenet121 --precisions fp32 fp16 int8 int8fp32 fp8
$PY scripts/22_bootstrap_ci.py
$PY scripts/18_memory_traffic.py
$PY scripts/09_model_stats.py
$PY scripts/23_deployed_agreement.py
# Analyses of the stored detections and calibrations (CPU), and the GPU cost of NMS.
$PY scripts/32_operating_point.py
$PY scripts/33_duplicates.py
$PY scripts/37_nms_cost.py
$PY scripts/30_calibration_scales.py --model efficientnet_b0
# Fleet scenario, localization, early exit, figures, tables and manuscript.
$PY scripts/11_aibo.py
$PY scripts/16_localization.py
$PY scripts/19_early_exit.py
$PY scripts/12_figures.py
$PY scripts/13_tables.py
PAPER=${PEPAI_PAPER:-../paper}
if [ -f "$PAPER/main.tex" ]; then $PY scripts/15_assemble_paper.py; fi
echo FINAL_DONE
