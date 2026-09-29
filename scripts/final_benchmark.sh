#!/usr/bin/env bash
# Final measurement phase, run on an otherwise idle GPU after every other test:
# rebuild all benchmark engines from scratch (fresh timing cache, detailed metadata, autotuning on an idle
# GPU), benchmark every precision, re-measure accuracy on the timed engines, then derive the remaining
# tables, figures and the assembled manuscript.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/engines/previous
mv results/engines/*_bs[18].engine results/engines/previous/ 2>/dev/null || true
mv results/engines/timing.cache results/engines/previous/ 2>/dev/null || true
$PY scripts/03_build_engines.py --precisions fp32 fp16 int8 int8fp32 fp8
$PY scripts/04_benchmark.py --precisions fp32 fp16 int8 int8fp32 fp8
# Accuracy on the very engines that were timed (Faster R-CNN is evaluated with its dynamic-shape
# engines, which are not part of the timing), then the statistics that depend on it.
$PY scripts/05_accuracy.py --models yolov10s yolov10x efficientnet_b0 densenet121 --precisions fp32 fp16 int8 int8fp32 fp8
$PY scripts/22_bootstrap_ci.py
$PY scripts/18_memory_traffic.py
$PY scripts/09_model_stats.py
$PY scripts/23_deployed_agreement.py
$PY scripts/11_aibo.py
$PY scripts/16_localization.py
$PY scripts/19_early_exit.py
$PY scripts/12_figures.py
$PY scripts/13_tables.py
$PY scripts/15_assemble_paper.py
echo FINAL_DONE
