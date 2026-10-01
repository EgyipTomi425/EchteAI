#!/usr/bin/env bash
# Reproduce every result of the article with one command.
#
#   scripts/reproduce.sh                 # all stages in order
#   scripts/reproduce.sh measure analyze compare    # selected stages
#
# Stages
#   check     GPU, driver, CUDA, TensorRT, ModelOpt versions compared with reference_results/environment.json
#   data      calibration subsets and ImageNetV2 (COCO val2017 must already be in data/coco, see README)
#   models    FP32 ONNX export, FP16/INT8 and FP8 quantization
#   engines   static- and dynamic-shape TensorRT engines (builds may run in parallel with other work)
#   measure   every measurement, strictly one at a time on an otherwise idle GPU (scripts/run_tests.py);
#             resumable: finished tasks are listed in results/logs/done.txt and are not repeated; the last
#             task (scripts/final_benchmark.sh) rebuilds and re-times all benchmark engines
#   analyze   fleet scenario, figures and tables from results/ (and the manuscript, if $PEPAI_PAPER or
#             paper/ contains main.tex)
#   compare   regenerated tables against reference_results/ (scripts/40_compare_results.py)
#
# About 35 GPU hours on an NVIDIA H200 for "measure", plus a few hours for "models" and "engines".
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(check data models engines measure analyze compare)

for stage in "${stages[@]}"; do
    echo "=== $stage ($(date '+%F %T'))"
    case "$stage" in
        check)
            $PY - <<'EOF'
import json, pathlib
import pynvml, tensorrt
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
name = pynvml.nvmlDeviceGetName(h)
cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
import torch, modelopt, onnxruntime
here = {"gpu": name, "compute_capability": f"{cc[0]}.{cc[1]}", "driver": pynvml.nvmlSystemGetDriverVersion(),
        "cuda": torch.version.cuda, "tensorrt": tensorrt.__version__, "modelopt": modelopt.__version__,
        "onnxruntime": onnxruntime.__version__, "torch": torch.__version__}
ref_p = pathlib.Path("reference_results/environment.json")
ref = json.loads(ref_p.read_text()) if ref_p.exists() else {}
for k, v in here.items():
    r = ref.get(k)
    print(f"  {k:18s} {v}" + ("" if r in (None, v) else f"   (reference: {r})"))
if cc < (8, 9):
    print("  WARNING: no FP8 tensor cores (compute capability < 8.9); FP8 results cannot be reproduced.")
print("  Latency and energy reproduce only on the same GPU, driver and TensorRT version;")
print("  accuracies and deviation statistics should reproduce within the tolerances of 40_compare_results.py.")
EOF
            ;;
        data)
            $PY scripts/00_download.py ;;
        models)
            $PY scripts/01_export.py
            $PY scripts/02_quantize.py
            $PY scripts/02_quantize.py --fp8 ;;
        engines)
            $PY scripts/03_build_engines.py ;;
        measure)
            $PY scripts/run_tests.py ;;
        analyze)
            $PY scripts/11_aibo.py
            $PY scripts/12_figures.py
            $PY scripts/13_tables.py
            PAPER=${PEPAI_PAPER:-paper}
            if [ -f "$PAPER/main.tex" ]; then $PY scripts/15_assemble_paper.py; fi ;;
        compare)
            $PY scripts/40_compare_results.py ;;
        *)
            echo "unknown stage: $stage" >&2; exit 2 ;;
    esac
done
