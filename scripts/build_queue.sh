#!/usr/bin/env bash
# Builds engines as soon as the quantized ONNX models of a model are complete.
# Usage: scripts/build_queue.sh [--debug] [--precisions ...]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
MODELS=${MODELS:-"efficientnet_b0 densenet121 yolov10s yolov10x frcnn_r50_fpn"}
stable() {  # file exists and its size did not change for 20 s (writer finished)
    [ -f "$1" ] || return 1
    local a; a=$(stat -c %s "$1"); sleep 20; [ "$a" = "$(stat -c %s "$1")" ]
}
for m in $MODELS; do
    # INT8 (FP16 fallback) is the last variant written by 02_quantize.py.
    until stable "results/onnx/${m}_int8.onnx" && stable "results/onnx/${m}_fp16.onnx"; do sleep 30; done
    $PY scripts/03_build_engines.py --models "$m" "$@"
done
echo BUILD_QUEUE_DONE "$@"
