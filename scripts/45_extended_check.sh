#!/bin/sh
# Out-of-sample check of the static analysis, the propagation factor and the loss relation on further networks
# (Section 3.9, Table tab:extcheck): simulated INT8/FP8 quantization on the CPU with 44_validate_static.py.
# ImageNetV2 (matched frequency) is expected in data/imagenetv2/; the seed comes from configs/default.yaml.
#   scripts/45_extended_check.sh                 # all networks of the article
#   scripts/45_extended_check.sh resnet18 ...    # selected torchvision networks
set -e
cd "$(dirname "$0")/.."
IMAGES=${IMAGES:-data/imagenetv2/imagenetv2-matched-frequency-format-val}
# networks of Table tab:staticval: 2 000 evaluation images; further networks of Table tab:extcheck: 1 000
FULL="mobilenet_v2 resnet50 efficientnet_b0 densenet121"
FURTHER="mobilenet_v3_large shufflenet_v2_x1_0 mnasnet1_0 resnet18 regnet_y_800mf googlenet resnet34 efficientnet_b1 densenet169 resnext50_32x4d resnet101"
NETS=${*:-"$FULL $FURTHER"}
for m in $NETS; do
  case " $FULL " in *" $m "*) n=2000 ;; *) n=1000 ;; esac
  echo "== $m ($n evaluation images)"
  python scripts/44_validate_static.py --torchvision "$m" --images "$IMAGES" --n-calib 128 --n-eval "$n"
done
python scripts/13_tables.py    # writes tables/extended_check.csv, extended_summary.csv and the LaTeX tables
