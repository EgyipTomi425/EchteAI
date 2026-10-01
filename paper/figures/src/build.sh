#!/usr/bin/env bash
# Compiles every standalone TikZ figure into ../<name>.pdf
set -e
cd "$(dirname "$0")"
for f in fig_*.tex; do
    pdflatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null
    mv "${f%.tex}.pdf" ..
done
rm -f *.aux *.log
