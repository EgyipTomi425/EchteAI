#!/bin/sh
# Split the compiled main.pdf into the article (with references) and the Supplementary Information
# (Appendices A-C), at the page of the label supp:start. Needs poppler-utils (pdfseparate, pdfunite).
set -e
cd "$(dirname "$0")"
pdf=${1:-main.pdf}
start=$(sed -n 's/.*newlabel{supp:start}{{[^}]*}{\([0-9]*\)}.*/\1/p' "${pdf%.pdf}.aux")
total=$(pdfinfo "$pdf" | awk '/^Pages:/ {print $2}')
tmp=$(mktemp -d)
pdfseparate "$pdf" "$tmp/p-%04d.pdf" 2>/dev/null
pdfunite $(seq -f "$tmp/p-%04g.pdf" 1 $((start - 1))) PEP-AI_manuscript.pdf 2>/dev/null
pdfunite $(seq -f "$tmp/p-%04g.pdf" "$start" "$total") PEP-AI_supplementary.pdf 2>/dev/null
rm -r "$tmp"
echo "PEP-AI_manuscript.pdf: pages 1-$((start - 1)); PEP-AI_supplementary.pdf: pages $start-$total"
