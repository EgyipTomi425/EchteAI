"""Copy generated figures into paper/figures and splice generated tables into paper/main.tex.

main.tex contains marker pairs
    % BEGIN GENERATED <name>
    % END GENERATED <name>
and everything between them is replaced by results/tables/tex/<name>.tex. The Springer template
asks for a single .tex file, so tables are inlined instead of \\input.
"""
import os
import re
import shutil
from pathlib import Path

from pepai.config import CODE_ROOT, load_config, results_dir

PAPER = Path(os.environ.get("PEPAI_PAPER", CODE_ROOT / "paper"))   # LaTeX source of the article

if __name__ == "__main__":
    cfg = load_config()
    fig_dir = PAPER / "figures"
    for f in results_dir(cfg, "figures").glob("*.pdf"):
        shutil.copy(f, fig_dir / f.name)
    tex = (PAPER / "main.tex").read_text()
    tables = results_dir(cfg, "tables", "tex")

    def splice(m):
        name = m.group(1)
        src = tables / f"{name}.tex"
        body = src.read_text() if src.exists() else f"% ({name} not generated yet)\n"
        return f"% BEGIN GENERATED {name}\n{body}% END GENERATED {name}"

    tex = re.sub(r"% BEGIN GENERATED (\S+)\n.*?% END GENERATED \1", splice, tex, flags=re.S)
    (PAPER / "main.tex").write_text(tex)
    print("assembled", sorted(p.stem for p in tables.glob("*.tex")))
