from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else CODE_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Relative paths in the config are relative to the code root, so scripts work from any cwd.
    for section in ("paths", "coco", "imagenetv2"):
        for key, value in cfg[section].items():
            if key in ("data", "results", "root"):
                cfg[section][key] = (CODE_ROOT / value).resolve()
    return cfg


def results_dir(cfg, *parts):
    d = Path(cfg["paths"]["results"]).joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
