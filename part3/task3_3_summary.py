import argparse
import json
import os
from pathlib import Path

import csv
import numpy as np


def get():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="part3/task3_3_elasticity.json")
    p.add_argument("--out", type=str, default="part3/task3_3_summary.json")
    return p.parse_args()


def dbp(cfg_path, db):
    if os.path.isabs(db):
        return db
    base = Path(cfg_path).resolve().parent / "GAOT-main"
    return str((base / db).resolve())


def outp(cfg_path, out):
    out = Path(out)
    if out.is_absolute():
        return out
    root = Path(cfg_path).resolve().parent.parent
    return (root / out).resolve()


def go():
    args = get()
    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = json.load(f)

    m = cfg["model"]
    size = m["latent_tokens_size"]
    strat = m.get("tokenization_strategy", "grid")
    tr_cfg = m["args"].get("transformer", {})
    patch = tr_cfg["patch_size"]
    tokens_latent = int(np.prod(size))

    use_ca = tr_cfg.get("use_cross_attention", False)
    seed_tokens = tr_cfg.get("num_seed_tokens") if use_ca else None
    tokens_proc = int(seed_tokens) if seed_tokens is not None else tokens_latent

    h, w = size
    tokens_patch = (h // patch) * (w // patch)
    if strat != "grid":
        tokens_patch = None
    tokens_used = tokens_latent
    tokens_tr = tokens_patch if strat == "grid" else tokens_proc

    db = dbp(cfg_path, cfg["path"]["database_path"])
    if not os.path.exists(db):
        raise FileNotFoundError("missing")

    with open(db, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("bad")
    last = rows[-1]

    out = {
        "config": str(cfg_path),
        "database": db,
        "relative_error_l1": float(last["relative error (direct)"]) if "relative error (direct)" in last else None,
        "training_time_sec": float(last["training time"]) if "training time" in last else None,
        "token_count": tokens_used,
        "token_count_used": tokens_used,
        "token_count_transformer": tokens_tr,
        "tokens_latent": tokens_latent,
        "tokens_seed": seed_tokens,
        "tokens_patches": tokens_patch,
        "latent_tokens_size": size,
        "patch_size": patch,
        "use_cross_attention": use_ca,
        "tokenization_strategy": strat,
    }

    out_file = outp(cfg_path, args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    

if __name__ == "__main__":
    go()
