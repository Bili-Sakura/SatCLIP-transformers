#!/usr/bin/env python3
"""Sync src/satclip custom code into model_repo subdirectories."""

import argparse
import shutil
from pathlib import Path


def sync_model_code(model_repo_root: Path, src_pkg: Path) -> None:
    for model_dir in sorted(model_repo_root.glob("SatCLIP-*")):
        if not model_dir.is_dir():
            continue
        dst = model_dir / "satclip"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src_pkg,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        print(f"Synced {src_pkg} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model_repo",
        help="Path to model_repo directory",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "satclip",
        help="Path to src/satclip package",
    )
    args = parser.parse_args()
    sync_model_code(args.model_repo, args.src)


if __name__ == "__main__":
    main()
