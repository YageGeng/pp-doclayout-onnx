#!/usr/bin/env python3
"""Download PaddlePaddle PP-DocLayoutV3 ONNX model files from Hugging Face."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "PaddlePaddle/PP-DocLayoutV3_onnx"
DEFAULT_FILES = ("inference.onnx", "inference.yml")
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PP-DocLayoutV3 ONNX files from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repository revision, branch, or commit. Default: main",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory to write downloaded files into. Default: repository models directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    return parser.parse_args()


def copy_downloaded_file(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"skip existing {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"wrote {destination}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    for filename in DEFAULT_FILES:
        downloaded_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                revision=args.revision,
            )
        )
        copy_downloaded_file(downloaded_path, output_dir / filename, args.force)


if __name__ == "__main__":
    main()
