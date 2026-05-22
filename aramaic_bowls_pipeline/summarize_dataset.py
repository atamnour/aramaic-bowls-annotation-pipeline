from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .io_utils import IMAGE_EXTENSIONS, save_csv


def count_images(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def summarize_processed_dataset(
    root: Path,
    letters_subdir: str = "letters/images",
    masks_subdir: str = "letters/masks",
    patches_subdir: str = "patches",
) -> List[Dict]:
    """Summarize processed dataset counts per bowl/image folder."""
    rows: List[Dict] = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        rows.append(
            {
                "image_id": item.name,
                "num_letter_images": count_images(item / letters_subdir),
                "num_letter_masks": count_images(item / masks_subdir),
                "num_patches": count_images(item / patches_subdir),
            }
        )
    return rows


def save_text_summary(rows: List[Dict], out_txt: Path) -> None:
    total_items = len(rows)
    total_letters = sum(int(r["num_letter_images"]) for r in rows)
    total_masks = sum(int(r["num_letter_masks"]) for r in rows)
    total_patches = sum(int(r["num_patches"]) for r in rows)
    avg_letters = total_letters / total_items if total_items else 0.0

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Aramaic Bowls Processed Dataset Summary\n")
        f.write("======================================\n\n")
        f.write(f"Number of image folders: {total_items}\n")
        f.write(f"Total letter images: {total_letters}\n")
        f.write(f"Total letter masks: {total_masks}\n")
        f.write(f"Total patches: {total_patches}\n")
        f.write(f"Average letters per image: {avg_letters:.2f}\n\n")
        f.write("Per-image details:\n")
        for row in rows:
            f.write(
                f"- {row['image_id']}: letters={row['num_letter_images']}, "
                f"masks={row['num_letter_masks']}, patches={row['num_patches']}\n"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a processed Aramaic bowls dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Processed dataset root.")
    parser.add_argument("--out_dir", type=Path, default=Path("dataset_summary"))
    parser.add_argument("--letters_subdir", type=str, default="letters/images")
    parser.add_argument("--masks_subdir", type=str, default="letters/masks")
    parser.add_argument("--patches_subdir", type=str, default="patches")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = summarize_processed_dataset(
        args.root,
        letters_subdir=args.letters_subdir,
        masks_subdir=args.masks_subdir,
        patches_subdir=args.patches_subdir,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, args.out_dir / "dataset_summary.csv")
    save_text_summary(rows, args.out_dir / "dataset_summary.txt")
    print(f"Saved summary to {args.out_dir}")


if __name__ == "__main__":
    main()
