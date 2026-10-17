# ============================================================
# Bee dataset download + verify
# Mendeley Data v6 DOI: 10.17632/8gb9r2yhfc.6
# Verifies local files against manifest.csv
# ============================================================
# Manual download is required because data.mendeley.com sits
# behind Cloudflare bot protection ("Just a moment...").
# Automated public-API / direct-URL downloads return 403.
#
# Workflow:
#   1. Manually download from:
#        https://doi.org/10.17632/8gb9r2yhfc.6
#      Use "Download All" (CC BY 4.0).
#   2. Put the downloaded zip(s) into ./downloads/
#   3. Run:
#        .\.venv\Scripts\python.exe download_dataset.py --check-only
#        .\.venv\Scripts\python.exe download_dataset.py --extract
#        .\.venv\Scripts\python.exe download_dataset.py --build-yolo
#
# manifest.csv columns: split,source_session,image,label
# Example: train,_bee_20230609a,20230609a1002.jpg,20230609a1002.txt
# ============================================================

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MANIFEST = BASE_DIR / "manifest.csv"
DATASET_DIR = BASE_DIR / "dataset"
DETECTION_DIR = DATASET_DIR / "detection"
# User's manual download uses capital-D folder
DATASETS_DIR = BASE_DIR / "Datasets"
DETECTION_DIRS = [d for d in (DETECTION_DIR, DATASETS_DIR / "detection") if d.exists()]
DATASET_ROOTS = [d for d in (DATASET_DIR, DATASETS_DIR) if d.exists()]
DOWNLOADS_DIR = BASE_DIR / "downloads"
VIDEOS_DIR = BASE_DIR / "videos"
YOLO_DIR = BASE_DIR / "dataset_yolo"

DATASET_URL = "https://doi.org/10.17632/8gb9r2yhfc.6"

EXPECTED_VIDEO = "20230711a-fan.mp4"


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        print(f"ERROR: manifest not found: {MANIFEST}")
        sys.exit(1)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


# ------------------------------------------------------------
# Locate files (flexible: Mendeley zips use nested subfolders
# like detection/_bee_XXX/images/*.jpg + labels/*.txt,
# or flat detection/_bee_XXX/*.jpg)
# ------------------------------------------------------------

def build_index(search_roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                index.setdefault(p.name, []).append(p)
    return index


def verify_manifest(rows: list[dict]) -> dict:
    search_roots = DETECTION_DIRS or [DETECTION_DIR]
    if DATASET_ROOTS:
        # also search whole dataset(s)/ (covers pose/, segmentation/, etc.)
        search_roots = DATASET_ROOTS
    index = build_index(search_roots)

    missing_images: list[str] = []
    missing_labels: list[str] = []
    found = 0

    for r in rows:
        img_hits = index.get(r["image"], [])
        lbl_hits = index.get(r["label"], [])
        if img_hits and lbl_hits:
            found += 1
        else:
            if not img_hits:
                missing_images.append(f"{r['source_session']}/{r['image']}")
            if not lbl_hits:
                missing_labels.append(f"{r['source_session']}/{r['label']}")

    by_split = Counter(f"{r['split']}" for r in rows)
    by_session = Counter(r["source_session"] for r in rows)

    return {
        "total": len(rows),
        "found_pairs": found,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "by_split": dict(by_split),
        "by_session": dict(by_session),
    }


def print_report(report: dict) -> None:
    print()
    print("=" * 70)
    print("MANIFEST VERIFICATION")
    print("=" * 70)
    print(f"Manifest rows : {report['total']}")
    print(f"Found pairs   : {report['found_pairs']}")
    print(f"Missing images: {len(report['missing_images'])}")
    print(f"Missing labels: {len(report['missing_labels'])}")
    print(f"By split      : {report['by_split']}")
    print(f"By session    : {report['by_session']}")
    if report["missing_images"][:5]:
        print("\nMissing images (first 5):")
        for m in report["missing_images"][:5]:
            print(f"  - {m}")
    if report["missing_labels"][:5]:
        print("\nMissing labels (first 5):")
        for m in report["missing_labels"][:5]:
            print(f"  - {m}")


# ------------------------------------------------------------
# Extract user-downloaded zips
# ------------------------------------------------------------

def extract_downloads() -> None:
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)
    zips = sorted(DOWNLOADS_DIR.glob("*.zip"))
    if not zips:
        print(f"No zips in {DOWNLOADS_DIR}/")
        print(f"Download manually from {DATASET_URL}")
        print('Place the zip(s) into ./downloads/ and re-run with --extract.')
        return
    for z in zips:
        print(f"Extracting {z.name} -> {DATASET_DIR}/ ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DATASET_DIR)
        print("  done.")


# ------------------------------------------------------------
# Build YOLO layout from manifest:
#   dataset_yolo/images/{train,val,test} + labels/...
# ------------------------------------------------------------

def build_yolo(rows: list[dict]) -> None:
    roots = DATASET_ROOTS or ([DATASET_DIR] if True else [])
    index = build_index(roots)
    copied, skipped = 0, 0
    for r in rows:
        split = r["split"]
        for kind, fname in (("images", r["image"]), ("labels", r["label"])):
            srcs = index.get(fname, [])
            if not srcs:
                skipped += 1
                continue
            # Prefer a path containing the source_session when ambiguous
            src = next(
                (p for p in srcs if r["source_session"] in str(p)),
                srcs[0],
            )
            dst = YOLO_DIR / kind / split / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
    print(f"YOLO layout: {YOLO_DIR}/ (copied {copied}, skipped {skipped})")


# ------------------------------------------------------------
# Videos check (tracking/behavior MP4s come from the same
# Mendeley v6 download; run_bee_behavior.py needs
# videos/20230711a-fan.mp4)
# ------------------------------------------------------------

def check_videos() -> None:
    print()
    print("=" * 70)
    print("VIDEOS CHECK")
    print("=" * 70)
    if not VIDEOS_DIR.exists():
        print(f"Missing folder: {VIDEOS_DIR}")
        return
    vids = sorted(p.name for p in VIDEOS_DIR.glob("*") if p.is_file())
    print(f"videos/ contains {len(vids)} file(s).")
    for v in vids[:20]:
        print(f"  - {v}")
    if EXPECTED_VIDEO not in vids:
        print(f"\nNOTE: {EXPECTED_VIDEO} is missing.")
        print("run_bee_behavior.py and preview.py expect it.")
        print("It comes from the v6 tracking/behavior MP4s:")
        print(f"  {DATASET_URL}")
        print("Copy/rename the fanning video to:")
        print(f"  videos/{EXPECTED_VIDEO}")


# ------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify/prepare bee dataset against manifest.csv"
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify local files against manifest.csv",
    )
    ap.add_argument(
        "--extract",
        action="store_true",
        help="Extract ./downloads/*.zip into ./dataset/",
    )
    ap.add_argument(
        "--build-yolo",
        action="store_true",
        help="Build dataset_yolo/{images,labels}/{train,val,test}/ from manifest",
    )
    ap.add_argument(
        "--check-videos",
        action="store_true",
        help="List videos/ and check for expected test video",
    )
    args = ap.parse_args()

    if args.extract:
        extract_downloads()

    rows = load_manifest()
    report = verify_manifest(rows)
    print_report(report)

    if args.build_yolo:
        build_yolo(rows)

    if args.check_videos or True:
        # videos check is cheap; always show unless extracting only
        check_videos()

    if report["found_pairs"] == 0:
        print()
        print("No dataset files found locally.")
        print(f"1. Download v6 from {DATASET_URL} (Download All, CC BY 4.0)")
        print("2. Put zip(s) into ./downloads/")
        print("3. Run: python download_dataset.py --extract")
        print("4. Re-run: python download_dataset.py --check-only")


if __name__ == "__main__":
    main()
