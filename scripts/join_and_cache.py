"""Milestone 1: validate the ShapeNetSDF x Captions join; build the CLIP cache.

Usage:
    python scripts/join_and_cache.py --category chair --cache-dir data/text_cache
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf.data.caption_index import ShapeNetCaptionIndex, build_clip_embedding_cache
from eg_hrdf.data.join_validator import load_category_ids, validate_join


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="None = all categories")
    parser.add_argument("--cache-dir", default="data/text_cache")
    parser.add_argument("--min-rate", type=float, default=0.95)
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    captions = ShapeNetCaptionIndex.load()
    print(f"captions loaded: {len(captions)} entries")

    ids = load_category_ids(args.category)
    report = validate_join(captions, ids, min_rate=args.min_rate)
    print("join report:", report.summary())
    if report.missing_ids:
        print("sample missing ids:", report.missing_ids[:5])

    os.makedirs(args.cache_dir, exist_ok=True)
    filtered = {mid: captions[mid] for mid in ids if mid in captions}
    with open(os.path.join(args.cache_dir, "captions_filtered.json"), "w") as f:
        json.dump(filtered, f)
    if not args.skip_clip:
        index = build_clip_embedding_cache(filtered, args.cache_dir, device=args.device)
        print(f"CLIP cache written: {len(index)} embeddings -> {args.cache_dir}")


if __name__ == "__main__":
    main()
