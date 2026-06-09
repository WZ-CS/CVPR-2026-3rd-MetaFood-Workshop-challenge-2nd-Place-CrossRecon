#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossrecon.config import load_config
from crossrecon.pipeline import CrossReconPipeline


def parse_ids(value):
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CrossRecon on MetaFood eating videos.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--video-dir", default="data/videos", help="Directory containing challenge mp4 videos.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--items", default="", help="Optional comma-separated item ids, e.g. 1,5,17.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N discovered videos.")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = CrossReconPipeline(config)
    results = pipeline.run(
        video_dir=Path(args.video_dir),
        output_dir=Path(args.output_dir),
        item_ids=parse_ids(args.items),
        limit=args.limit,
    )
    print(f"Processed {len(results)} item(s). Results written to {args.output_dir}/volumes.csv")


if __name__ == "__main__":
    main()

