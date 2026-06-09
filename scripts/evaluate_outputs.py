#!/usr/bin/env python
import argparse
import csv
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check CrossRecon output files.")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "volumes.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    rows = list(csv.DictReader(open(csv_path, "r", encoding="utf-8")))
    missing = []
    non_positive = []
    for row in rows:
        item_id = int(row["id"])
        slug = row["food"].lower().replace(" ", "_").replace("-", "_")
        for state in ("before", "after"):
            mesh = output_dir / "meshes" / f"{item_id:02d}_{slug}_{state}.obj"
            if not mesh.exists():
                missing.append(str(mesh))
        if float(row["before_volume_ml"]) <= 0:
            non_positive.append(f"{item_id} before")
        if float(row["after_volume_ml"]) < 0:
            non_positive.append(f"{item_id} after")

    if missing or non_positive:
        if missing:
            print("Missing meshes:")
            for path in missing:
                print(f"  {path}")
        if non_positive:
            print("Invalid volumes:")
            for item in non_positive:
                print(f"  {item}")
        raise SystemExit(1)

    print(f"OK: {len(rows)} item(s), {len(rows) * 2} mesh files checked.")


if __name__ == "__main__":
    main()

