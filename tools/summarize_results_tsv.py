#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize MTIL TSV results.")
    parser.add_argument("tsv", help="Input TSV with columns: dataset, top1")
    parser.add_argument(
        "--append-mean",
        action="store_true",
        help="Append or replace the mean_top1 row in the TSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.tsv)
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty TSV: {path}")

    header = lines[0]
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, value = parts[0], parts[1]
        if name == "mean_top1":
            continue
        rows.append((name, value))

    numeric = []
    for name, value in rows:
        if value == "NA":
            continue
        numeric.append((name, float(value)))

    if not numeric:
        raise ValueError(f"No numeric top1 values found in {path}")

    mean = sum(value for _, value in numeric) / len(numeric)
    print(f"[summary] {path}")
    for name, value in numeric:
        print(f"{name}\t{value:.2f}")
    print(f"mean_top1\t{mean:.4f}")
    print(f"count\t{len(numeric)}")

    if args.append_mean:
        output_lines = [header]
        output_lines.extend(f"{name}\t{value}" for name, value in rows)
        output_lines.append(f"mean_top1\t{mean:.4f}")
        path.write_text("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
