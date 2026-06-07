import argparse
import sys
from pathlib import Path

import pandas as pd


def clone_col(src: Path, col: str, col_out: str, overwrite: bool, dst: Path) -> None:
    df = pd.read_csv(src)
    if col not in df.columns:
        print(f"  [skip] column '{col}' not found in {src.name} (columns: {list(df.columns)})")
        return
    if col_out in df.columns and not overwrite:
        sys.exit(f"  [error] column '{col_out}' already exists in {src.name} — use --overwrite to replace it")
    df[col_out] = df[col]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"  {src.name} -> {dst}  |  '{col}' cloned to '{col_out}'")


def resolve_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            sys.exit(f"No CSV files found in directory: {path}")
        return files
    sys.exit(f"Path does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone a column in CSV files.")
    parser.add_argument("--path", required=True, type=Path, help="CSV file or folder of CSV files")
    parser.add_argument("--col", required=True, help="Source column to clone")
    parser.add_argument("--col_out", default="val_acc", help="Name for the new column (default: val_acc)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite col_out if it already exists (default: error)")
    parser.add_argument("--output", type=Path, default=None, help="Output file or folder (default: overwrite in-place)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    files = resolve_files(args.path)
    print(f"Cloning '{args.col}' -> '{args.col_out}' across {len(files)} file(s):")

    for src in files:
        if args.output is None:
            dst = src
        elif args.path.is_dir():
            dst = args.output / src.name
        else:
            dst = args.output
        clone_col(src, args.col, args.col_out, args.overwrite, dst)


if __name__ == "__main__":
    main()
