import csv
from pathlib import Path


def compact_file(path: Path, verbose: bool = False) -> Path:
    """Compact a sparse HF-trainer CSV by merging rows with the same (epoch, step).

    CsvLoggerHF writes one sparse row per event (train log, eval split), leaving
    all other columns empty. This merges those rows into one dense row per step
    by taking the first non-empty value for each column.
    Overwrites the input file in place. Returns the path.
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        if "epoch" not in fieldnames or "step" not in fieldnames:
            if verbose:
                print(f"  [skip] no epoch/step columns: {path}")
            return path
        rows = list(reader)

    merged: dict[tuple, dict] = {}
    order: list[tuple] = []
    for row in rows:
        key = (row["epoch"], row["step"])
        if key not in merged:
            merged[key] = {k: "" for k in fieldnames}
            order.append(key)
        for k, v in row.items():
            if v != "" and merged[key][k] == "":
                merged[key][k] = v

    output_rows = [merged[k] for k in order]

    if len(output_rows) == len(rows):
        if verbose:
            print(f"  [noop] already compacted: {path}")
        return path

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    if verbose:
        print(f"  {len(rows):>4} → {len(output_rows):>4} rows  {path}")

    return path


def main(path: str, verbose: bool = False) -> list[Path]:
    p = Path(path)
    if p.is_file():
        targets = [p]
    elif p.is_dir():
        targets = sorted(f for f in p.rglob("*.csv") if f.is_file())
        if not targets:
            raise ValueError(f"No CSV files found under: {path}")
    else:
        raise ValueError(f"Path does not exist: {path}")

    if verbose and len(targets) > 1:
        print(f"Compacting {len(targets)} CSV files under {p}:")

    results = [compact_file(t, verbose=verbose) for t in targets]
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compact sparse HF-trainer CSV(s).")
    parser.add_argument("--path", help="Path to a CSV file or a folder (recurses into subfolders).")
    args = parser.parse_args()
    if args.path is None:
        parser.print_help()
    else:
        main(path=args.path, verbose=True)
