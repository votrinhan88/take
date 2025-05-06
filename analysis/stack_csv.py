import csv
from pathlib import Path
import re


def main(path: str, verbose: bool = False) -> Path:
    """Stack all CSV files in `path` into one CSV after validating identical headers.

    The output filename is inferred from the input filenames by removing any
    "run=<value>" suffix from the stem. Returns the created output file path.
    If `verbose` is True, prints a short stacking summary.
    """
    folder = Path(path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Not a valid directory: {path}")

    csv_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not csv_files:
        raise ValueError(f"No CSV files found in: {path}")

    headers = None
    rows = []
    file_sample_counts = []

    for file_path in csv_files:
        with file_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {file_path}")

            current_headers = list(reader.fieldnames)
            if headers is None:
                headers = current_headers
            elif current_headers != headers:
                raise ValueError(
                    f"Header mismatch in {file_path}. Expected {headers}, got {current_headers}"
                )

            current_rows = list(reader)
            rows.extend(current_rows)
            file_sample_counts.append((file_path.name, len(current_rows)))

    def _trim_run_suffix(stem):
        # Remove patterns like "_run=3", "-run=abc", or trailing "run=5".
        trimmed = re.sub(r"([_-]?run=[^_-]+)$", "", stem)
        return trimmed.rstrip("_-") or "stacked"

    trimmed_stems = {_trim_run_suffix(p.stem) for p in csv_files}
    output_stem = trimmed_stems.pop() if len(trimmed_stems) == 1 else "stacked"
    output_path = folder / f"{output_stem}.csv"

    if headers is None:
        raise ValueError(f"No headers found in CSV files under: {path}")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        name_header = "File"
        count_header = "Count"
        
        type_width = 5
        name_width = max(
            len(str(output_path)),
            *(len(file_name) for file_name, _ in file_sample_counts),
        )
        count_width = max(5, *(len(str(count)) for _, count in file_sample_counts))

        print("Stack summary:")
        print(f"{'':>{type_width}}  {name_header:<{name_width}}  {count_header:>{count_width}}")
        for i, (file_name, sample_count) in enumerate(file_sample_counts):
            print(
                f"{f'[{i}]':>{type_width}}  {file_name:<{name_width}}  {sample_count:>{count_width}}"
            )
        print()
        print(f"{'Out':>{type_width}}  {str(output_path):<{name_width}}  {len(rows):>{type_width}}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stack CSV files in a directory.")
    parser.add_argument("--path", help="Path to the directory containing CSV files.")
    args = parser.parse_args()

    output_file = main(path=args.path, verbose=True)
