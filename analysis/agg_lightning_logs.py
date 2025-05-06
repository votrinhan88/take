from typing import Optional

import os
import pandas as pd
import shutil

def aggregate_lightning_logs(
    input_dir: str,
    output_dir: Optional[str] = "./results",
    delete: bool = False,
    verbose: bool = True,
):
    """Aggregate and organize Lightning metrics.csv logs.

    Args:
      `input_dir`: Path of to-be-aggregated Lightning logs for an experiment.
      `output_dir`: Path to save output .csv files. Defaults to `"./results"`.
      `verbose`: Defaults to `True`.

    Example:
    ```
    aggregate_lightning_logs(
        input_dir="../logs/clf-logistic-tfidf-agnews",
        output_dir="../results/clf-logistic-tfidf-agnews",
    )
    ```
    """
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    expt = os.path.basename(input_dir)

    for run in sorted(os.listdir(input_dir)):
        # Check if run_dir is a directory
        version_dir = os.path.join(input_dir, run, "logs")
        if not os.path.exists(version_dir):
            print(f"Warning: {version_dir} not found. Skipping run {run}.")
            continue

        # Get metrics from the latest version
        versions = sorted(
            [d for d in os.listdir(version_dir) if d.startswith("version_")],
            key=lambda d: int(d.split("_")[1]),
            reverse=True,
        )
        for v in versions:
            csv_in = os.path.join(version_dir, v, "metrics.csv")
            csv_out = os.path.join(output_dir, f"{expt} - run {run}.csv")
            if os.path.exists(csv_in):
                df = pd.read_csv(csv_in)
                df = aggregate_metrics(df, group_key="epoch")
                df.to_csv(csv_out, index=False)
                if verbose:
                    print(f"Done: {csv_in} --> {csv_out}")
                break
        else:
            print(f"Warning: No metrics.csv found in any version_* folder in {run}")
    
    if delete:
        shutil.rmtree(input_dir)
        if verbose:
            print(f"Deleted: {input_dir}")


def last_valid(series: pd.Series) -> pd.Series:
    """Group by 'epoch' and for each metric column, take the last non-null value."""
    return series.dropna().iloc[-1] if not series.dropna().empty else None


def aggregate_metrics(df: pd.DataFrame, group_key: str = "epoch") -> pd.DataFrame:
    if group_key not in df.columns:
        raise ValueError(f"'{group_key}' column not found in the input file.")

    agg_dict = {col: last_valid for col in df.columns if col != group_key}
    df_out = df.groupby(group_key).agg(agg_dict).reset_index()
    return df_out
