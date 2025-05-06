import os
from typing import Any, Callable, Literal, Optional, Sequence
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd


def load_preset_plt():
    """Config preset parameters for matplotlib."""
    plt.rcParams.update(
        {
            "figure.titlesize": 12,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "font.size": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1,
        }
    )


def maybe(task: callable, exception: Exception) -> Any | None:
    """Catch an expected exception then continue specified task.

    Args:
    + `task`: Task with expected exception.
    + `exception`: Exception.

    Returns:
    + Original output if task is performed, `None` if exception occurs.
    """
    try:
        return task()
    except exception:
        return None


def walk_contents(
    path: str,
    whitelist: Optional[Sequence[str]] = None,
    blacklist: Optional[Sequence[str]] = None,
    filetype: str = "",
    pre_callbacks: Sequence[Callable] = [],
    post_callbacks: Sequence[Callable] = [],
    level: int = 0,
):
    """Recursively walk through the contents of a specified directory.

    Args:
    + `path`: Path to directory.
    + `filetype`: File extensions to filter in. Defaults to `''`.
    + `pre_callbacks`: Callbacks to perform before walking into subfolders.  \
        Defaults to `[]`.
    + `post_callbacks`: Callbacks to perform after walking into subfolders.  \
        Defaults to `[]`.
    + `level`: Current recursive level. Defaults to `0`.
    """
    subfolders = []
    files = []

    # Iterate to save subfolders or suitable filetypes
    for content in sorted(os.listdir(path)):
        # White/Black-list
        if whitelist is not None:
            whitelist_flags = [wl in content for wl in whitelist]
            if not all(whitelist_flags):
                continue
        if blacklist is not None:
            blacklist_flags = [bl in content for bl in blacklist]
            if any(blacklist_flags):
                continue

        if os.path.isdir(os.path.join(path, content)):
            subfolders.append(content)
        else:
            if content.lower().endswith(filetype):
                files.append(content)

    # Pre-walk callbacks
    for callback in pre_callbacks:
        callback(path=path, files=files, level=level)

    # Recursively walk subfolder
    for subfolder in subfolders:
        walk_contents(
            path=os.path.join(path, subfolder),
            whitelist=whitelist,
            blacklist=blacklist,
            filetype=filetype,
            pre_callbacks=pre_callbacks,
            post_callbacks=post_callbacks,
            level=level + 1,
        )

    # Post-walk callbacks
    for callback in post_callbacks:
        callback(path=path, files=files, level=level)


def list_contents_pre_callback(
    path: str,
    files: list[str],
    level: int,
    indent: int = 4,
    tree_like: bool = True,
):
    """Print contents of a directory.

    Args:
    + `path`: Path to current directory.
    + `files`: Files of current directory.
    + `level`: Current recursive level.
    + `indent`: Number of character of indentation per recursive level.      \
        Defaults to `4`.
    + `tree_like`: Flag to print in a tree-like hierchary. Defaults to `True`.
    """
    if tree_like & (indent >= 3):
        if level == 0:
            pre_pad = ""
        if level > 0:
            pre_pad = ("│" + (indent - 1) * " ") * (level - 1) + (
                "├" + "─" * (indent - 2) + " "
            )
    else:
        pre_pad = indent * level * " "

    if level > 0:
        path = os.path.basename(path)

    print(f"{pre_pad}{path}/")
    for content in files:
        pre_pad = ("│" + (indent - 1) * " ") * level + ("├" + "─" * (indent - 2) + " ")
        print(f"{pre_pad}{content}")


def list_results_from_csv_pre_callback(
    path: str,
    files: list[str],
    level: int,
    metric: str = "val_acc",
    target: Literal["max", "min", "last"] = "max",
    format: Literal["percent", "raw", "full"] = "percent",
    indent: int = 4,
    width: int = 80,
    tree_like: bool = True,
    check_csv: bool = False,
):
    """Print results from csv files in a directory.

    Args:
    + `path`: Path to current directory.
    + `files`: Files of current directory.
    + `level`: Current recursive level.
    + `metric`: Metric to extract from csv files. Defaults to `'val_acc'`.
    + `target`: Target value of metric, one of `'max'`, `'min'`, `'last'`.   \
        Defaults to `'max'`.
    + `format`: One of `'percent'`, `'raw'`, `'full'`. Defaults to `'percent'`.
    + `width`: Width of print screen. Defaults to `80`.
    + `indent`: Number of character of indentation per recursive level.      \
        Defaults to `4`.
    + `tree_like`: Flag to print in a tree-like hierchary. Defaults to `True`.
    + `check_csv`: Flag to check for csv file only. Defaults to `False`.
    """
    results = []
    for content in files:
        # Check to skip non-csv files (if specified)
        if check_csv is True:
            if not content.lower().endswith(".csv"):
                continue
        # Extract target results
        df = pd.read_csv(os.path.join(path, content)).loc[:, [metric]].to_numpy()
        if target == "max":
            best_of_run = np.nanmax(df)
        elif target == "min":
            best_of_run = np.nanmin(df)
        elif target == "last":
            best_of_run = df[-1]
        results.append(best_of_run.item())

    # Format print
    ## Pre-pad, with optional tree-like hiarchery
    if tree_like & (indent >= 3):
        if level == 0:
            pre_pad = ""
        if level > 0:
            pre_pad = ("│" + (indent - 1) * " ") * (level - 1) + (
                "├" + "─" * (indent - 2) + " "
            )
    else:
        pre_pad = indent * level * " "
    ## Path name, shortened if not base
    if level > 0:
        path = os.path.basename(path)
    ## Results
    if len(results) > 0:
        num_runs = f" ({len(results):02} runs)"
        if format == "percent":
            results = f"{np.mean(results)*100:.2f} ± {np.std(results)*100:.2f}%"
        elif format == "raw":
            results = f"{np.mean(results):.4g} ± {np.std(results):.4g}"
        elif format == "full":
            results = f"{np.round(results, 4)}"
        results = f" {results}"
    elif len(results) == 0:
        num_runs = ""
        results = ""
    ## Mid-path in the middle
    mid_path = (
        width - len(pre_pad) - len(path) - len(num_runs) - 1 - len(results)
    ) * " "
    ## Print
    print(f"{pre_pad}{path}{num_runs}:{mid_path}{results}")


def aggregate_results_from_csv_to_dict_callback(
    path: str,
    files: list[str],
    level: int,
    metric: str = "val_acc",
    target: Literal["max", "min", "last"] = "max",
    format: Literal["percent", "raw", "full"] = "percent",
    check_csv: bool = False,
    results_dict: dict[str, Any] = {},
):
    """Aggregate results from csv files in a directory to a dict.

    Args:
    + `path`: Path to current directory.
    + `files`: Files of current directory.
    + `level`: Current recursive level.
    + `metric`: Metric to extract from csv files. Defaults to `'val_acc'`.
    + `target`: Target value of metric, one of `'max'`, `'min'`, `'last'`.   \
        Defaults to `'max'`.
    + `format`: One of `'percent'`, `'raw'`, `'full'`. Defaults to `'percent'`.
    + `check_csv`: Flag to check for csv file only. Defaults to `False`.
    """
    results = []
    for content in files:
        # Check to skip non-csv files (if specified)
        if check_csv is True:
            if not content.lower().endswith(".csv"):
                continue
        # Extract target results
        df = pd.read_csv(os.path.join(path, content)).loc[:, [metric]].to_numpy()
        if target == "max":
            best_of_run = np.nanmax(df)
        elif target == "min":
            best_of_run = np.nanmin(df)
        elif target == "last":
            best_of_run = df[-1]
        results.append(best_of_run.item())

    # Results
    n_runs = len(results)
    if len(results) > 0:
        if format == "percent":
            results = {
                'mean': np.mean(results).item() * 100,
                'std': np.std(results).item() * 100,
                'n_runs': n_runs,
            }
        elif format == "raw":
            results = {
                'mean': np.mean(results).item(),
                'std': np.std(results).item(),
                'n_runs': n_runs,
            }
    else:
        results = {}

    # Parse keys from path
    keys = path.split(os.sep)
    cur_dict = results_dict
    for key in keys:
        if key not in cur_dict:
            cur_dict[key] = {}
        cur_dict = cur_dict[key]
    cur_dict.update(results)


def plot_from_csv_callback(
    path: str,
    files: list[str],
    level: int,
    fig: Figure,
    ax: Axes,
    metric_y: str,
    metric_x: Optional[str] = None,
    interp: bool = False,
    plot_range: Optional[Sequence[int]] = None,
    scale_y: Literal["raw", "percent"] = "raw",
    agg_line: Literal[
        "mean", "median", "min", "max", "full", "accum_min", "accum_max"
    ] = "mean",
    agg_area: Literal["stddev", "stderr", "mimmax"] = "stddev",
    legend_loc: Optional[str] = None,
    check_csv: bool = False,
    preset_plt: bool = True,
):
    """Plot results from csv files in a directory.

    Args:
    + `path`: Path to current directory.
    + `files`: Files of current directory.
    + `level`: Current recursive level.
    + `fig`: Figure handler.
    + `ax`: Axes handler.
    + `metric_y`: Metric for y-axis to extract from csv files.
    + `metric_y`: Metric for x-axis to extract from csv files. Defaults to     \
        `None`, skip for increasing integers.
    + `plot_range`: Lower and upper bounds of plotting epochs. Defaults to     \
        `None`, skip to select widest range available.
    + `scale_y`: Scaling of y-axis in plot: `'raw'` | `'percent'`. `'raw'`: no \
        scaling, `'percent'`: y-axis is scaled as percentages (eg. accuracies).\
        Defaults to `'percent'`.
    + `format`: Format of multiple runs of the same folder: `'mean'` |         \
        `'meanstd'` | `'full'`. `'mean'`: mean of runs along the y-axis,       \
        `'meanstd'`: mean ± 2 std confidence interval, `'full'`: all lines.    \
        Defaults to `'mean'`.
    + `legend_loc`: Location of legend, eg: `'lower left'`, `'center'`, `'upper\
        right'`. Defaults to `None`, skip to best location available.
    + `check_csv`: Flag to check for csv file only. Defaults to `False`.
    + `preset_plt`: Flag to load preset Matplotlib's parameters. Defaults to   \
        `True`.
    """
    if level == 0:
        if preset_plt is True:
            load_preset_plt()

        if fig._suptitle is None:
            fig.suptitle(f"{path}\nFormat: line={agg_line}, area={agg_area}")

    if len(files) == 0:
        return

    # Pre-allocate
    color = None
    ys = []
    xs = []

    # Load & pre-process data
    for i, content in enumerate(files):
        # Check to skip non-csv files (if specified)
        if check_csv is True:
            if not content.lower().endswith(".csv"):
                continue
        # Extract target metrics
        df = pd.read_csv(os.path.join(path, content))
        y = df.loc[:, metric_y].to_numpy().astype(float)
        if metric_x is None:
            x = np.arange(stop=y.shape[0])
        elif metric_x is not None:
            x = df.loc[:, metric_x].to_numpy().astype(float)
        # Plot range
        if plot_range is not None:
            y = y[plot_range[0] : plot_range[1]]
            x = x[plot_range[0] : plot_range[1]]
        if scale_y == "percent":
            y = y * 100
        # Append data
        ys.append(y)
        xs.append(x)

    # Pad to same length & stack
    length = max(*[y.shape[0] for y in ys]) if len(files) > 1 else ys[0].shape[0]

    for i, content in enumerate(files):
        if xs[i].shape[0] < length:
            xs[i] = np.pad(
                xs[i],
                pad_width=[[0, length - xs[i].shape[0]]],
                mode="constant",
                constant_values=np.nan,
            )
        if ys[i].shape[0] < length:
            ys[i] = np.pad(
                ys[i],
                pad_width=[[0, length - ys[i].shape[0]]],
                mode="constant",
                constant_values=np.nan,
            )
    xs = np.stack(xs, axis=0)
    ys = np.stack(ys, axis=0)

    if interp is True:
        for i in range(xs.shape[0]):
            xs[i, :] = np.interp(
                x=np.arange(xs.shape[1]),
                xp=np.arange(xs.shape[1])[~np.isnan(xs[i])],
                fp=xs[i][~np.isnan(xs[i])],
            )
        for i in range(ys.shape[0]):
            ys[i, :] = np.interp(
                x=np.arange(ys.shape[1]),
                xp=np.arange(ys.shape[1])[~np.isnan(ys[i])],
                fp=ys[i][~np.isnan(ys[i])],
            )

    # Warn for inconsistent x
    if np.any(xs.std(axis=0) > 0):
        warnings.warn("Metric x is inconsistent between runs. Their mean is used.")
    x_mean = np.nanmean(xs, axis=0)

    # Plot line(s)
    label = path.replace(fig._suptitle.get_text().split(sep="\n")[0], "")
    if label == "":
        label = "."
    elif label[0] in ["/", "\\"]:
        label = label[1:]

    if agg_line == "full":
        for i, (x, y) in enumerate(zip(xs, ys)):
            line = ax.plot(x, y, color=color, label=label if i == 0 else None)[0]
            if i == 0:
                color = line.get_color()
    else:
        if agg_line == "mean":
            y_line = np.nanmean(ys, axis=0)
        elif agg_line == "median":
            y_line = np.nanmedian(ys, axis=0)
        elif agg_line == "min":
            y_line = np.nanmin(ys, axis=0)
        elif agg_line == "max":
            y_line = np.nanmax(ys, axis=0)
        elif agg_line == "accum_min":
            y_line = np.minimum.accumulate(np.nanmin(ys, axis=0), axis=0)
        elif agg_line == "accum_max":
            y_line = np.maximum.accumulate(np.nanmax(ys, axis=0), axis=0)
        line = ax.plot(x_mean, y_line, color=color, label=label)[0]
        color = line.get_color()

    # Plot area
    if agg_area == "stddev":
        stddev = np.nanstd(ys, axis=0)
        mean = np.nanmean(ys, axis=0)
        y_low = mean - stddev
        y_high = mean + stddev
    elif agg_area == "stderr":
        stderr = np.nanstd(ys, axis=0) / np.sqrt(
            np.count_nonzero(~np.isnan(ys), axis=0)
        )
        mean = np.nanmean(ys, axis=0)
        y_low = mean - stderr
        y_high = mean + stderr
    elif agg_area == "minmax":
        y_low = np.nanmin(ys, axis=0)
        y_high = np.nanmax(ys, axis=0)
    ax.fill_between(x=x_mean, y1=y_low, y2=y_high, color=color, alpha=0.2)

    # Set axis attributes (only once)
    if ax.get_xlabel() == "":
        ax.set(
            ylim=[0, 100] if scale_y == "percent" else None,
            xlim=[np.min(xs), np.max(xs)],
            ylabel=f"{metric_y}%" if scale_y == "percent" else metric_y,
            xlabel=metric_x if metric_x is not None else "iteration",
        )
    # Update axis attributes repeatedly
    ax.set(
        ylim=(
            None
            if scale_y == "percent"
            else [
                min(np.nanmin(ys), ax.get_ylim()[0]),
                max(np.nanmax(ys), ax.get_ylim()[1]),
            ]
        ),
        xlim=[
            min(np.nanmin(xs), ax.get_xlim()[0]),
            max(np.nanmax(xs), ax.get_xlim()[1]),
        ],
    )
    # Legend
    if legend_loc != "off":
        ax.legend(loc=legend_loc)


from typing import Sequence


def traverse_nested(input:dict|list, keys:Sequence[str|int], strict:bool=False):
    """Traverses a nested dictionary using a list of keys.

    Args:
    + input (dict): The dictionary to traverse.
    + keys (Sequence[str]): A list of keys representing the traversal path.

    Returns:
    + The value at the specified path or None if the path does not exist.
    
    Examples:
    >>> a = {'b': {'c': {'d': {'e': 0}}}}
    >>> traverse_xmldict(input=a, keys=['b', 'c', 'd', 'e'])
    0
    >>> traverse_xmldict(input=a, keys=['b', 'f'])
    None
    """
    current = input
    for k in keys:
        if isinstance(current, (dict)) and k in current.keys():
            current = current[k]
        elif isinstance(current, Sequence):
            current = current[k]
        else:
            if strict:
                raise ValueError(f'Could not find key `{k}` for input {input}')
            else:
                return None  # Return None if the path does not exist
    return current