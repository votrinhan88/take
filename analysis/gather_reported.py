"""
Gather reported experiment files from results/processed into results/reported/<table>.

Tables and their sources:
  main-cls      eval_cls/*/*/{rand,randeda,take}
  main-nli      eval_nli/*/*/{rand,randeda,take}
  abla-components  agnews+qqp x {albert,logistic/siamlog} for rand,kmeans-base,kmeans-cond,take-donttake,take
  abla-kernels     agnews+qqp x {albert,logistic/siamlog} for take + all take-kernels/*
"""

import shutil
import warnings
from pathlib import Path

PROCESSED = Path("results/processed")
REPORTED = Path("results/reported")

CLS_DATASETS = ["agnews", "imdb", "sst2"]
CLS_MODELS = ["albert", "logistic", "textcnn", "textrnn"]
NLI_DATASETS = ["mnlim", "qnli", "qqp"]
NLI_MODELS = ["albert", "siamlog"]

ABLA_CLS_DATASETS = ["agnews"]
ABLA_CLS_MODELS = ["albert", "logistic"]
ABLA_NLI_DATASETS = ["qqp"]
ABLA_NLI_MODELS = ["albert", "siamlog"]

KERNELS = ["constant", "cosine", "first", "last", "linear"]  # donttake handled separately


def copy_dir(src: Path, dst: Path, missing: list[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)


def copy_file(src: Path, dst: Path, missing: list[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / src.name)


def gather_main(table: str, eval_dir: str, datasets: list[str], models: list[str]) -> list[str]:
    missing = []
    splits = ["base", "rand", "randeda", "take"]
    for ds in datasets:
        for mdl in models:
            exp = f"clf-{ds}-{mdl}"
            for split in splits:
                src = PROCESSED / eval_dir / exp / split
                dst = REPORTED / table / exp / split
                copy_dir(src, dst, missing)
    return missing


def gather_abla_components(datasets_cls, models_cls, datasets_nli, models_nli) -> list[str]:
    missing = []
    table = "abla-components"
    splits = ["rand", "kmeans-base", "kmeans-cond", "take"]

    for eval_dir, datasets, models in [
        ("eval_cls", datasets_cls, models_cls),
        ("eval_nli", datasets_nli, models_nli),
    ]:
        for ds in datasets:
            for mdl in models:
                exp = f"clf-{ds}-{mdl}"
                for split in splits:
                    src = PROCESSED / eval_dir / exp / split
                    dst = REPORTED / table / exp / split
                    copy_dir(src, dst, missing)
                # take-donttake lives in take-kernels/donttake
                src = PROCESSED / eval_dir / exp / "take-kernels" / "donttake"
                dst = REPORTED / table / exp / "take-donttake"
                copy_dir(src, dst, missing)
    return missing


def gather_abla_kernels(datasets_cls, models_cls, datasets_nli, models_nli) -> list[str]:
    missing = []
    table = "abla-kernels"

    for eval_dir, datasets, models in [
        ("eval_cls", datasets_cls, models_cls),
        ("eval_nli", datasets_nli, models_nli),
    ]:
        for ds in datasets:
            for mdl in models:
                exp = f"clf-{ds}-{mdl}"
                # take (default kernel)
                src = PROCESSED / eval_dir / exp / "take"
                dst = REPORTED / table / exp / "take"
                copy_dir(src, dst, missing)
                # all take-kernels subdirs
                kernels_root = PROCESSED / eval_dir / exp / "take-kernels"
                if not kernels_root.exists():
                    missing.append(str(kernels_root))
                else:
                    for kernel_dir in sorted(kernels_root.iterdir()):
                        if kernel_dir.is_dir() and kernel_dir.name != "first":
                            src = kernel_dir
                            dst = REPORTED / table / exp / "take-kernels" / kernel_dir.name
                            copy_dir(src, dst, missing)
    return missing


def main():
    # Wipe and recreate each table directory for a clean gather
    for table in ["main-cls", "main-nli", "abla-components", "abla-kernels"]:
        table_dir = REPORTED / table
        if table_dir.exists():
            shutil.rmtree(table_dir)

    all_missing = {}

    all_missing["main-cls"] = gather_main(
        "main-cls", "eval_cls", CLS_DATASETS, CLS_MODELS
    )
    all_missing["main-nli"] = gather_main(
        "main-nli", "eval_nli", NLI_DATASETS, NLI_MODELS
    )
    all_missing["abla-components"] = gather_abla_components(
        ABLA_CLS_DATASETS, ABLA_CLS_MODELS, ABLA_NLI_DATASETS, ABLA_NLI_MODELS
    )
    all_missing["abla-kernels"] = gather_abla_kernels(
        ABLA_CLS_DATASETS, ABLA_CLS_MODELS, ABLA_NLI_DATASETS, ABLA_NLI_MODELS
    )

    any_missing = False
    for table, paths in all_missing.items():
        if paths:
            any_missing = True
            warnings.warn(f"[{table}] {len(paths)} missing source(s):")
            for p in paths:
                print(f"  MISSING  {p}")

    if not any_missing:
        print("All experiments found and copied successfully.")
    else:
        print("\nDone (with warnings above).")


if __name__ == "__main__":
    main()
