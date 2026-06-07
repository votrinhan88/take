# TAKE: Trajectory-Aware Knowledge Estimation for Text Dataset Distillation

Official implementation of the paper **"TAKE: Trajectory-Aware Knowledge Estimation for Text Dataset Distillation"**, accepted at **ECML-PKDD 2026**.

## Repository Structure
- `src/` — installable Python package
  - `src/models/` — classifiers, encoders, LLMs, modules
  - `src/finetune/` — fine-tuning callbacks, collators, templates
  - `src/generate/` — generation utilities
  - `src/influence/` — influence function scorers
  - `src/metrics/` — text quality and diversity metrics
  - `src/prototypes/` — prototype/distillation methods
  - `src/utils/` — callbacks, data utilities, metadata
- `run.py` — interactive TUI launcher for experiments
- `expts/` — experiment scripts
- `datasets/` — raw and embedded
- `models/` — pretrained and finetuned
- `results/` — raw, processed, and reported

## Quick Start

### 1. Setup Environment
```bash
uv sync
source .venv/bin/activate
```

### 2. Run Experiments
```bash
python run.py
```
Interactive TUI — select infrastructure (Slurm/Local), spec, duration, experiment, then fill in arguments. Submits or runs directly.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{take2026,
  title     = {TAKE: Trajectory-Aware Knowledge Estimation for Text Dataset Distillation},
  author    = {Tri-Nhan Vo, Dang Nguyen, Sunil Gupta},
  booktitle = {Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year      = {2026},
}
```
