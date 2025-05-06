# Text Dataset Distillation (TextDD)
A comprehensive framework for text dataset distillation and condensation experiments.


## Repository Structure
- `src/` — installable Python package
  - `src/models/` — classifiers, encoders, LLMs, modules
  - `src/finetune/` — fine-tuning callbacks, collators, templates
  - `src/generate/` — generation utilities
  - `src/influence/` — influence function scorers
  - `src/metrics/` — text quality and diversity metrics
  - `src/prototypes/` — prototype/distillation methods
  - `src/utils/` — callbacks, data utilities, metadata
- `pipelines/` — user-facing entrypoint scripts
- `configs/` — experiment configuration YAMLs
- `experiments/` — sbatch job definitions
- `references/` — external dependencies (submodules)
- `datasets/` — raw and embedded datasets (gitignored)
- `models/` — pretrained and finetuned weights (gitignored)
- `analysis/` — aggregation scripts and report notebooks
- `results/` — training logs and raw outputs (gitignored)
- `slurm/` — SLURM job stdout/stderr (gitignored)

## Quick Start

### 1. Setup Environment
```bash
uv sync
source .venv/bin/activate
```

### 2. Prepare Embeddings (Optional)
```bash
python pipelines/embed.py --config=emb-<encoder>-<dataset>-<setting>
```
- **Available encoders**: `tfidf`, `glove`, `e5`, `minilm`
- **Available datasets**: `agnews`, `imdb`
- **Available settings**: `train`, `test`

### 3. Condense a Dataset
```bash
python pipelines/condense.py --config=cds-<dataset>-<llm>-<mode>
```
- **Available datasets**: `agnews`, `imdb`
- **Available LLMs**: `gemma3_270m`
- **Available modes**: `dense`, `sparse`

### 4. Evaluate Distilled Datasets
```bash
python pipelines/classify.py --config=clfeval-<classifier>-<encoder>-<dataset>
```
- **Available classifiers**: `logistic`, `nbayes`, `svm`
- **Available encoders**: `tfidf`
- **Available datasets**: `agnews`, `imdb`
