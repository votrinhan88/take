# Change path
import os
import sys

repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from copy import deepcopy

from datasets import Dataset, DatasetDict
import mauve
import numpy as np
import pprint
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from utils.metadata import DatasetMetadata
from models.finetune.templates import TextTemplate
from models.generate.utils import generate
from utils.pythonic.numeric_utils import balanced_partition
from utils.pythonic.dict_utils import flatten_dictlist

from models.metrics.perplexity import Perplexity
from models.metrics.distinctn import DistinctN
from models.metrics.selfbleu import SelfBLEU
from models.metrics.dcr import DistanceToClosestRecord


def merge_generation_config(
    old: GenerationConfig | dict, new: GenerationConfig | dict
) -> GenerationConfig:
    if isinstance(old, GenerationConfig):
        old = old.to_diff_dict()
    if isinstance(new, GenerationConfig):
        new = new.to_diff_dict()
    old.update(new)
    return GenerationConfig(**old)


class SampleGenerationCallback(TrainerCallback):
    genconfig = GenerationConfig(
        max_new_tokens=512,  # Cap at 512 for debug purpose.
        do_sample=True,
        num_beams=1,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=4,
    )

    def __init__(
        self,
        output: str,
        dataset: str,
        tokenizer,
        genconfig: GenerationConfig | None = None,
        num_samples_uncond: int = 6,
        num_samples_cond_per_class: int = 3,
        every_n_steps: int = 1000,
    ):
        self.output = output
        self.dataset = dataset
        self.tokenizer = tokenizer
        if genconfig is not None:
            self.genconfig = merge_generation_config(old=self.genconfig, new=genconfig)
        self.num_samples_uncond = num_samples_uncond
        self.num_samples_cond_per_class = num_samples_cond_per_class
        self.every_n_steps = every_n_steps

        self.template = TextTemplate(dataset=dataset, default_task="generation")
        self.classes = self.template.classes

        os.makedirs(os.path.dirname(self.output), exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        with open(self.output, "a", encoding="utf-8") as f:
            if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
                model.eval()

                f.write(f" STEP {state.global_step} ".center(88, "=") + "\n")
                counter = 0

                f.write(f"[Unconditional Samples]\n")
                texts = generate(
                    model=model,
                    tokenizer=self.tokenizer,
                    num_samples=self.num_samples_uncond,
                    prompt=self.template.template_generation(label=None),
                    genconfig=self.genconfig,
                    ensure_bos_token=True,
                )
                for t in texts:
                    f.write(f"[Sample {counter}]\n")
                    f.write(t)
                    f.write("\n\n")
                    counter += 1

                for k in self.classes:
                    f.write(f"[Conditional Samples - {k}]\n")
                    texts = generate(
                        model=model,
                        tokenizer=self.tokenizer,
                        num_samples=self.num_samples_cond_per_class,
                        prompt=self.template.template_generation(label=k),
                        genconfig=self.genconfig,
                        ensure_bos_token=True,
                    )
                    for t in texts:
                        f.write(f"[Sample {counter}]\n")
                        f.write(t)
                        f.write("\n\n")
                        counter += 1
        return control


class SampleInferenceCallback(TrainerCallback):
    genconfig_generation = GenerationConfig(
        max_new_tokens=512,  # Cap at 512 for debug purpose.
        do_sample=True,
        num_beams=1,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=4,
    )

    # Greedy decoding
    genconfig_inference = GenerationConfig(
        max_new_tokens=32,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
    )

    def __init__(
        self,
        output: str,
        dataset: str,
        tokenizer,
        genconfig_generation: GenerationConfig | None = None,
        genconfig_inference: GenerationConfig | None = None,
        num_samples_per_class: int = 3,
        every_n_steps: int = 1000,
    ):
        self.output = output
        self.dataset = dataset
        self.tokenizer = tokenizer
        if genconfig_generation is not None:
            self.genconfig_generation = merge_generation_config(
                old=self.genconfig_generation, new=genconfig_generation
            )
        if genconfig_inference is not None:
            self.genconfig_inference = merge_generation_config(
                old=self.genconfig_inference, new=genconfig_inference
            )

        self.num_samples_per_class = num_samples_per_class
        self.every_n_steps = every_n_steps

        self.template = TextTemplate(dataset=dataset, default_task="inference")
        self.classes = self.template.classes

        os.makedirs(os.path.dirname(self.output), exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps != 0 or state.global_step == 0:
            return

        model = kwargs["model"]
        with open(self.output, "a", encoding="utf-8") as f:
            model.eval()

            f.write(f" STEP {state.global_step} ".center(88, "=") + "\n")
            counter = 0

            for k in self.classes:
                generated = generate(
                    model=model,
                    tokenizer=self.tokenizer,
                    num_samples=self.num_samples_per_class,
                    prompt=self.template.template_generation(label=k),
                    genconfig=self.genconfig_generation,
                    ensure_bos_token=True,
                )
                texts: list[str] = self.template.parse(generated, task="generation")["text"]
                for t in texts:
                    generated_with_pred = generate(
                        model=model,
                        tokenizer=self.tokenizer,
                        num_samples=1,
                        prompt=self.template.template_inference(text=t),
                        genconfig=self.genconfig_inference,
                        ensure_bos_token=True,
                    )[0]
                    # self.infer(model=model, text=t)
                    f.write(f"[Synthetic Sample {counter} - Label: {k}]\n")
                    f.write(generated_with_pred)
                    f.write("\n\n")
                    counter += 1


class FinetuneEvalCallback(TrainerCallback):
    """Callback for evaluating and logging metrics during model training, including saving metrics
    to CSV/log files and printing sample outputs.

    Args:
    + `dataset`: Dataset name.
    + `tokenizer`.
    + `encoder`.
    + `output_csv`: Path to CSV file for saving flattened metrics (one row per step).
    + `output_log`: Path to log file for saving detailed sample outputs and metrics.
    + `metrics`: List of metric names to compute. Available: `"length"`, `"perplexity"`, \
        `"distinctn"`, `"dcr"`, `"selfbleu"`, `"mauve"`, `"accuracy"`. Defaults to `"full"`.
    + `gen_config`: Generation config for generation. Defaults to greedy decoding.
    + `every_n_steps`: Frequency (in steps) to run evaluation and logging. Defaults to `100`.
    + `num_samples_eval`: Number of samples to evaluate per step. Defaults to `100`.
    """

    avail_metrics = ["length", "perplexity", "distinctn", "dcr", "selfbleu", "mauve", "accuracy"]

    genconfig_default = GenerationConfig(
        max_new_tokens=512,  # Cap at 512 for debug purpose.
        do_sample=True,
        num_beams=1,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=4,
    )
    genconfig_greedy = GenerationConfig(
        max_new_tokens=32,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
    )

    def __init__(
        self,
        tokenizer,
        encoder,
        output_csv: str | None,
        output_log: str | None,
        dataset: str,
        metrics: list[str] | str = "full",
        genconfig: GenerationConfig | None = None,
        every_n_steps: int = 1000,
        num_samples_eval: int = 500,
        num_samples_log: int = 10,
    ):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.dataset = dataset
        self.output_csv = output_csv
        self.output_log = output_log
        self.metrics = self._validate_args("metrics", metrics)
        self.genconfig = self._validate_args("genconfig", genconfig)
        self.every_n_steps = every_n_steps
        self.num_samples_eval = num_samples_eval
        self.num_samples_log = num_samples_log

        self.metadata = DatasetMetadata(dataset=dataset)
        self.classes = self.metadata.classes
        self.pprint = pprint.PrettyPrinter(indent=1, width=120, compact=True, sort_dicts=False)
        self.template = TextTemplate(dataset=dataset, default_task="generation")

        if self.output_csv is not None:
            # Create an empty CSV file
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            with open(self.output_csv, "w", encoding="utf-8") as f:
                pass

    def _validate_args(self, arg, value):
        if arg == "metrics":
            metrics = value
            if isinstance(value, str):
                if value == "full":
                    metrics = deepcopy(self.avail_metrics)
                else:
                    metrics = [metrics]
            for m in metrics:
                if m not in self.avail_metrics:
                    raise ValueError(f"Unknown metric: {m}. Available: {self.avail_metrics}")
            return metrics

        elif arg == "genconfig":
            if value is not None:
                return merge_generation_config(old=self.genconfig_default, new=value)
            else:
                return self.genconfig_default

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        # Log metrics at every_n_steps, and end
        if not (
            (state.global_step % self.every_n_steps == 0) or (state.global_step == state.max_steps)
        ):
            return

        model = kwargs["model"]
        train_dataloader = kwargs["train_dataloader"]
        eval_dataloader = kwargs["eval_dataloader"]

        metrics = {"step": state.global_step, "epoch": state.epoch}
        dataset = DatasetDict()
        # Get subsample dataset
        dataset["train"] = self.get_subsampled_dataset(train_dataloader)
        if eval_dataloader is not None:
            dataset["eval"] = self.get_subsampled_dataset(eval_dataloader)
        dataset["synth"] = self.get_synthetic_dataset(model=model)

        metrics.update(self.compute_metrics(model=model, dataset=dataset))

        self.log_metrics(metrics=metrics, dataset=dataset)

    def get_subsampled_dataset(self, dataloader: DataLoader) -> Dataset:
        # Subsample a dataset for evaluation
        dataset = {"text": [], "label": [], "tokens": [], "embeddings": []}
        counts = [0] * len(self.classes)
        quota = balanced_partition(total=self.num_samples_eval, num_parts=len(self.classes))

        for i, batch in enumerate(dataloader):
            if all(counts[k] >= quota[k] for k in range(len(self.classes))):
                break

            batch_size = batch["input_ids"].shape[0]
            text = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            text_parsed = self.template.parse(text)
            text, label = text_parsed["text"], text_parsed["label"]
            embeddings = self.encoder(text)

            for k in range(len(self.classes)):
                if counts[k] >= quota[k]:
                    continue
                idx_k = [i for i in range(batch_size) if label[i] == self.classes[k]]
                idx_k = idx_k[0 : max(0, quota[k] - counts[k])]
                if len(idx_k) == 0:
                    continue
                dataset["text"].extend([text[i] for i in idx_k])
                dataset["label"].extend([label[i] for i in idx_k])
                dataset["embeddings"].append(embeddings[idx_k])
                counts[k] += len(idx_k)
        # Re-tokenize to get clean tokens (no special, padding, truncation, etc.)
        tok = self.tokenizer(text=dataset["text"], add_special_tokens=False)
        dataset["tokens"] = tok["input_ids"]
        dataset["embeddings"] = torch.cat(tensors=dataset["embeddings"], dim=0)
        dataset: Dataset = Dataset.from_dict(dataset).with_format("torch")
        return dataset

    def get_synthetic_dataset(self, model) -> Dataset:
        dataset = {"text": [], "label": [], "tokens": [], "embeddings": []}

        num_samples_k = balanced_partition(total=self.num_samples_eval, num_parts=len(self.classes))
        for k in range(len(self.classes)):
            generated = generate(
                model=model,
                tokenizer=self.tokenizer,
                num_samples=num_samples_k[k],
                prompt=self.template(label=k),
                genconfig=self.genconfig,
                ensure_bos_token=True,
                validation_fn=self.template.validate,
            )
            text = self.template.parse(generated, task="generation")["text"]
            dataset["text"].extend(text)
            dataset["label"].extend([self.classes[k]] * num_samples_k[k])

        dataset["tokens"] = self.tokenizer(text=dataset["text"], add_special_tokens=False)[
            "input_ids"
        ]
        if self.encoder is not None:
            dataset["embeddings"] = self.encoder(dataset["text"])
        dataset = Dataset.from_dict(dataset).with_format("torch")
        return dataset

    def compute_metrics(self, model, dataset: DatasetDict) -> dict:
        model_train_state = model.training
        model.eval()

        metrics = {}

        if ("dcr" in self.metrics) or ("mauve" in self.metrics):
            embeddings = {}
            for split in dataset.keys():
                if f"embeddings" not in dataset[split].column_names:
                    raise ValueError(
                        f"Embeddings are required for dcr/mauve metrics, "
                        f"but not found in split '{split}'."
                    )
                embeddings[split]: Tensor = dataset[split]["embeddings"][:]

        if "length" in self.metrics:
            results = {}
            for split in dataset:
                length = torch.tensor([len(t) for t in dataset[split]["tokens"][:]])
                results[split] = length.float().mean(dim=0).item()
            metrics["length"] = results

        if "perplexity" in self.metrics:
            results = {}

            def add_special_tokens(batch: dict) -> dict:
                texts = [
                    self.tokenizer.bos_token
                    + self.template(text=t, label=l, task="generation")
                    + self.tokenizer.eos_token
                    for t, l in zip(batch["text"], batch["label"])
                ]
                batch.update(self.tokenizer(texts, add_special_tokens=False))
                return batch

            dataset_ppl = dataset.map(
                function=add_special_tokens,
                batched=True,
            ).remove_columns(column_names=["text", "label", "tokens", "embeddings"])

            ppl = Perplexity(model=model, tokenizer=self.tokenizer)
            for split in dataset:
                results[split] = ppl(dataset_ppl[split])
            metrics["perplexity"] = results

        if "distinctn" in self.metrics:
            results = {}
            for n in [1, 2, 3]:
                distinctn = DistinctN(n=n)
                results_n = {}
                for split in dataset.keys():
                    results_n[split] = distinctn(tokens=dataset[split]["tokens"][:])["corpus"]
                results[f"distinct-{n}"] = results_n
            metrics["distinctn"] = results

        if "selfbleu" in self.metrics:
            # Can be time-consuming for large datasets - complexity O(N^2)
            results = {}
            selfbleu = SelfBLEU()
            for split in dataset.keys():
                results[split] = selfbleu(tokens=dataset[split]["tokens"][:])["corpus"]
            metrics["selfbleu"] = results

        if "dcr" in self.metrics:
            results = {}
            dcr = DistanceToClosestRecord(distance_fn="euclidean")

            splits_dcr = ["synth", "train", "eval", "synth:train", "synth:eval", "train:eval"]
            if "eval" not in dataset.keys():
                splits_dcr = [s for s in splits_dcr if "eval" not in s]

            for split in splits_dcr:
                split_list = split.split(":")
                emb1 = embeddings[split_list[0]]
                emb2 = None
                if len(split_list) == 2:
                    emb2 = embeddings[split_list[1]]
                results[split] = dcr(input=emb1, other=emb2)
            metrics["dcr"] = results

        if "mauve" in self.metrics:
            results = {}

            for split in ["synth:train", "synth:eval", "train:eval"]:
                split_list = split.split(":")
                if split_list[0] not in dataset.keys() or split_list[1] not in dataset.keys():
                    continue

                emb1 = embeddings[split_list[0]]
                emb2 = embeddings[split_list[1]]
                results[split] = mauve.compute_mauve(p_features=emb1, q_features=emb2).mauve
            metrics["mauve"] = results

        if "accuracy" in self.metrics:
            results = {}
            for split in dataset.keys():
                correct = 0
                total = 0
                for text, label in zip(dataset[split]["text"][:], dataset[split]["label"][:]):
                    generated = generate(
                        model=model,
                        tokenizer=self.tokenizer,
                        num_samples=1,
                        prompt=self.template(text=text, task="inference"),
                        genconfig=self.genconfig_greedy,
                        ensure_bos_token=True,
                    )[0]
                    pred = self.template.parse(generated, task="inference", strict=False)["label"]
                    if pred == label:
                        correct += 1
                    total += 1

                results[split] = correct / total if total > 0 else 0.0
            metrics["accuracy"] = results

        model.train(model_train_state)
        return metrics

    def log_metrics(self, metrics: dict, dataset: DatasetDict):
        if self.output_csv is not None:
            metrics_flattened = flatten_dictlist(metrics)
            is_empty = os.path.getsize(self.output_csv) == 0
            with open(self.output_csv, "a", encoding="utf-8") as f:
                # Write header if file is empty
                if is_empty:
                    header = ",".join(metrics_flattened.keys())
                    f.write(header + "\n")
                line = ",".join([str(v) for v in metrics_flattened.values()])
                f.write(line + "\n")

        if self.output_log is not None:
            with open(self.output_log, "a", encoding="utf-8") as f:
                f.write(f" STEP {metrics['step']} ".center(100, "=") + "\n")

                f.write("[Synthetic samples]\n")
                idx = torch.randperm(len(dataset["synth"]))[0 : self.num_samples_log].tolist()
                for i, ii in enumerate(idx):
                    f.write(f"[Sample {i} - {dataset['synth']['label'][ii]}]\n")
                    f.write(dataset["synth"]["text"][ii] + "\n\n")

                f.write("[Metrics]\n")
                metrics_str = self.pprint.pformat(metrics)
                f.write(metrics_str + "\n\n")


class StateDictCheckpointCallback(TrainerCallback):
    def __init__(self, save_dir: str, every_n_steps: int = 5000):
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if (state.global_step % self.every_n_steps == 0 and state.global_step > 0) or (
            state.global_step == state.max_steps
        ):
            path = os.path.join(self.save_dir, f"checkpoint_step_{state.global_step}.pt")
            self.save(model=model, path=path)

    def save(self, model, path):
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(obj=model.state_dict(), f=path)
