from collections.abc import Callable
from copy import deepcopy
import csv
import os
import sys

from datasets import ClassLabel, DatasetDict, concatenate_datasets
import evaluate
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, SparseEncoder
import torch
from torch import nn, Tensor
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import yaml


sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

from pipelines.classify import train_classifier
from pipelines.expt_utils import (
    get_classifier,
    get_dataset,
    get_dataloader,
    get_encoder,
    get_llm_model,
    get_llm_tokenizer,
)
from src.models.classifiers import ClassifierMetadata, ClassifierTrainer, SiameseLogistic
from src.models.encoders import EncoderMetadata, GloVeEncoder
from src.models.llms import LLMMetadata
from src.models.modules import EasyDataAugmentation
from src.utils.metadata import DatasetMetadata
from src.utils.pythonic.dict_utils import traverse_dictlist


class BERTSequencePairMap:
    supported_datasets = {
        "mnlim": {
            "text_cols": ["premise", "hypothesis"],
            "preprocess_tokenizer_kwargs": {
                "padding_side": "right",
                "truncation_side": "right",
                "max_length": 128,
                "padding": "longest",
                "truncation": True,
            },
        },
        "qqp": {"text_cols": ["question1", "question2"]},
    }

    def __init__(self, tokenizer, dataset: str):
        self.tokenizer = tokenizer
        self.dataset = self._validate_args("dataset", dataset)

        self.metadata = DatasetMetadata(dataset=self.dataset)
        self.col_0 = self.supported_datasets[dataset]["text_cols"][0]
        self.col_1 = self.supported_datasets[dataset]["text_cols"][1]

    def _validate_args(self, arg: str, value):
        if arg == "dataset":
            if value not in self.supported_datasets.keys():
                raise ValueError(
                    f"Unsupported dataset: {value}. Supported: {self.supported_datasets}"
                )
            return value

    def preprocess(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        def map_fn(batch: dict) -> dict:
            tokenized = self.tokenizer(
                text=batch[self.col_0],
                text_pair=batch[self.col_1],
                add_special_tokens=True,
                truncation=True,
            )
            batch.update(tokenized)
            return batch

        dataset = dataset.map(map_fn, batched=True)
        return dataset


class SiamesePairMap:
    supported_datasets = {
        "mnlim": {"text_cols": ["premise", "hypothesis"]},
        "qqp": {"text_cols": ["question1", "question2"]},
    }

    def __init__(self, dataset: str):
        self.dataset = self._validate_args("dataset", dataset)

        self.metadata = DatasetMetadata(dataset=self.dataset)
        self.col_0 = self.supported_datasets[dataset]["text_cols"][0]
        self.col_1 = self.supported_datasets[dataset]["text_cols"][1]

    def _validate_args(self, arg: str, value):
        if arg == "dataset":
            if value not in self.supported_datasets.keys():
                raise ValueError(
                    f"Unsupported dataset: {value}. Supported: {self.supported_datasets}"
                )
            return value

    def preprocess(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        def map_fn(batch: dict) -> dict:
            batch["text_0"] = batch[self.col_0]
            batch["text_1"] = batch[self.col_1]
            return batch

        dataset = dataset.map(map_fn, batched=True)
        return dataset


class SequenceClassificationMetrics:
    supported_metrics = ["accuracy", "precision", "recall", "f1"]

    def __init__(self, metrics: str | list = "all"):
        self.metrics = self._validate_args("metrics", metrics)

        self.metric_objs = {}
        for metric in self.metrics:
            self.metric_objs[metric] = evaluate.load(metric)

    def _validate_args(self, arg: str, value):
        if arg == "metrics":
            if isinstance(value, str):
                if value == "all":
                    return [m for m in self.supported_metrics]
                else:
                    value = [value]

            for metric in value:
                if metric not in self.supported_metrics:
                    raise ValueError(
                        f"Unsupported metric: {metric}. Supported: {self.supported_metrics}"
                    )
            return value

    def __call__(self, eval_pred, **kwargs) -> dict:
        return self.compute_metrics(eval_pred)

    def compute_metrics(self, eval_pred, **kwargs) -> dict:
        preds, labels = eval_pred
        results = {}
        preds = np.argmax(preds, axis=1)
        for metric in self.metrics:
            metric_obj = self.metric_objs[metric]
            if metric in ["accuracy"]:
                results.update(metric_obj.compute(predictions=preds, references=labels))
            elif metric in ["precision", "recall", "f1"]:
                results.update(
                    metric_obj.compute(predictions=preds, references=labels, average="macro")
                )
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metrics={self.metrics})"


class MetricsCSVCallback(TrainerCallback):
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.rows = []
        self.fieldnames = ["epoch", "step"]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # filter to training metrics only — eval metrics are handled by on_evaluate
            train_keys = {"loss", "learning_rate", "grad_norm", "epoch"}
            train_logs = {k: v for k, v in logs.items() if k in train_keys}
            if not train_logs:
                return

            row = {"epoch": state.epoch, "step": state.global_step, **train_logs}
            self.rows.append(row)

            new_keys = [k for k in row if k not in self.fieldnames]
            if new_keys:
                self.fieldnames.extend(new_keys)
                self._rewrite()
            else:
                self._append(row)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            row = {"epoch": state.epoch, "step": state.global_step, **metrics}
            self.rows.append(row)

            new_keys = [k for k in row if k not in self.fieldnames]
            if new_keys:
                self.fieldnames.extend(new_keys)
                self._rewrite()
            else:
                self._append(row)

    def _append(self, row: dict):
        file_exists = os.path.exists(self.output_path)
        write_header = not file_exists or os.path.getsize(self.output_path) == 0

        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, restval="")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _rewrite(self):
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, restval="")
            writer.writeheader()
            writer.writerows(self.rows)


class ConfigFactory:
    supported_overrides = [
        "dataset_path",
        "randsubset",
        "eda",
        "batch_size",
        "embed_dimnum_epochs",
        "lr",
        "wdecay",
        "steps",
    ]
    config: dict

    def __init__(self, **kwargs):
        self.args = kwargs
        self.override_args = {k: v for k, v in kwargs.items() if k in self.supported_overrides}

        if self.args.get("base_config") is not None:
            parser = ConfigParser(globals=globals(), locals=locals())
            path = parser.parse_path_config(config=self.args["base_config"], ext="yaml")
            self.base_config = yaml.safe_load(stream=open(file=path, mode="r"))
            self.base_config["metaconfig"]["args"] = self.args
        else:
            self.base_config = None
            self.dataset = self.args["dataset"]
            self.encoder = self.args.get("encoder")
            self.classifier = self.args["classifier"]

        assert self.dataset is not None, "Dataset must be specified"
        assert self.dataset in ["mnlim", "qqp"], f"Unsupported dataset: {self.dataset}"
        assert self.classifier is not None, "Classifier must be specified"

        self.metadata_dataset = DatasetMetadata(dataset=self.dataset)
        if self.classifier in ["albert"]:
            self.metadata_llm = LLMMetadata(model=self.classifier)
            assert self.encoder is None, "Encoder should not be specified for albert."
        else:
            self.metadata_classifier = ClassifierMetadata(model=self.classifier)
            self.metadata_encoder = EncoderMetadata(model=self.encoder)

    def get_config(self) -> dict:
        if self.base_config is not None:
            config = deepcopy(self.base_config)
        else:
            config = {
                "metaconfig": self.get_set_metaconfig(),
                "dataset": self.get_config_dataset(),
                "models": self.get_config_models(),
                "trainer": self.get_config_trainer(),
            }

        self.config = self.override_config(config=config, **self.args)
        return deepcopy(self.config)

    def get_set_metaconfig(self) -> dict:
        self.metaconfig = {
            "name": f"clf-{self.dataset}-{self.classifier}",
            "expt": "clf",
            "path": "./logs/classify",
            "args": self.args,
            "run": "eval:f'{run}'",
        }
        return self.metaconfig

    def get_config_dataset(self) -> dict:
        splits = {
            "mnlim": ["train", "validation_matched", "validation_mismatched"],
            "qqp": ["train", "validation"],
        }
        config = {
            "abbrev": self.dataset,
            "splits": splits[self.dataset],
        }
        if self.classifier in ["siamlog"]:
            config.update(
                {"loader_kwargs": {"batch_size": 256, "shuffle": {"train": True}, "num_workers": 4}}
            )
        return config

    def get_config_models(self) -> dict:
        config = {}

        if self.classifier in ["albert"]:
            config["classifier"] = self.metadata_llm.get_preset_model()
            extra_kwargs = {
                "num_labels": self.metadata_dataset.num_classes,
                "id2label": self.metadata_dataset.classes,
                "label2id": {k: i for i, k in enumerate(self.metadata_dataset.classes)},
            }
            config["classifier"]["kwargs"].update(extra_kwargs)
            config["tokenizer"] = self.metadata_llm.get_preset_tokenizer()

        elif self.classifier in ["siamlog"]:
            config = {}
            splits_corpus = "+".join(self.metadata_dataset.get_preset()["splits_corpus"])
            if self.encoder == "tfidf":
                config["encoder"] = {
                    "abbrev": "tfidf",
                    "init_with": "torch_load",
                    "torch_load_kwargs": {
                        "f": f"./pretrained/encoders/{self.encoder}/{self.dataset}/emb-{self.encoder}-{self.dataset}-{splits_corpus}-3072d.pt",
                        "weights_only": False,
                    },
                    "embed_dim": 3072,
                }
            elif self.encoder == "glove":
                config["encoder"] = self.metadata_encoder.get_preset()
                config["encoder"]["kwargs"]["embed_level"] = "sentence"
            else:
                raise ValueError(f"Unsupported encoder: {self.encoder}")

            config["classifier"] = self.metadata_classifier.get_preset()
            extra_kwargs = {
                "input_dim": config["encoder"]["embed_dim"],
                "num_classes": self.metadata_dataset.num_classes,
            }
            config["classifier"]["kwargs"].update(extra_kwargs)
        else:
            raise ValueError(f"Unsupported classifier: {self.args['classifier']}")

        return config

    def get_config_trainer(self) -> dict:
        path = self.metaconfig["path"]
        name = self.metaconfig["name"]

        if self.classifier in ["albert"]:
            metric_for_best_model = {
                "mnlim": "eval_validation_matched_accuracy",
                "qqp": "eval_validation_accuracy",
            }[self.dataset]
            config = {
                "fit_with": "tf-trainer",
                "args": {
                    "output_dir": f"eval:f'{path}/{name}/{name}-run={{run}}'",
                    "per_device_train_batch_size": 32,
                    "per_device_eval_batch_size": 32,
                    "gradient_accumulation_steps": 1,
                    "optim": "adamw_torch",
                    "learning_rate": 3e-5,  # ALBERT paper: 1e-5 for xxlarge, 3e-5 for base
                    "lr_scheduler_type": "linear",
                    "warmup_ratio": 0.1,  # 10% of total steps for warmup
                    "weight_decay": 0.01,
                    "num_train_epochs": 3,  # 3-5 typical; ALBERT paper used 3 for MNLI
                    "logging_strategy": "steps",
                    "logging_steps": 1000,
                    "logging_first_step": True,
                    "disable_tqdm": True,
                    "eval_strategy": "steps",
                    "eval_steps": 1000,
                    "eval_on_start": True,
                    "save_strategy": "best",
                    "save_steps": 1000,
                    "save_total_limit": 2,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": metric_for_best_model,
                    "greater_is_better": True,
                    "fp16": True,
                    "report_to": "none",
                },
                "data_collator": {
                    "Class": "eval:DataCollatorWithPadding",
                    "kwargs": {},
                },
                "compute_metrics": "eval:SequenceClassificationMetrics(metrics='all')",
                "callbacks": [
                    f"eval:MetricsCSVCallback(output_path=f'{path}/{name}/{name}-run={{run}}.csv')"
                ],
            }

        elif self.classifier in ["siamlog"]:
            config = {
                "fit_with": "clf_trainer",
                "Class": "eval:ClassifierTrainer",
                "kwargs": {
                    "optimizer_kw": {
                        "classifier": {
                            "Class": "eval:torch.optim.AdamW",
                            "kwargs": {
                                "lr": 0.003,
                                "weight_decay": 0.0005,
                            },
                        },
                    },
                    "num_classes": self.metadata_dataset.num_classes,
                    "paired": True,
                },
                "L_trainer_kw": {
                    "callbacks": {
                        "printer": {"event_name": "train_epoch_end"},
                        "csv_logger": {
                            "save_path": f"eval:f'{path}/{name}/{name}-run={{run}}.csv'",
                            "event_name": "train_epoch_end",
                        },
                    },
                    "check_val_every_n_epoch": 1,
                    "devices": "auto",
                    "enable_checkpointing": False,
                    "enable_progress_bar": False,
                    "max_epochs": 20,
                    "logger": False,
                },
                "fit_kw": {},
                "save_state_dict": f"eval:f'{path}/{name}/{name}-run={{run}}.pt'",
            }

        else:
            raise ValueError(f"Unsupported classifier: {self.classifier}")

        return config

    def override_config(self, config: dict, **kwargs) -> dict:
        keys_avail = ["dataset_path", "batch_size", "randsubset", "eda"]
        keys_avail.extend(["num_epochs", "lr", "wdecay", "steps"])

        for k, v in self.override_args.items():
            assert k in keys_avail, f"Unknown key: {k}"

            if k == "dataset_path":
                config["dataset"]["splits"].remove("train")
                config["dataset"]["splits_custom"] = {
                    "train": {"init_with": "from_csv", "from_csv_kwargs": {"path_or_paths": v}}
                }
                config["dataset"]["cast_label"] = True

            elif k == "batch_size":
                if self.classifier in ["albert"]:
                    config["trainer"]["args"]["per_device_train_batch_size"] = v
                    config["trainer"]["args"]["per_device_eval_batch_size"] = v
                else:
                    config["dataset"]["loader_kwargs"]["batch_size"] = v

            elif k == "randsubset":
                config["dataset"]["randsubset"] = v

            elif k == "eda":
                config["dataset"]["eda"] = v

            elif k == "embed_dim":
                msg = f"`embed_dim` override not supported for {self.classifier}`"
                assert self.classifier in ["siamlog"], msg
                if config["models"]["encoder"]["abbrev"] == "tfidf":
                    path_encoder = config["models"]["encoder"]["torch_load_kwargs"]["f"]
                    config["models"]["encoder"]["torch_load_kwargs"]["f"] = path_encoder.replace(
                        "3072d.pt", f"{v}d.pt"
                    )
                elif config["models"]["encoder"]["abbrev"] == "glove":
                    config["models"]["encoder"]["kwargs"]["embed_dim"] = v
                else:
                    raise ValueError(
                        f"Unsupported `encoder`: {config['models']['encoder']['abbrev']}"
                    )
                config["models"]["encoder"]["embed_dim"] = v
                config["models"]["classifier"]["kwargs"]["input_dim"] = v

            elif k == "num_epochs":
                if self.classifier in ["albert"]:
                    config["trainer"]["args"]["num_train_epochs"] = v
                else:
                    config["trainer"]["L_trainer_kw"]["max_epochs"] = v

            elif k == "lr":
                if self.classifier in ["albert"]:
                    config["trainer"]["args"]["learning_rate"] = v
                else:
                    config["trainer"]["kwargs"]["optimizer_kw"]["classifier"]["kwargs"]["lr"] = v

            elif k == "wdecay":
                if self.classifier in ["albert"]:
                    config["trainer"]["args"]["weight_decay"] = v
                else:
                    config["trainer"]["kwargs"]["optimizer_kw"]["classifier"]["kwargs"][
                        "weight_decay"
                    ] = v  # noqa: E501

            elif k == "steps":
                msg = f"`steps` override not supported for {self.classifier}`"
                assert self.classifier in ["albert"], msg
                config["trainer"]["args"]["logging_steps"] = v
                config["trainer"]["args"]["eval_steps"] = v
                config["trainer"]["args"]["save_steps"] = v

            else:
                raise ValueError(f"Unknown key: {k}")
        return config

    def export_config(self, path_config: str | None = None, verbose: bool = True) -> str:
        if path_config is None:
            path = self.metaconfig["path"]
            name = self.metaconfig["name"]
            args_str = ""
            for k, v in self.args.items():
                if k in ["base_config", "run", "n_runs", "dataset", "classifier"]:
                    continue
                args_str += f"-{k}={v}"
            path_config = f"{path}/{name}/{name}-config{args_str}.yaml"

        os.makedirs(os.path.dirname(path_config), exist_ok=True)
        with open(path_config, "w") as f:
            yaml.dump(data=self.config, stream=f, sort_keys=False)

        if verbose:
            print(f"Config exported to {path_config}.")

        return path_config


def preprocess_data(dataset: DatasetDict, tokenizer, classifier: bool, config: dict) -> DatasetDict:
    metadata = DatasetMetadata(dataset=config["abbrev"])

    if config.get("cast_label") is not None:
        dataset = dataset.cast_column(column="label", feature=ClassLabel(names=metadata.classes))

    if config.get("randsubset") is not None:
        n_subset = int(len(dataset["train"]) * config["randsubset"])
        dataset["train"] = dataset["train"].shuffle().select(range(n_subset))

    if config.get("eda") is not None:
        eda = EasyDataAugmentation(**config["eda"]["kwargs"])
        dataset["train"] = eda.augment_dataset(dataset["train"]).shuffle()

    if classifier in ["albert"]:
        mapper = BERTSequencePairMap(tokenizer=tokenizer, dataset=config["abbrev"])
        dataset = mapper.preprocess(dataset=dataset)
    else:
        mapper = SiamesePairMap(dataset=config["abbrev"])
        dataset = mapper.preprocess(dataset=dataset)

        if config["abbrev"] == "mnlim":
            dataset["test"] = dataset.pop("validation_matched")
        elif config["abbrev"] == "qqp":
            dataset["test"] = dataset.pop("validation")

    return dataset


def expt_clf2(config: dict, run: int | str = 0):
    # Template: https://huggingface.co/docs/transformers/v5.2.0/en/tasks/sequence_classification
    classifier_abbrev = config["models"]["classifier"]["abbrev"]
    if classifier_abbrev in ["albert"]:
        model = get_llm_model(config=config["models"]["classifier"])
        tokenizer = get_llm_tokenizer(config=config["models"]["tokenizer"])
        dataset: DatasetDict = get_dataset(config["dataset"])
        dataset = preprocess_data(
            dataset=dataset,
            tokenizer=tokenizer,
            classifier=classifier_abbrev,
            config=config["dataset"],
        )
        metric_fn = SequenceClassificationMetrics(metrics="all")
        trainer = Trainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset,
            args=TrainingArguments(**config["trainer"]["args"]),
            data_collator=config["trainer"]["data_collator"]["Class"](
                **config["trainer"]["data_collator"]["kwargs"]
            ),
            compute_metrics=metric_fn,
            callbacks=config["trainer"]["callbacks"],
        )
        trainer.train()
    else:
        dataset: DatasetDict = get_dataset(config["dataset"])
        dataset = preprocess_data(
            dataset=dataset,
            tokenizer=None,
            classifier=classifier_abbrev,
            config=config["dataset"],
        )
        dataloader = get_dataloader(dataset=dataset, **config["dataset"]["loader_kwargs"])
        encoder = get_encoder(config=config["models"]["encoder"], dataloader=dataloader["train"])
        classifier = get_classifier(config=config["models"]["classifier"])
        classifier, metrics = train_classifier(
            config=config["trainer"],
            classifier=classifier,
            encoder=encoder,
            dataloader=dataloader,
        )
        return classifier, metrics


if __name__ == "__main__":
    import argparse
    from expt_utils import ConfigParser, TypeArgparse, pprint, rename_runs

    args = argparse.ArgumentParser()
    args_group = args.add_argument_group("Metaconfig arguments")
    args_group.add_argument("--base_config", type=str)
    args_group.add_argument("--run", type=TypeArgparse.int_or_str, default=0)
    args_group.add_argument("--n_runs", type=int, default=1)
    args_group = args.add_argument_group("Dataset arguments")
    args_group.add_argument("--dataset", type=str, choices=["mnlim", "qqp"])
    args_group.add_argument("--dataset_path", type=str)
    args_group.add_argument("--batch_size", type=int)
    args_group.add_argument("--randsubset", type=float)
    args_group.add_argument("--eda", type=int)
    args_group = args.add_argument_group("Model arguments")
    args_group.add_argument("--encoder", type=str, choices=["tfidf", "glove"])
    args_group.add_argument("--classifier", type=str, choices=["albert", "siamlog"])
    args_group.add_argument("--embed_dim", type=int)
    args_group = args.add_argument_group("Optimization arguments")
    args_group.add_argument("--num_epochs", type=int)
    args_group.add_argument("--lr", type=float)
    args_group.add_argument("--wdecay", type=float)
    args_group.add_argument("--steps", type=int)
    args = args.parse_args()
    custom_args = {k: v for k, v in vars(args).items() if v is not None}

    parser = ConfigParser(globals=globals(), locals=locals())
    config_factory = ConfigFactory(**custom_args)
    config = config_factory.get_config()
    config_factory.export_config(verbose=True)
    for run in rename_runs(run=args.run, n_runs=args.n_runs):
        config_run = parser.parse_eval_config(deepcopy(config), parse_flag="eval:")
        pprint(config_run)
        expt_clf2(config=config_run, run=run)
