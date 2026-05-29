from __future__ import annotations
import argparse
from copy import deepcopy
import os
import sys

repo_path = os.path.abspath(os.path.join(__file__, "../.."))
assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from src.metadata import DatasetMetadata, EncoderMetadata, ClassifierMetadata, LLMMetadata


def get_parser():
    # fmt: off
    p = argparse.ArgumentParser()
    g = p.add_argument_group("Metaconfig arguments")
    g.add_argument("--base_config", type=str)
    g.add_argument("--run", default=0)
    g.add_argument("--n_runs", type=int, default=1)
    g = p.add_argument_group("Dataset arguments")
    g.add_argument("--dataset", type=str, choices=DatasetMetadata.supported_nli)
    g.add_argument("--dataset_path", type=str)
    g.add_argument("--randsubset", type=float)
    g.add_argument("--eda", type=int)
    g.add_argument("--batch_size", type=int)
    g = p.add_argument_group("Model arguments")
    g.add_argument("--encoder", type=str, choices=EncoderMetadata.supported + [None])
    g.add_argument("--classifier", type=str, choices=ClassifierMetadata.supported_nli + LLMMetadata.supported_nli)
    g.add_argument("--embed_dim", type=int)
    g = p.add_argument_group("Optimization arguments")
    g.add_argument("--num_epochs", type=int)
    g.add_argument("--lr", type=float)
    g.add_argument("--wdecay", type=float)
    # fmt: on
    return p


class BERTSequencePairMap:
    def __init__(self, tokenizer, dataset: str):
        self.tokenizer = tokenizer
        self.metadata = DatasetMetadata(dataset=dataset)
        self.col_0 = self.metadata.text_keys[0]
        self.col_1 = self.metadata.text_keys[1]

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
    def __init__(self, dataset: str):
        self.metadata = DatasetMetadata(dataset=dataset)
        self.col_0 = self.metadata.text_keys[0]
        self.col_1 = self.metadata.text_keys[1]

    def preprocess(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        def map_fn(batch: dict) -> dict:
            batch["text_0"] = batch[self.col_0]
            batch["text_1"] = batch[self.col_1]
            return batch

        dataset = dataset.map(map_fn, batched=True)
        return dataset


class ConfigFactory:
    supported_overrides = [
        "dataset_path",
        "randsubset",
        "eda",
        "batch_size",
        "embed_dim",
        "num_epochs",
        "lr",
        "wdecay",
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
        assert self.dataset in DatasetMetadata.supported_nli, f"Unsupported dataset: {self.dataset}"
        assert self.classifier is not None, "Classifier must be specified"

        self.metadata_dataset = DatasetMetadata(dataset=self.dataset)
        if self.classifier in LLMMetadata.supported_nli:
            self.metadata_llm = LLMMetadata(model=self.classifier)
            assert self.encoder is None, "Encoder should not be specified for LLM classifiers."
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
            "path": "./results/raw/eval_nli",
            "args": self.args,
            "run": "eval:f'{run}'",
        }
        return self.metaconfig

    def get_config_dataset(self) -> dict:
        splits = {
            "mnlim": ["train", "validation_matched", "validation_mismatched"],
            "qqp": ["train", "validation"],
            "qnli": ["train", "validation"],
        }
        config = {
            "abbrev": self.dataset,
            "splits": splits[self.dataset],
        }
        if self.classifier in ["siamlog"]:
            config.update({
                "loader_kwargs": {"batch_size": 256, "shuffle": {"train": True}, "num_workers": 4}
            })
        return config

    def get_config_models(self) -> dict:
        config = {}

        if self.classifier in LLMMetadata.supported_nli:
            config["classifier"] = self.metadata_llm.get_preset_model()
            extra_kwargs = {
                "num_labels": self.metadata_dataset.num_classes,
                "id2label": {i: k for i, k in enumerate(self.metadata_dataset.classes)},
                "label2id": {k: i for i, k in enumerate(self.metadata_dataset.classes)},
            }
            config["classifier"]["kwargs"].update(extra_kwargs)
            config["tokenizer"] = self.metadata_llm.get_preset_tokenizer()

        elif self.classifier in ["siamlog"]:
            config = {}
            if self.encoder == "tfidf":
                config["encoder"] = {
                    "abbrev": "tfidf",
                    "init_with": "self_load",
                    "Class": "eval:Tfidf",
                    "kwargs": {"embed_dim": 3072, "sparse": True},
                    "self_load_kwargs": {
                        "f": f"./models/pretrained/encoders/{self.encoder}/{self.dataset}/emb-{self.encoder}-{self.dataset}-3072d.pkl",
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
            raise ValueError(f"Unsupported classifier: {self.classifier}")

        return config

    def get_config_trainer(self) -> dict:
        path = self.metaconfig["path"]
        name = self.metaconfig["name"]

        if self.classifier in ClassifierMetadata.supported_nli:
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

        elif self.classifier in LLMMetadata.supported_nli:
            metric_for_best_model = {
                "mnlim": "eval_validation_matched_accuracy",
                "qqp": "eval_validation_accuracy",
                "qnli": "eval_validation_accuracy",
            }[self.dataset]
            if self.classifier == "albert":
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
                        f"eval:CsvLoggerHF(output_path=f'{path}/{name}/{name}-run={{run}}.csv')"
                    ],
                }

        else:
            raise ValueError(f"Unsupported classifier: {self.classifier}")

        return config

    def override_config(self, config: dict, **kwargs) -> dict:
        for k, v in self.override_args.items():

            if k == "dataset_path":
                if os.path.isdir(v):
                    run_str = str(kwargs.get("run", ""))
                    last = run_str.split("-")[-1]  # -v<number> suffix
                    if not (last.startswith("v") and last[1:].isdigit()):
                        raise ValueError(
                            f"dataset_path is a folder but args.run={run_str!r} has no -v<number> suffix"
                        )
                    matches = glob.glob(os.path.join(v, f"*-{last}.csv"))
                    if len(matches) != 1:
                        raise ValueError(
                            f"Expected exactly 1 CSV matching *-{last}.csv in {v!r}, found {len(matches)} matches."
                        )
                    v = matches[0]

                config["dataset"]["splits"].remove("train")
                config["dataset"]["splits_custom"] = {
                    "train": {"init_with": "from_csv", "from_csv_kwargs": {"path_or_paths": v}}
                }
                config["dataset"]["cast_label"] = True

            elif k == "batch_size":
                if self.classifier in LLMMetadata.supported_nli:
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
                    path_encoder = config["models"]["encoder"]["self_load_kwargs"]["f"]
                    config["models"]["encoder"]["self_load_kwargs"]["f"] = path_encoder.replace("3072d.pkl", f"{v}d.pkl")
                elif config["models"]["encoder"]["abbrev"] == "glove":
                    config["models"]["encoder"]["kwargs"]["embed_dim"] = v
                else:
                    raise ValueError(
                        f"Unsupported `encoder`: {config['models']['encoder']['abbrev']}"
                    )
                config["models"]["encoder"]["embed_dim"] = v
                config["models"]["classifier"]["kwargs"]["input_dim"] = v

            elif k == "num_epochs":
                if self.classifier in LLMMetadata.supported_nli:
                    config["trainer"]["args"]["num_train_epochs"] = v
                else:
                    config["trainer"]["L_trainer_kw"]["max_epochs"] = v

            elif k == "lr":
                if self.classifier in LLMMetadata.supported_nli:
                    config["trainer"]["args"]["learning_rate"] = v
                else:
                    config["trainer"]["kwargs"]["optimizer_kw"]["classifier"]["kwargs"]["lr"] = v

            elif k == "wdecay":
                if self.classifier in LLMMetadata.supported_nli:
                    config["trainer"]["args"]["weight_decay"] = v
                else:
                    config["trainer"]["kwargs"]["optimizer_kw"]["classifier"]["kwargs"][
                        "weight_decay"
                    ] = v  # noqa: E501

            else:
                raise ValueError(f"Unknown key: {k}")
        return config

    def export_config(self, path_config: str | None = None, verbose: bool = True) -> str:
        if path_config is None:
            path = self.metaconfig["path"]
            name = self.metaconfig["name"]
            args_str = ""
            for k, v in self.args.items():
                if k in ["base_config", "run", "n_runs", "dataset", "encoder", "classifier", "dataset_path"]:
                    continue
                args_str += f"-{k}={v}"
            path_config = f"{path}/{name}/{name}-config{args_str}.yaml"

        os.makedirs(os.path.dirname(path_config), exist_ok=True)
        with open(path_config, "w") as f:
            yaml.dump(data=self.config, stream=f, sort_keys=False)

        if verbose:
            print(f"Exported config to {path_config}.")

        return path_config


def preprocess_data(dataset: DatasetDict, tokenizer, classifier: bool, config: dict) -> DatasetDict:
    metadata = DatasetMetadata(dataset=config["abbrev"])

    if config.get("cast_label") is not None:
        if dataset["train"].features["label"].dtype in ["string", "large_string"]:

            def map_fn(batch: dict) -> dict:
                batch["label_int"] = metadata.label_2_idx(batch["label"])
                return batch

            dataset["train"] = dataset["train"].map(function=map_fn, batched=True)
            dataset["train"] = dataset["train"].remove_columns(column_names=["label"])
            dataset["train"] = dataset["train"].rename_column(
                original_column_name="label_int",
                new_column_name="label",
            )
        dataset = dataset.cast_column(column="label", feature=ClassLabel(names=metadata.classes))

    if config.get("randsubset") is not None:
        n_subset = int(len(dataset["train"]) * config["randsubset"])
        dataset["train"] = dataset["train"].shuffle().select(range(n_subset))

    if config.get("eda") is not None:
        eda = EasyDataAugmentation(aug_factor=config["eda"])
        text_keys = metadata.text_keys
        dataset["train"] = eda.augment_dataset(dataset["train"], text_keys=text_keys).shuffle()

    if classifier in LLMMetadata.supported_nli:
        mapper = BERTSequencePairMap(tokenizer=tokenizer, dataset=config["abbrev"])
        dataset = mapper.preprocess(dataset=dataset)
    else:
        mapper = SiamesePairMap(dataset=config["abbrev"])
        dataset = mapper.preprocess(dataset=dataset)

        if config["abbrev"] == "mnlim":
            dataset["test"] = dataset.pop("validation_matched")
        elif config["abbrev"] in ["qqp", "qnli"]:
            dataset["test"] = dataset.pop("validation")

    return dataset


def prepare_callbacks(config: dict) -> list[Callback]:
    callbacks = []
    for cb_name, cb_config in config.items():
        if cb_name == "printer":
            callbacks.append(PrintCallback(**cb_config))
        elif cb_name == "csv_logger":
            callbacks.append(CsvLoggerPL(**cb_config))
        else:
            raise ValueError(f"Unsupported callback: {cb_name}")
    return callbacks


def train_classifier(
    config: dict,
    classifier,
    encoder,
    dataloader: dict[str, torch.utils.data.DataLoader] = {},
):
    if config["fit_with"] == "clf_trainer":
        trainer: ClassifierTrainer = config["Class"](
            classifier=classifier,
            encoder=encoder,
            **config["kwargs"],
        )
        config["L_trainer_kw"]["callbacks"] = prepare_callbacks(
            config=config["L_trainer_kw"]["callbacks"],
        )
        metrics = trainer.fit(
            **config["L_trainer_kw"],
            fit_kw={
                "train_dataloaders": dataloader["train"],
                "val_dataloaders": dataloader["test"],
                **config["fit_kw"],
            },
        )

    else:
        raise ValueError(f"Unsupported `fit_with`: {config['fit_with']}")

    if config.get("save_state_dict"):
        torch.save(obj=classifier.state_dict(), f=config["save_state_dict"])
    return classifier, metrics


def expt_clf2(config: dict, run: int | str = 0):
    # Template: https://huggingface.co/docs/transformers/v5.2.0/en/tasks/sequence_classification
    classifier_abbrev = config["metaconfig"]["args"]["classifier"]
    if classifier_abbrev in LLMMetadata.supported_nli:
        model = get_llm_model(config=config["models"]["classifier"])
        tokenizer = get_llm_tokenizer(config=config["models"]["tokenizer"])
        dataset: DatasetDict = get_dataset(config["dataset"])
        dataset = preprocess_data(
            dataset=dataset,
            tokenizer=tokenizer,
            classifier=classifier_abbrev,
            config=config["dataset"],
        )
        trainer = Trainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset,
            args=TrainingArguments(**config["trainer"]["args"]),
            data_collator=config["trainer"]["data_collator"]["Class"](
                tokenizer, **config["trainer"]["data_collator"]["kwargs"]
            ),
            compute_metrics=config["trainer"]["compute_metrics"],
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
    import csv
    import glob
    
    from datasets import ClassLabel, DatasetDict
    import evaluate
    import numpy as np
    from pytorch_lightning import Callback
    import torch
    import yaml

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from src.utils.callbacks import CsvLoggerHF
    from src.metadata import DatasetMetadata, EncoderMetadata, ClassifierMetadata, LLMMetadata
    from src.models.classifiers import ClassifierTrainer, SiameseLogistic
    from src.models.encoders import GloVeEncoder, Tfidf
    from src.models.modules import EasyDataAugmentation
    from src.utils.callbacks import CsvLoggerHF, CsvLoggerPL, PrintCallback
    from expts.expt_utils import (
        ConfigParser,
        TypeArgparse,
        get_classifier,
        get_dataset,
        get_dataloader,
        get_encoder,
        get_llm_model,
        get_llm_tokenizer,
        pprint,
        rename_runs,
        SequenceClassificationMetrics,
    )

    args = get_parser()
    for action in args._actions:
        if action.dest == "run":
            action.type = TypeArgparse.int_or_str
            break
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
