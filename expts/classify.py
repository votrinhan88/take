from collections.abc import Callable
from copy import deepcopy
import os
import sys

from datasets import ClassLabel, DatasetDict, concatenate_datasets
import pandas as pd
from sentence_transformers import SentenceTransformer, SparseEncoder
import torch
from torch import nn, Tensor
import yaml
from lightning import Callback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

from pipelines.expt_utils import get_dataset, get_encoder, get_classifier, get_dataloader
from src.models.classifiers import (
    ClassifierMetadata,
    ClassifierTrainer,
    LogisticRegression,
    NaiveBayes,
    SupportVectorMachine,
    TextCNN,
    TextRNN,
)
from src.models.encoders import EncoderMetadata, Tfidf, GloVeEncoder
from src.models.modules import EasyDataAugmentation
from src.utils.callbacks import CsvLoggerCallback, PrintCallback
from src.utils.metadata import DatasetMetadata
from src.utils.pythonic.dict_utils import traverse_dictlist


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
            self.classifier = self.args["classifier"]
            self.encoder = self.args["encoder"]

        assert self.dataset is not None, "Dataset must be specified"
        assert self.classifier is not None, "Classifier must be specified"
        assert self.encoder is not None, "Encoder must be specified"

        self.metadata_dataset = DatasetMetadata(dataset=self.dataset)
        self.metadata_encoder = EncoderMetadata(model=self.encoder)
        self.metadata_classifier = ClassifierMetadata(model=self.classifier)

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
        return self.config

    def get_set_metaconfig(self) -> dict:
        self.metaconfig = {
            "name": f"clf-{self.dataset}-{self.classifier}-{self.encoder}",
            "expt": "clf",
            "path": "./logs/classify",
            "args": self.args,
            "run": "eval:f'{run}'",
        }
        return self.metaconfig

    def get_config_dataset(self) -> dict:
        splits = {
            "agnews": ["train", "test"],
            "imdb": ["train", "test"],
            "mnlim": ["train", "validation_matched"],
            "qqp": ["train", "validation"],
            "sst2": ["train", "validation"],
        }
        config = {
            "abbrev": self.dataset,
            "splits": splits[self.dataset],
            "loader_kwargs": {
                "batch_size": 256,
                "shuffle": {"train": True, "test": False},
                "num_workers": 4,
            },
        }
        if self.dataset in ["sst2"]:
            config["unify_text"] = True
        return config

    def get_config_models(self) -> dict:
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
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder}")

        config["classifier"] = self.metadata_classifier.get_preset()
        if self.args["classifier"] in ["logistic", "svm"]:
            extra_kwargs = {
                "input_dim": config["encoder"]["embed_dim"],
                "num_classes": self.metadata_dataset.num_classes,
            }
            config["classifier"]["kwargs"].update(extra_kwargs)
        elif self.args["classifier"] in ["nbayes"]:
            extra_kwargs = {"input_dim": config["encoder"]["embed_dim"]}
            config["classifier"]["kwargs"].update(extra_kwargs)
        elif self.args["classifier"] in ["textcnn", "textrnn"]:
            extra_kwargs = {
                "embed_dim": config["encoder"]["embed_dim"],
                "num_classes": self.metadata_dataset.num_classes,
            }
            config["classifier"]["kwargs"].update(extra_kwargs)
        else:
            raise ValueError(f"Unsupported classifier: {self.args['classifier']}")

        return config

    def get_config_trainer(self) -> dict:
        path = self.metaconfig["path"]
        name = self.metaconfig["name"]
        if self.classifier == "nbayes":
            config = {
                "fit_with": "self",
                "fit_kw": {},
                "csv_log_path": f"eval:f'{path}/{name}/{name}-run={{run}}.csv'",
                "save_state_dict": f"eval:f'{path}/{name}/{name}-run={{run}}.pt'",
            }
        else:
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
                    "paired": False,
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
        return config

    def override_config(self, config: dict, **kwargs) -> dict:
        # fmt: off
        keys_avail = [
            "dataset_path", "randsubset", "eda", "batch_size",
            "embed_dim", "p_dropout",
            "num_epochs", "lr", "wdecay",
        ]
        for k, v in self.override_args.items():
            assert k in keys_avail, f"Unknown key: {k}"

            if k == "dataset_path":
                config["dataset"]["splits"].remove("train")
                config["dataset"]["splits_custom"] = {
                    "train": {"init_with": "from_csv", "from_csv_kwargs": {"path_or_paths": v}}
                }
                config["dataset"]["cast_label"] = True

            elif k == "batch_size":
                config["dataset"]["loader_kwargs"]["batch_size"] = v

            elif k == "randsubset":
                config["dataset"]["randsubset"] = v
            
            elif k == "eda":
                config["dataset"]["eda"] = v

            elif k == "embed_dim":
                if config["models"]["encoder"]["abbrev"] == "tfidf":
                    path_encoder = config["models"]["encoder"]["torch_load_kwargs"]["f"]
                    config["models"]["encoder"]["torch_load_kwargs"]["f"] = path_encoder.replace("3072d.pt", f"{v}d.pt")
                elif config["models"]["encoder"]["abbrev"] == "glove":
                    config["models"]["encoder"]["kwargs"]["embed_dim"] = v
                else:
                    raise ValueError(f"Unsupported `encoder`: {config['models']['encoder']['abbrev']}")
                config["models"]["encoder"]["embed_dim"] = v
                config["models"]["classifier"]["kwargs"]["input_dim"] = v
            
            elif k == "p_dropout":
                config["models"]["classifier"]["kwargs"]["p_dropout"] = v

            elif k == "num_epochs":
                config["trainer"]["L_trainer_kw"]["max_epochs"] = v
            
            elif k == "lr":
                config["trainer"]["kwargs"]["optimizer_kw"]["classifier"]["kwargs"]["lr"] = v
            
            elif k == "wdecay":
                config["trainer"]["kwargs"]["optimizer_kw"]["classifier"]["kwargs"]["weight_decay"] = v
                    
            else:
                raise ValueError(f"Unknown key: {k}")
        # fmt: on
        return config

    def export_config(self, path_config: str | None = None, verbose: bool = True) -> str:
        if path_config is None:
            path = self.metaconfig["path"]
            name = self.metaconfig["name"]
            args_str = ""
            for k, v in self.args.items():
                if k in ["base_config", "run", "n_runs", "dataset", "encoder", "classifier"]:
                    continue
                args_str += f"-{k}={v}"
            path_config = f"{path}/{name}/{name}-config{args_str}.yaml"

        os.makedirs(os.path.dirname(path_config), exist_ok=True)
        with open(path_config, "w") as f:
            yaml.dump(data=self.config, stream=f, sort_keys=False)

        if verbose:
            print(f"Exported config to {path_config}.")

        return path_config


def preprocess_data(dataset: DatasetDict, config: dict) -> DatasetDict:
    metadata = DatasetMetadata(dataset=config["abbrev"])

    if config["abbrev"] == "mnlim":
        dataset["test"] = dataset.pop("validation_matched")
    elif config["abbrev"] in ["qqp", "sst2"]:
        dataset["test"] = dataset.pop("validation")

    if config.get("unify_text") is not None:
        map_fn, map_kwargs = metadata.get_unify_map()
        dataset = dataset.map(map_fn, **map_kwargs)

    if config.get("cast_label") is not None:
        dataset = dataset.cast_column(column="label", feature=ClassLabel(names=metadata.classes))

    if config.get("randsubset") is not None:
        n_subset = int(len(dataset["train"]) * config["randsubset"])
        dataset["train"] = dataset["train"].shuffle().select(range(n_subset))

    if config.get("eda") is not None:
        eda = EasyDataAugmentation(aug_factor=config["eda"])
        dataset["train"] = eda.augment_dataset(dataset["train"]).shuffle()

    return dataset


def prepare_callbacks(config: dict) -> list[Callback]:
    callbacks = []
    for cb_name, cb_config in config.items():
        if cb_name == "printer":
            callbacks.append(PrintCallback(**cb_config))
        elif cb_name == "csv_logger":
            callbacks.append(CsvLoggerCallback(**cb_config))
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

    elif config["fit_with"] == "self":
        classifier.fit(train_loader=dataloader["train"], encoder=encoder)
        train_metrics = classifier.evaluate(val_loader=dataloader["train"], encoder=encoder)
        test_metrics = classifier.evaluate(val_loader=dataloader["test"], encoder=encoder)
        metrics = {
            "train_acc": train_metrics.get("accuracy"),
            "train_loss": train_metrics.get("loss"),
            "val_acc": test_metrics.get("accuracy"),
            "val_loss": test_metrics.get("loss"),
        }
        df = pd.DataFrame.from_dict(
            data={"epoch": [0], **{k: [v] for k, v in metrics.items()}},
            orient="columns",
        )
        os.makedirs(os.path.dirname(config["csv_log_path"]), exist_ok=True)
        df.to_csv(path_or_buf=config["csv_log_path"], index=False)
    else:
        raise ValueError(f"Unsupported `fit_with`: {config['fit_with']}")

    if config.get("save_state_dict"):
        torch.save(obj=classifier.state_dict(), f=config["save_state_dict"])
    return classifier, metrics


def expt_clf(config: dict, run: int | str = 0):
    dataset: DatasetDict = get_dataset(config["dataset"])
    dataset = preprocess_data(dataset=dataset, config=config["dataset"])
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

    # fmt: off
    args = argparse.ArgumentParser()
    args_group = args.add_argument_group("Metaconfig arguments")
    args_group.add_argument("--base_config", type=str)
    args_group.add_argument("--run", type=TypeArgparse.int_or_str, default=0)
    args_group.add_argument("--n_runs", type=int, default=1)
    args_group = args.add_argument_group("Dataset arguments")
    args_group.add_argument("--dataset", type=str, choices=["agnews", "imdb", "sst2"])
    args_group.add_argument("--dataset_path", type=str)
    args_group.add_argument("--randsubset", type=float)
    args_group.add_argument("--eda", type=int)
    args_group.add_argument("--batch_size", type=int)
    args_group = args.add_argument_group("Model arguments")
    args_group.add_argument("--encoder", type=str, choices=["tfidf", "glove"])
    args_group.add_argument("--classifier", type=str, choices=["logistic", "svm", "nbayes", "textcnn", "textrnn"])
    args_group.add_argument("--embed_dim", type=int)
    args_group.add_argument("--p_dropout", type=float)
    args_group = args.add_argument_group("Optimization arguments")
    args_group.add_argument("--num_epochs", type=int)
    args_group.add_argument("--lr", type=float)
    args_group.add_argument("--wdecay", type=float)
    args = args.parse_args()
    custom_args = {
        k: getattr(args, k)
        for k in [
            "base_config", "run", "n_runs",
            "dataset", "dataset_path", "randsubset", "eda", "batch_size",
            "classifier", "encoder", "embed_dim", "p_dropout",
            "num_epochs", "lr", "wdecay",
        ]
        if getattr(args, k) is not None
    }
    # fmt: on
    parser = ConfigParser(globals=globals(), locals=locals())
    config_factory = ConfigFactory(**custom_args)
    config = config_factory.get_config()
    config_factory.export_config(verbose=True)
    for run in rename_runs(run=args.run, n_runs=args.n_runs):
        config_run = parser.parse_eval_config(deepcopy(config), parse_flag="eval:")
        pprint(config_run)
        expt_clf(config=config_run, run=run)
