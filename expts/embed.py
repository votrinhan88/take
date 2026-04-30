from copy import deepcopy
import os
import sys
from typing import Callable

from datasets import concatenate_datasets, load_dataset, Dataset, DatasetDict
import torch
from torch import nn
from torch.utils.data import DataLoader

repo_path = os.path.abspath(os.path.join(__file__, "../.."))
assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from expts.expt_utils import get_dataset, get_dataloader, get_encoder
from src.models.encoders import (
    EncoderMetadata,
    Tfidf,
    GloVeEncoder,
    E5Wrapper,
    MiniLMWrapper,
    JinaWrapper,
)
from src.utils.metadata import DatasetMetadata


class ConfigFactory:
    supported_overrides = [
        "embed_dim",
        "batch_size",
        "splits_corpus",
        "num_workers",
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
            self.encoder = self.args["encoder"]

        self.encoder = self.args["encoder"]
        self.dataset = self.args["dataset"]

        self.metadata_dataset = DatasetMetadata(dataset=self.dataset)
        self.metadata_encoder = EncoderMetadata(model=self.encoder)

    def get_config(self) -> dict:
        config = {
            "metaconfig": self.get_set_metaconfig(),
            "dataset": self.get_config_dataset(),
            "models": self.get_config_models(),
            "embed": self.get_config_embed(),
        }
        self.config = self.override_config(config, **self.args)
        return self.config

    def get_set_metaconfig(self) -> dict:
        self.metaconfig = {
            "name": f"emb-{self.encoder}-{self.dataset}",
            "expt": "emb",
            "path": "./results/raw/embed",
            "args": self.args,
            "run": "eval:f'{run}'",
        }
        return self.metaconfig

    def get_config_dataset(self) -> dict:
        config = {
            "abbrev": self.dataset,
            "splits": list(self.metadata_dataset.get_preset()["original_splits"].keys()),
            "loader_kwargs": {"batch_size": 1000, "shuffle": False, "num_workers": 4},
            "splits_corpus": self.metadata_dataset.get_preset()["splits_corpus"],
        }
        if self.dataset in self.metadata_dataset.requires_unify_map:
            config["unify_text"] = True
        return config

    def get_config_models(self) -> dict:
        name = self.metaconfig["name"]
        config = {"encoder": self.metadata_encoder.get_preset()}
        if self.encoder == "tfidf":
            config["encoder"]["kwargs"]["sparse"] = False
            config["encoder"].update(
                {
                    "fit_with": "self",
                    "save_path": f"eval:f'./results/raw/embed/{name}/{name}-run={{run}}.pt'",
                }
            )
        return config

    def get_config_embed(self) -> dict:
        name = self.metaconfig["name"]
        config = {
            "path_output": f"eval:f'./results/raw/embed/{name}/{name}-run={{run}}/'",
        }
        return config

    def override_config(self, config: dict, **kwargs) -> dict:
        name = self.metaconfig["name"]
        for k, v in self.override_args.items():
            if k == "embed_dim":
                config["models"]["encoder"]["kwargs"]["embed_dim"] = v
                config["models"]["encoder"]["embed_dim"] = v
            elif k == "batch_size":
                config["dataset"]["loader_kwargs"]["batch_size"] = v
            elif k == "path_output":
                if self.encoder == "tfidf":
                    config["models"]["encoder"]["save_path"] = (
                        f"eval:f'{v}/{name}/{name}-run={{run}}.pt'"
                    )
                config["embed"]["path_output"] = f"eval:f'{v}/{name}/{name}-run={{run}}/'"
            else:
                raise ValueError(f"Unknown key: {k}")
        return config

    def export_config(self, path_config: str | None = None, verbose: bool = True) -> str:
        if path_config is None:
            path = self.metaconfig["path"]
            name = self.metaconfig["name"]
            args_str = ""
            for k, v in self.args.items():
                if k in ["base_config", "run", "n_runs", "dataset", "encoder"]:
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

    if config.get("unify_text") is not None:
        map_fn, map_kwargs = metadata.get_unify_map()
        dataset = dataset.map(map_fn, **map_kwargs)

    return dataset


def prepare_corpus_loader(
    dataset: DatasetDict, config: dict, verbose: bool = True
) -> DataLoader | None:
    if "splits_corpus" not in config:
        if verbose:
            print("No splits_corpus key in config. Returning None.")
        return None

    splits_corpus = config["splits_corpus"]
    assert splits_corpus is None or isinstance(splits_corpus, (str, list)), (
        "splits_corpus must be None, a list or a string."
    )

    if isinstance(splits_corpus, str):
        corpus = dataset[splits_corpus]
    elif isinstance(splits_corpus, list):
        corpus = concatenate_datasets([dataset[split] for split in splits_corpus])
    else:
        raise ValueError(f"splits_corpus must be str or list. Got {splits_corpus}.")
    if verbose:
        print(f"Corpus from splits {splits_corpus} of size {len(corpus)}.")

    corpus_loader: DataLoader = get_dataloader(dataset=corpus, **config["loader_kwargs"])
    return corpus_loader


def expt_emb(config: dict, run: int | str):
    dataset: DatasetDict = get_dataset(config=config["dataset"])
    dataset = preprocess_data(dataset=dataset, config=config["dataset"])
    corpus_loader = prepare_corpus_loader(dataset=dataset, config=config["dataset"])
    encoder: nn.Module = get_encoder(
        config=config["models"]["encoder"],
        dataloader=corpus_loader,
    )
    if hasattr(encoder, "embeddings"):
        print(f"Encoder {encoder} with embeddings {encoder.embeddings.shape}.")
    if config["models"]["encoder"].get("save_path"):
        os.makedirs(os.path.dirname(config["models"]["encoder"]["save_path"]), exist_ok=True)
        torch.save(obj=encoder.state_dict(), f=config["models"]["encoder"]["save_path"])

    def preembed(batch: dict) -> dict:
        batch["embeddings"] = encoder(batch["text"])
        return batch

    dataset = dataset.map(function=preembed, batched=True, batch_size=100)
    os.makedirs(os.path.dirname(config["embed"]["path_output"]), exist_ok=True)
    dataset.save_to_disk(dataset_dict_path=config["embed"]["path_output"])


if __name__ == "__main__":
    import argparse
    import yaml

    from expts.expt_utils import ConfigParser, TypeArgparse, pprint, rename_runs

    # fmt: off
    args = argparse.ArgumentParser()
    args_group = args.add_argument_group("Metaconfig arguments")
    args_group.add_argument("--base_config", type=str)
    args_group.add_argument("--run", type=TypeArgparse.int_or_str, default=0)
    args_group.add_argument("--n_runs", type=int, default=1)
    args_group = args.add_argument_group("Dataset arguments")
    args_group.add_argument("--dataset", type=str, choices=DatasetMetadata.supported)
    args_group.add_argument("--batch_size", type=int)
    args_group = args.add_argument_group("Model arguments")
    args_group.add_argument("--encoder", type=str, choices=EncoderMetadata.supported)
    args_group.add_argument("--embed_dim", type=int)
    args_group = args.add_argument_group("Embed arguments")
    args_group.add_argument("--path_output", type=str)
    args = args.parse_args()
    custom_args = {
        k: getattr(args, k)
        for k in [
            "base_config", "run", "n_runs",
            "dataset", "batch_size",
            "encoder", "embed_dim",
            "path_output",
        ]
        if getattr(args, k) is not None}
    # fmt: on

    parser = ConfigParser(globals=globals(), locals=locals())
    config_factory = ConfigFactory(**custom_args)
    config = config_factory.get_config()
    config_factory.export_config(verbose=True)
    for run in rename_runs(run=args.run, n_runs=args.n_runs):
        config_run = parser.parse_eval_config(deepcopy(config), parse_flag="eval:")
        pprint(config_run)
        expt_emb(config=config_run, run=run)
