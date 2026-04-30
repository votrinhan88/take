import os
from copy import deepcopy
import sys
from typing import overload

from datasets import load_dataset, ClassLabel, Dataset, DatasetDict
from peft import get_peft_model, LoraConfig
import pprint as Pprint
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedModel, PreTrainedTokenizerBase

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

from src.models.encoders.metadata import EncoderMetadata
from src.models.llms.metadata import LLMMetadata
from src.utils.metadata import DatasetMetadata
from src.utils.pythonic.dict_utils import spread_dict


class ConfigParser:
    def __init__(self, globals: dict, locals: dict):
        self.globals = globals
        self.locals = locals

    @staticmethod
    def parse_path_config(config: str, ext: str = "yaml") -> str:
        config_splits = config.split("-")
        return f"./configs/{config_splits[0]}/{config}.{ext}"

    def parse_eval_config(self, config, parse_flag="eval:"):
        if isinstance(config, dict):
            return {
                k: self.parse_eval_config(config=v, parse_flag=parse_flag)
                for k, v in config.items()
            }
        elif isinstance(config, list):
            return [self.parse_eval_config(config=vi, parse_flag=parse_flag) for vi in config]
        elif isinstance(config, str) and config.startswith(parse_flag):
            return eval(config[len(parse_flag) :], self.globals, self.locals)
        else:
            return config


def rename_runs(run: int | str, n_runs: int, prefix: str = "v") -> list[str]:
    """Gives a range of runs organized names.

    Args:
      `run`: Name of the first run.
      `n_runs`: Number of runs.
      `prefix`: The syntax to prepend to the current run number. Only works when\
        `n_runs` > 0. Defaults to `'v'`.
    
    Returns:
      A list of run names.
    
    Examples:
    ```
    >>> rename_runs(run=0, n_runs=5)
    ['0', '1', '2', '3', '4']
    >>> rename_runs(run=10, n_runs=5)
    ['10', '11', '12', '13', '14']
    >>> rename_runs(run='10', n_runs=5)
    ['10-v0', '10-v1', '10-v2', '10-v3', '10-v4']
    >>> rename_runs(run='beta', n_runs=5)
    ['beta-v0', 'beta-v1', 'beta-v2', 'beta-v3', 'beta-v4']
    >>> rename_runs(run='adam-lr=0.3', n_runs=5)
    ['adam-lr=0.3-v0', 'adam-lr=0.3-v1', 'adam-lr=0.3-v2', 'adam-lr=0.3-v3', 'adam-lr=0.3-v4']
    ```
    """
    if n_runs == 1:
        return [str(run)]

    runs = []
    for i in range(n_runs):
        if isinstance(run, int):
            run_i = str(run + i)
        else:
            run_i = f"{run}-{prefix}{i}"
        runs.append(run_i)
    return runs


class TypeArgparse:
    @staticmethod
    def int_or_str(value) -> int | str:
        """Convert a value to int if possible, otherwise return it as string."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def bool_strict(value: bool | str) -> bool:
        """Convert a value to bool if possible, otherwise return it as string."""
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            if value.lower() in ("true", "yes", "1"):
                return True
            elif value.lower() in ("false", "no", "0"):
                return False

        raise ValueError(f"Cannot convert '{value}' ({type(value)}) to bool.")


def pprint(obj):
    pp = Pprint.PrettyPrinter(indent=1, compact=True, sort_dicts=False)
    return pp.pprint(obj)


def get_dataset(config: dict) -> DatasetDict:
    dataset_metadata = DatasetMetadata(dataset=config["abbrev"])
    dataset = DatasetDict()
    # Load default splits
    if config.get("splits") is not None:
        dataset_default = load_dataset(
            **dataset_metadata.get_preset()["load_dataset_kwargs"],  # {path, ?name}
            split=config["splits"],
            cache_dir="./datasets",
        )
        dataset.update({k: ds for k, ds in zip(config["splits"], dataset_default)})
    # Load custom splits
    for split, s_config in config.get("splits_custom", {}).items():
        if s_config["init_with"] == "load_dataset":
            s_dataset = load_dataset(**s_config["load_dataset_kwargs"])
        elif s_config["init_with"] == "from_csv":
            s_dataset = Dataset.from_csv(**s_config["from_csv_kwargs"])
        else:
            raise ValueError(f"Unknown init_with option: {s_config['init_with']}")

        dataset[split] = s_dataset

    dataset = dataset.with_format("torch")
    return dataset


# fmt: off
@overload
def get_dataloader(dataset: Dataset, **kwargs) -> DataLoader: ...
@overload
def get_dataloader(dataset: DatasetDict | dict, **kwargs) -> dict[str, DataLoader]: ...
# fmt: on
def get_dataloader(
    dataset: Dataset | DatasetDict | dict, **kwargs
) -> DataLoader | dict[str, DataLoader]:
    """Get dataloader for the given dataset in the pipeline.

    Args:
    + `dataset`: Dataset or dict of Datasets.
    + `kwargs`: Additional args for DataLoader. Each arg can be specified globally or separately \
        for each split. For example, `batch_size=32` or `batch_size={"train": 32, "test": 64}`.

    Returns: A dataloader or dict of `(split: dataloader)`.
    """
    if isinstance(dataset, Dataset):
        dataset: dict = {"_temp_split": dataset}

    split_kwargs = spread_dict(kwargs, groups=list(dataset.keys()))
    dataloader = {}
    for split in dataset.keys():
        dataloader[split] = DataLoader(dataset=dataset[split], **split_kwargs[split])

    if "_temp_split" in dataloader.keys():
        return dataloader["_temp_split"]
    else:
        return dataloader


def get_encoder(config: dict, dataloader: DataLoader | None = None) -> nn.Module:
    if "abbrev" in config.keys():
        base_config = EncoderMetadata(model=config["abbrev"]).get_preset()
        base_config.update(config)
        config = deepcopy(base_config)

    # Init encoder
    if config.get("init_with") == "torch_load":
        encoder = torch.load(**config["torch_load_kwargs"])
    else:
        encoder = config["Class"](**config["kwargs"])

    # Fit encoder
    if config.get("fit_with") is not None:
        if config["fit_with"] == "self":
            if dataloader is None:
                raise ValueError("Dataloader required to fit encoder with 'self'")
            encoder.fit(train_loader=dataloader, **config.get("fit_kwargs", {}))
        else:
            raise NotImplementedError(f"Unknown fit_with option: {config['fit_with']}")

    # Wrap encoder
    if config.get("wrap") is not None:
        encoder = config["wrap"]["Class"](encoder, **config["wrap"]["kwargs"])

    return encoder


def get_classifier(config: dict, strict: bool = True) -> nn.Module:
    classifier = config["Class"](**config["kwargs"])
    if config.get("load_state_dict"):
        device = next(classifier.parameters()).device
        classifier.load_state_dict(torch.load(**config["load_state_dict"], map_location=device))
    return classifier


def get_llm_model(config: dict, verbose: bool = True) -> PreTrainedModel:
    model = config["Class"](**config["kwargs"])
    if config.get("special_tokens") is not None:
        if config["special_tokens"].get("bos") is not None:
            model.generation_config.bos_token_id = config["special_tokens_ids"]["bos"]
        if config["special_tokens"].get("eos") is not None:
            model.generation_config.eos_token_id = config["special_tokens_ids"]["eos"]
        if config["special_tokens"].get("pad") is not None:
            model.generation_config.pad_token_id = config["special_tokens_ids"]["pad"]
    if config.get("lora_config") is not None:
        model.config.use_cache = False
        for param in model.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(**config["lora_config"])
        model = get_peft_model(model=model, peft_config=lora_config)
    if config.get("load_state_dict") is not None:
        model.load_state_dict(torch.load(**config["load_state_dict"], map_location=model.device))

    if verbose:
        n_params = 0
        n_params_trainable = 0
        for _, p in model.named_parameters():
            n_params += p.numel()
            if p.requires_grad:
                n_params_trainable += p.numel()
        ratio = n_params_trainable / n_params
        print(f"Model: {model}")
        print(f"trainable params: {n_params_trainable} / {n_params} ({100 * ratio:.4f} %)")

    return model


def get_llm_tokenizer(config: dict) -> PreTrainedTokenizerBase:
    tokenizer = config["Class"](**config["kwargs"])
    if config.get("set_eos_as_pad") == True:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


if __name__ == "__main__":

    def test_rename_runs():
        assert rename_runs(run=0, n_runs=5) == ["0", "1", "2", "3", "4"]
        assert rename_runs(run=10, n_runs=5) == ["10", "11", "12", "13", "14"]
        assert rename_runs(run="10", n_runs=5) == [f"10-v{i}" for i in range(5)]
        assert rename_runs(run="beta", n_runs=5) == [f"beta-v{i}" for i in range(5)]
        assert rename_runs(run="adam-lr=0.3", n_runs=5) == [f"adam-lr=0.3-v{i}" for i in range(5)]
        print("test_rename_runs: All tests passed.")

    test_rename_runs()
