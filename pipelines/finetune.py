# TODO: Fix the mismatch in state_dict when saving via (1) StateDictCheckpointCallback vs via (2) model.save

import csv
from copy import deepcopy
import os
import sys
from typing import Callable
import warnings

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from lightning.pytorch.loggers import CSVLogger
from peft import LoraConfig, PeftModel
import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

from pipelines.expt_utils import get_dataset, get_llm_model, get_llm_tokenizer, get_encoder
from src.models.classifiers import ClassifierTrainer, LogisticRegression, TextCNN
from src.models.encoders import EncoderMetadata, Tfidf, GloVeEncoder, MiniLMWrapper
from src.finetune import (
    # callback
    FinetuneEvalCallback,
    SampleGenerationCallback,
    SampleInferenceCallback,
    StateDictCheckpointCallback,
    # ClosedEndedCollator,  # collator
    InstructionFinetuneMapFunction,  # map_fn
    TextTemplate,  # template
)
from src.models.llms.metadata import LLMMetadata
from src.prototypes.kmeans import KMeansClassifier
from src.utils.callbacks import PrintCallback
from src.utils.metadata import DatasetMetadata


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ConfigFactory:
    supported_overrides = ["batch_size", "num_epochs", "lr"]

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
            self.dataset = self.args.get("dataset")
            self.model = self.args.get("model")
            self.encoder = self.args.get("encoder")

        if self.base_config is None:
            assert self.dataset is not None, "Dataset must be specified if base_config is None"
            assert self.model is not None, "Model must be specified if base_config is None"
            assert self.encoder is not None, "Encoder must be specified if base_config is None"

        if self.args.get("dataset"):
            self.metadata_dataset = DatasetMetadata(dataset=self.args["dataset"])
        if self.args.get("model"):
            self.metadata_llm = LLMMetadata(model=self.args["model"])
        if self.args.get("encoder"):
            self.metadata_encoder = EncoderMetadata(model=self.args["encoder"])

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
            "name": f"ftn-{self.dataset}-{self.model}",
            "expt": "ftn",
            "path": "./logs/finetune",
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
            "qnli": ["train", "validation"],
        }
        config = {
            "abbrev": self.dataset,
            "splits": splits[self.dataset],
            "train": {"task": "generation", "max_length": 512},
        }
        if self.dataset in ["mnlim", "qqp", "sst2", "qnli"]:
            config["unify_text"] = True
        return config

    def get_config_models(self) -> dict:
        path = self.metaconfig["path"]
        name = self.metaconfig["name"]
        config = {
            "model": {
                "abbrev": self.model,
                **self.metadata_llm.get_preset_model(),
                "lora_config": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
            },
            "tokenizer": {
                "abbrev": self.model,
                **self.metadata_llm.get_preset_tokenizer(),
            },
            "encoder": self.metadata_encoder.get_preset(),
            "save_path": f"eval:f'{path}/{name}/run={{run}}/lora.pt'",
        }
        return config

    def get_config_trainer(self) -> dict:
        path = self.metaconfig["path"]
        name = self.metaconfig["name"]

        max_new_tokens = self.metadata_dataset.get_length_statistics()["quantiles"][100]
        max_new_tokens = min(max_new_tokens, 512)  # Cap at 512 for debug purpose.
        min_new_tokens = None
        if self.dataset in ["agnews", "imdb", "sst2"]:
            # Do not enforce min_new_tokens for paired datasets - second text column is elongated.
            min_new_tokens = self.metadata_dataset.get_length_statistics()["quantiles"][50]

        config = {
            "args": {
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 3e-4,
                "weight_decay": 1e-2,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "num_train_epochs": 10,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "save_steps": 1000000,
                "logging_steps": 1,
                "report_to": "none",
                "auto_find_batch_size": True,
            },
            "data_collator": {
                "Class": "eval:DataCollatorForLanguageModeling",
                "kwargs": {"mlm": False},
            },
            "callbacks": {
                "sample_gen": {
                    "output": f"eval:f'{path}/{name}/run={{run}}/sample_gen.txt'",
                    "dataset": self.dataset,
                    "genconfig": {
                        "max_new_tokens": max_new_tokens,
                        "min_new_tokens": min_new_tokens,
                    },
                    "every_n_steps": 1000,
                },
                "sample_inf": {
                    "output": f"eval:f'{path}/{name}/run={{run}}/sample_inf.txt'",
                    "dataset": self.dataset,
                    "genconfig_generation": {
                        "max_new_tokens": max_new_tokens,
                        "min_new_tokens": min_new_tokens,
                    },
                    "every_n_steps": 1000,
                },
                "ftn_eval": {
                    "output_csv": f"eval:f'{path}/{name}/run={{run}}/ftn_eval.csv'",
                    "output_log": f"eval:f'{path}/{name}/run={{run}}/ftn_eval.log'",
                    "dataset": self.dataset,
                    "genconfig": {
                        "max_new_tokens": max_new_tokens,
                        "min_new_tokens": min_new_tokens,
                    },
                    "every_n_steps": 1000,
                },
                "checkpoint": {
                    "save_dir": f"eval:f'{path}/{name}/run={{run}}/checkpoints'",
                    "every_n_steps": 5000,
                },
            },
        }
        return config

    def override_config(self, config: dict, **kwargs) -> dict:
        keys_avail = self.supported_overrides
        for k, v in self.override_args.items():
            assert k in keys_avail, f"Unknown key: {k}"

            if k == "batch_size":
                config["trainer"]["args"]["per_device_train_batch_size"] = v
                config["trainer"]["args"]["per_device_eval_batch_size"] = v
            elif k == "num_epochs":
                config["trainer"]["args"]["num_train_epochs"] = v
            elif k == "lr":
                config["trainer"]["args"]["learning_rate"] = v

        return config

    def export_config(self, path_config: str | None = None, verbose: bool = True) -> str:
        if path_config is None:
            path = self.metaconfig["path"]
            name = self.metaconfig["name"]
            args_str = ""
            for k, v in self.args.items():
                if k in ["base_config", "run", "n_runs", "dataset", "model", "encoder"]:
                    continue
                args_str += f"-{k}={v}"
            path_config = f"{path}/{name}/{name}-config{args_str}.yaml"

        os.makedirs(os.path.dirname(path_config), exist_ok=True)
        with open(path_config, "w") as f:
            yaml.dump(data=self.config, stream=f, sort_keys=False)

        if verbose:
            print(f"Config exported to {path_config}.")

        return path_config


def preprocess_data(dataset: DatasetDict, config: dict) -> DatasetDict:
    metadata = DatasetMetadata(dataset=config["abbrev"])

    if config["abbrev"] == "mnlim":
        dataset["test"] = dataset.pop("validation_matched")
    elif config["abbrev"] in ["qqp", "sst2", "qnli"]:
        dataset["test"] = dataset.pop("validation")

    if config.get("unify_text") is not None:
        map_fn, map_kwargs = metadata.get_unify_map()
        dataset = dataset.map(map_fn, **map_kwargs)

    return dataset


def map_dataset_by_task(
    dataset: Dataset,
    tokenizer,
    dataset_abbrev: str,
    model_abbrev: str,
    task: str | list[str],
    max_length: int | None = 512,
    batched: bool = True,
) -> Dataset:
    supported_tasks = ["generation", "inference"]

    if (task is None) or (task == []):
        warnings.warn(f"No task specified. Returning original dataset. Expect: {supported_tasks}")
        return dataset

    if isinstance(task, str):
        task = [task]
    datasets = []
    for t in task:
        assert t in supported_tasks, f"Unsupported task: {t}. Supported tasks: {supported_tasks}"

        map_fn = InstructionFinetuneMapFunction(
            tokenizer=tokenizer,
            dataset=dataset_abbrev,
            model=model_abbrev,
            task=t,
            max_length=max_length,
            batched=batched,
        )
        datasets.append(dataset.map(map_fn, batched=batched))

    if len(datasets) == 1:
        return datasets[0]
    else:
        return concatenate_datasets(datasets)


def get_finetune_callbacks(tokenizer, encoder, config: dict) -> list[TrainerCallback]:
    supported_callbacks = ["sample_gen", "sample_inf", "ftn_eval", "checkpoint"]
    callbacks = []
    for cb in config.keys():
        if cb == "sample_gen":
            callback = SampleGenerationCallback(tokenizer=tokenizer, **config[cb])
            callbacks.append(callback)

        elif cb == "sample_inf":
            callback = SampleInferenceCallback(tokenizer=tokenizer, **config[cb])
            callbacks.append(callback)

        elif cb == "ftn_eval":
            callback = FinetuneEvalCallback(tokenizer=tokenizer, encoder=encoder, **config[cb])
            callbacks.append(callback)

        elif cb in ["checkpoint"]:
            callback = StateDictCheckpointCallback(**config[cb])
            callbacks.append(callback)

        else:
            msg = f"Unknown callback: {cb}. Supported callbacks: {supported_callbacks}"
            raise NotImplementedError(msg)

    return callbacks


def expt_finetune(config: dict, run: int = 0):
    model = get_llm_model(config=config["models"]["model"], verbose=True)
    tokenizer = get_llm_tokenizer(config=config["models"]["tokenizer"])
    dataset: DatasetDict = get_dataset(config=config["dataset"])
    dataset = preprocess_data(dataset=dataset, config=config["dataset"])
    encoder: nn.Module = get_encoder(config=config["models"]["encoder"])

    def preembed(batch: dict) -> dict:
        batch["embeddings"] = encoder(batch["text"])
        return batch

    dataset = dataset.map(function=preembed, batched=True)

    if config["dataset"]["train"].get("task") is not None:
        dataset["train"] = map_dataset_by_task(
            dataset=dataset["train"],
            tokenizer=tokenizer,
            dataset_abbrev=config["dataset"]["abbrev"],
            model_abbrev=config["models"]["model"]["abbrev"],
            task=config["dataset"]["train"]["task"],
            max_length=config["dataset"]["train"].get("max_length", 512),
            batched=True,
        )

        examples_indices = torch.randperm(len(dataset["train"]))[0:10].tolist()
        print("\n[Examples from dataset]")
        for i in examples_indices:
            print(f"\n[Example {i}]")
            print(tokenizer.decode(dataset["train"][i]["input_ids"], skip_special_tokens=True))

    callbacks = get_finetune_callbacks(
        tokenizer=tokenizer,
        encoder=encoder,
        config=config["trainer"]["callbacks"],
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        args=TrainingArguments(**config["trainer"]["args"]),
        data_collator=config["trainer"]["data_collator"]["Class"](
            tokenizer=tokenizer, **config["trainer"]["data_collator"]["kwargs"]
        ),
        callbacks=callbacks,
    )
    trainer.train()
    torch.save(obj=model.state_dict(), f=config["models"]["save_path"])


if __name__ == "__main__":
    import argparse

    from expts.expt_utils import ConfigParser, TypeArgparse, pprint, rename_runs

    # fmt: off
    args = argparse.ArgumentParser()
    args_group = args.add_argument_group("Metaconfig arguments")
    args_group.add_argument("--base_config", type=str)
    args_group.add_argument("--run", type=TypeArgparse.int_or_str, default=0)
    args_group.add_argument("--n_runs", type=int, default=1)
    args_group = args.add_argument_group("Dataset arguments")
    args_group.add_argument("--dataset", type=str, choices=DatasetMetadata.supported)
    args_group = args.add_argument_group("Model arguments")
    args_group.add_argument("--model", type=str, choices=LLMMetadata.supported)
    args_group.add_argument("--encoder", type=str, choices=EncoderMetadata.supported, default="minilm")
    args_group = args.add_argument_group("Optimization arguments")
    args_group.add_argument("--batch_size", type=int)
    args_group.add_argument("--num_epochs", type=int)
    args_group.add_argument("--lr", type=float)
    args = args.parse_args()
    custom_args = {
        k: getattr(args, k)
        for k in [
            "base_config", "run", "n_runs",
            "dataset",
            "model", "encoder",
            "batch_size", "num_epochs", "lr",
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
        expt_finetune(config=config_run, run=run)
