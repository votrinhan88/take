from copy import deepcopy
import csv
import os
import sys

from datasets import Dataset
import tqdm.auto as tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

from pipelines.expt_utils import get_llm_model, get_llm_tokenizer
from src.finetune.templates import TextTemplate
from src.generate.utils import generate
from src.models.llms.metadata import LLMMetadata
from src.utils.metadata import DatasetMetadata
from src.utils.pythonic.numeric_utils import balanced_partition


class ConfigFactory:
    supported_overrides = [
        "load_state_dict",
        "conditional",
        "num_samples",
        "export_csv",
        "export_hfds",
    ]
    config: dict

    def __init__(self, **kwargs):
        self.args = kwargs
        self.override_args = {k: v for k, v in kwargs.items() if k in self.supported_overrides}

        # Load base config from YAML if provided
        if self.args.get("base_config") is not None:
            parser = ConfigParser(globals=globals(), locals=locals())
            path = parser.parse_path_config(config=self.args["base_config"], ext="yaml")
            self.base_config = yaml.safe_load(stream=open(file=path, mode="r"))
            self.base_config["metaconfig"]["args"] = self.args
        else:
            self.base_config = None
            self.dataset = self.args["dataset"]
            self.model = self.args["model"]

        # Metadata
        self.metadata_dataset = DatasetMetadata(dataset=self.args["dataset"])
        self.metadata_llm = LLMMetadata(model=self.args["model"])

    def get_config(self) -> dict:
        if self.base_config is not None:
            config = deepcopy(self.base_config)
        else:
            config = {
                "metaconfig": self.get_set_metaconfig(),
                "dataset": self.get_config_dataset(),
                "models": self.get_config_models(),
                "generate": self.get_config_generate(),
            }
        self.config = self.override_config(config, **self.args)
        return deepcopy(self.config)

    def get_set_metaconfig(self) -> dict:
        self.metaconfig = {
            "name": f"gen-{self.dataset}-{self.model}",
            "expt": "gen",
            "path": "./logs/generate",
            "args": self.args,
            "run": "eval:f'{run}'",
        }
        return self.metaconfig

    def get_config_dataset(self) -> dict:
        config = {"abbrev": self.dataset}
        return config

    def get_config_models(self) -> dict:
        config = {
            "model": {
                "abbrev": self.model,
                **self.metadata_llm.get_preset_model(),
                "load_state_dict": {
                    "f": f"./logs/finetune/{self.dataset}-{self.model}/lora.pt",
                    "weights_only": True,
                },
            },
            "tokenizer": self.metadata_llm.get_preset_tokenizer(),
        }
        return config

    def get_config_generate(self) -> dict:
        path = self.metaconfig["path"]
        name = self.metaconfig["name"]

        max_new_tokens = self.metadata_dataset.get_length_statistics()["quantiles"][100]
        min_new_tokens = None
        if self.dataset in ["agnews", "imdb", "sst2"]:
            min_new_tokens = self.metadata_dataset.get_length_statistics()["quantiles"][50]

        config = {
            "conditional": True,
            "num_samples": 5000,
            "genconfig": {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "do_sample": True,
                "num_beams": 1,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "num_return_sequences": 4,
            },
            "export_csv": f"eval:f'{path}/{name}/{name}-run={{run}}.csv'",
            "export_hfds": False,
        }
        return config

    def override_config(self, config: dict, **kwargs) -> dict:
        name = (
            config["metaconfig"]["name"]
            if "metaconfig" in config
            else f"gen-{self.args['dataset']}-{self.args['model']}"
        )
        for k, v in self.override_args.items():
            if k == "load_state_dict":
                config["models"]["model"]["load_state_dict"]["f"] = v
            elif k == "conditional":
                config["generate"]["conditional"] = v
            elif k == "num_samples":
                config["generate"]["num_samples"] = v
            elif k == "export_csv":
                config["generate"]["export_csv"] = v
            elif k == "export_hfds":
                config["generate"]["export_hfds"] = v
            else:
                raise ValueError(f"Unknown override: {k}. Supported: {self.supported_overrides}")
        return config

    def export_config(self, path_config: str | None = None, verbose: bool = True) -> str:
        if path_config is None:
            path = self.metaconfig["path"]
            name = self.metaconfig["name"]
            args_str = ""
            for k, v in self.args.items():
                if k in ["base_config", "run", "n_runs", "dataset", "model"]:
                    continue
                args_str += f"-{k}={v}"
            path_config = f"{path}/{name}/{name}-config{args_str}.yaml"

        os.makedirs(os.path.dirname(path_config), exist_ok=True)
        with open(path_config, "w") as f:
            yaml.dump(data=self.config, stream=f, sort_keys=False)

        if verbose:
            print(f"Config exported to {path_config}.")

        return path_config



class DatasetExporter:
    def __init__(self, dataset: str, export_csv: str | None, export_hfds: str | None):
        self.dataset = dataset
        self.export_csv = export_csv
        self.export_hfds = export_hfds

        self.metadata = DatasetMetadata(dataset=self.dataset)

        if self.dataset in self.metadata.requires_unify_map:
            self.inv_unify_map, kwargs = self.metadata.get_unify_map(inverse=True)
            self.remove_columns = kwargs["remove_columns"]
        else:
            self.inv_unify_map = lambda x: x
            self.remove_columns = []

    def append(self, parsed: dict):
        unified = self.inv_unify_map(parsed)
        for k in self.remove_columns:
            unified.pop(k, None)
        # Make sure "label" is at first location
        unified = {"label": unified["label"], **{k: v for k, v in unified.items() if k != "label"}}

        if self.export_csv not in [None, False]:
            self.append_csv(unified)
        if self.export_hfds not in [None, False]:
            self.append_hfds(unified)

    def append_csv(self, batch: dict):
        # Create directory if not exists
        if not isinstance(self.export_csv, str):
            return

        all_keys = list(batch.keys())
        all_columns = list(batch.values())
        num_rows = len(all_columns[0])
        os.makedirs(os.path.dirname(self.export_csv), exist_ok=True)
        write_header = not os.path.isfile(self.export_csv)

        with open(self.export_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=all_keys,
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )

            if write_header:
                writer.writeheader()

            for row in range(num_rows):
                writer.writerow({all_keys[c]: all_columns[c][row] for c in range(len(all_keys))})

    def append_hfds(self, batch: dict):
        raise NotImplementedError()


def export_dataset(dataset: dict, config: dict, verbose: bool = False):
    if isinstance(config["generate"].get("export_csv"), str):
        path_csv = config["generate"]["export_csv"]
        keys = ["label", "text"]
        # Create directory if not exists
        os.makedirs(os.path.dirname(path_csv), exist_ok=True)
        with open(path_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(len(dataset["text"])):
                row = [dataset[k][i] for k in keys]
                writer.writerow(row)
        if verbose:
            print(f"Saved generated samples to {path_csv}")

    if isinstance(config["generate"].get("export_hfds"), str):
        path_hfds = config["generate"]["export_hfds"]
        dataset_hfds = Dataset.from_dict(dataset)
        dataset_hfds.save_to_disk(path_hfds)
        if verbose:
            print(f"Saved generated samples to {path_hfds}")


def expt_generate(config: dict, run: int | str = 0):
    model = get_llm_model(config=config["models"]["model"], verbose=True)
    tokenizer = get_llm_tokenizer(config=config["models"]["tokenizer"])
    template = TextTemplate(dataset=config["dataset"]["abbrev"], default_task="generation")
    genconfig = GenerationConfig(**config["generate"]["genconfig"])
    classes = template.classes

    dataset_exporter = DatasetExporter(
        dataset=config["dataset"]["abbrev"],
        export_csv=config["generate"].get("export_csv"),
        export_hfds=config["generate"].get("export_hfds"),
    )

    def generate_from_prompt(prompt: str, quota: int, pbar_desc: str):
        pbar = tqdm.tqdm(total=quota, desc=pbar_desc)
        num_parts = quota // genconfig.num_return_sequences
        num_samples_step = balanced_partition(total=quota, num_parts=num_parts)

        # This for-loop is needed for periodic saving
        for n in num_samples_step:
            generated = generate(
                model=model,
                tokenizer=tokenizer,
                num_samples=n,
                prompt=prompt,
                genconfig=genconfig,
                validation_fn=template.validate,
            )
            parsed = template.parse(strings=generated)
            dataset_exporter.append(parsed=parsed)
            pbar.update(len(generated))

    if config["generate"]["conditional"] is True:
        quota = balanced_partition(total=config["generate"]["num_samples"], num_parts=len(classes))
        for i, k in enumerate(classes):
            prompt = template.template_generation(label=k)
            generate_from_prompt(prompt=prompt, quota=quota[i], pbar_desc=f"generate() - {k}")
    elif config["generate"]["conditional"] is False:
        prompt = template.template_generation(label=None)
        quota = config["generate"]["num_samples"]
        generate_from_prompt(prompt=prompt, quota=quota, pbar_desc="generate() - Uncond")


if __name__ == "__main__":
    import argparse
    from expts.expt_utils import ConfigParser, TypeArgparse, pprint, rename_runs

    # fmt: off
    parser = argparse.ArgumentParser()
    args_group = parser.add_argument_group("Metaconfig arguments")
    args_group.add_argument("--base_config", type=str)
    args_group.add_argument("--run", type=TypeArgparse.int_or_str, default=0)
    args_group.add_argument("--n_runs", type=int, default=1)
    args_group = parser.add_argument_group("Dataset arguments")
    args_group.add_argument("--dataset", type=str, choices=DatasetMetadata.supported)
    args_group = parser.add_argument_group("Model arguments")
    args_group.add_argument("--model", type=str, choices=LLMMetadata.supported)
    args_group.add_argument("--load_state_dict", type=str)
    args_group = parser.add_argument_group("Generation arguments")
    args_group.add_argument("--conditional", type=bool)
    args_group.add_argument("--num_samples", type=int)
    args_group.add_argument("--export_csv", type=str)
    args_group.add_argument("--export_hfds", type=str)
    args = parser.parse_args()
    custom_args = {
        k: getattr(args, k)
        for k in [
            "base_config", "run", "n_runs",
            "dataset",
            "model", "load_state_dict",
            "conditional", "num_samples", "export_csv", "export_hfds",
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
        expt_generate(config=config_run, run=run)
