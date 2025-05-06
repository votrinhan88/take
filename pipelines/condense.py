from collections import Counter, defaultdict
from copy import deepcopy
import os
import sys
from typing import Callable

import datasets
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets
import mauve
import numpy as np
from peft import LoraConfig, PeftModel
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import tqdm.auto as tqdm

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

from pipelines.classify import train_classifier
from pipelines.expt_utils import (
    get_dataset,
    get_dataloader,
    get_classifier,
    get_encoder,
    get_llm_model,
    get_llm_tokenizer,
)
from src.models.classifiers import (
    ClassifierMetadata,
    ClassifierTrainer,
    LogisticRegression,
    SupportVectorMachine,
    TextCNN,
    TextRNN,
)
from src.metrics.infogain import (
    DeterminantalPointProcess,
    AverageSimilarityGain,
    NearestNeighborDissimilarity,
)
from src.metrics.similarity import (
    CosineSimilarity,
    ExponentialCosineSimilarity,
    NormalizedCosineSimilarity,
    GeneralizedJaccardSimilarity,
    InnerProductSimilarity,
    JaccardSimilarity,
    RBFKernelSimilarity,
)
from src.models.encoders import EncoderMetadata, E5Wrapper, MiniLMWrapper, JinaWrapper
from src.influence import (
    BatchUnpacker,
    LiSSAInfluenceScorer,
    JointMarginalInfluenceScorer,
    ResidualInfluenceScorer,
)
from src.finetune.collators import ClosedEndedCollator
from src.finetune.map_function import InstructionFinetuneMapFunction
from src.models.llms import LLMMetadata
from src.metrics import DistanceToClosestRecord, DistinctN, Perplexity, SelfBLEU
from src.prototypes.discreteot import (
    DiscreteOTDistiller,
    TrajectoryAwareKnowledgeEstimator,
    TemperatureScheduler,
)
from src.prototypes.kmeans import KMeansClassifier
from src.utils.callbacks import PrintCallback
from src.utils.metadata import DatasetMetadata
from src.utils.pythonic.numeric_utils import balanced_partition, ensure_tensor


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ConfigFactory:
    supported_overrides = ["batch_size"]  # TODO

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
            self.encoder = self.args["encoder"]
            self.influencer = self.args["influencer"]
            self.llm = self.args["llm"]
            self.condense = self.args["condense"]

        # Metadata
        self.metadata_dataset = DatasetMetadata(dataset=self.dataset)
        self.metadata_encoder = EncoderMetadata(model=self.encoder)
        self.metadata_influencer = ClassifierMetadata(model=self.influencer)
        self.metadata_llm = LLMMetadata(model=self.llm)

    def get_config(self) -> dict:
        if self.base_config is not None:
            config = deepcopy(self.base_config)
        else:
            config = {
                "metaconfig": self.get_set_metaconfig(),
                "dataset": self.get_config_dataset(),
                "models": self.get_config_models(),
                "condense": self.get_config_condense(),
                "evaluate": self.get_config_evaluate(),
            }
        self.config = self.override_config(config, **self.args)
        return deepcopy(self.config)

    def get_set_metaconfig(self) -> dict:
        self.metaconfig = {
            "name": f"cds-{self.dataset}-{self.llm}-{self.encoder}-{self.influencer}-{self.condense}",
            "expt": f"expt_{self.condense}",
            "path": "./logs/condense",
            "args": self.args,
            "run": "eval:f'{run}'",
        }
        return self.metaconfig

    def get_config_dataset(self) -> dict:
        splits = {
            "agnews": ["train", "test"],
            "imdb": ["train", "test"],
            "mnlim": ["train", "validation_matched", "validation_mismatched"],
            "qqp": ["train", "validation"],
            "sst2": ["train", "validation"],
        }
        name_gen = f"gen-{self.dataset}-{self.llm}"
        config = {
            "abbrev": self.dataset,
            "splits": splits[self.dataset],
            "splits_custom": {
                "pool": {
                    "init_with": "from_csv",
                    "from_csv_kwargs": {
                        "path_or_paths": f"./logs/generate/{name_gen}/{name_gen}.csv",
                    },
                    "cast_label": True,
                }
            },
            "cast_label": True,
            "preembed": True,
        }
        if self.dataset in ["mnlim", "qqp", "sst2"]:
            config["unify_text"] = True
        return config

    def get_config_models(self) -> dict:
        config = {}
        splits_corpus = "+".join(self.metadata_dataset.get_preset()["splits_corpus"])
        if self.encoder == "tfidf":
            raise NotImplementedError("Please code this asap!")
        elif self.encoder == "glove":
            config["encoder"] = self.metadata_encoder.get_preset()
            if self.influencer in ["logistic", "svm"]:
                config["encoder"]["kwargs"]["embed_level"] = "sentence"
        elif self.encoder in ["e5", "jina_nano", "jina_small", "minilm"]:
            config["encoder"] = self.metadata_encoder.get_preset()
        else:
            # Maybe a larger one also since minilm can only handle 256 first tokens
            raise ValueError(f"Unsupported encoder: {self.encoder}")

        config["influencer"] = self.metadata_influencer.get_preset()
        if self.influencer in ["logistic", "svm"]:
            extra_kwargs = {
                "input_dim": config["encoder"]["embed_dim"],
                "num_classes": self.metadata_dataset.num_classes,
            }
            config["influencer"]["kwargs"].update(extra_kwargs)
        elif self.influencer in ["textcnn", "textrnn"]:
            extra_kwargs = {
                "embed_dim": config["encoder"]["embed_dim"],
                "num_classes": self.metadata_dataset.num_classes,
            }
            config["influencer"]["kwargs"].update(extra_kwargs)
        else:
            raise ValueError(f"Unsupported influencer: {self.influencer}")

        config["llm"] = {
            "model": {
                "abbrev": self.llm,
                **self.metadata_llm.get_preset_model(),
                "load_state_dict": {
                    "f": f"./logs/finetune/{self.dataset}-{self.llm}/lora.pt",
                    "weights_only": True,
                },
            },
            "tokenizer": self.metadata_llm.get_preset_tokenizer(),
        }

        if self.condense == "discreteot":
            lr = {"logistic": 3e-3, "svm": 3e-3, "textcnn": 3e-4, "textrnn": 3e-4}[self.influencer]
            config["take"] = {
                "kwargs": {
                    "params_inf": "linear",  # linear for logistic, svm, textcnn, textrnn, ? albert
                    "temporal_kernel": "exponential",
                    "loss_fn": "ce",
                    "verbose": True,
                    "device": "auto",
                },
                "kwargs_call": {
                    "trajectory_dir": None,
                    "opt_kwargs": {
                        "Class": "eval:torch.optim.AdamW",
                        "kwargs": {
                            "lr": lr,
                            "weight_decay": 5e-4,
                        },
                    },
                    "batch_size": 128,
                    "num_updates_per_step": 2,
                    "num_steps": 50,
                },
            }
        return config

    def get_config_condense(self) -> dict:
        path = self.metaconfig["path"]
        name = self.metaconfig["name"]

        n_train_samples = self.metadata_dataset.get_preset()["original_splits"]["train"]["num_rows"]
        config = {
            "add_train": False,
            "n_samples": n_train_samples // 1000,
            "conditional": True,
            "export_csv": f"eval:f'./{path}/{name}/{name}-run={{run}}.csv'",
        }

        if self.condense == "kmeans":
            config_kmeans = {
                "kmeans_fit_kwargs": {
                    "num_epochs": 200,
                    "centroids_init": "kmeans++",
                    "save_history": False,
                }
            }
            config.update(config_kmeans)
        elif self.condense == "discreteot":
            config_dot = {
                "dot_kwargs": {
                    "init_strat": "kmeans",
                    "mode": "cosine",
                    "batch_size": "sqrt",
                    "reg": 0.02,
                    "accept_strat": "strict",
                    "temp_scheduler": "eval:TemperatureScheduler(start_temp=1e-2, stop_temp=1e-4, strategy='cosine')",
                },
                "dot_fit_kwargs": {"num_epochs": 200},
            }
            config.update(config_dot)
        else:
            raise NotImplementedError(f"Unsupported condense method: {self.condense}.")

        return config

    def get_config_evaluate(self) -> dict:
        config = {}

        classifier = "logistic"
        metadata_logistic = ClassifierMetadata(model=classifier)

        config.update({"metrics": "all", "n_samples": 1000, "verbose": True})
        config.update({"length": {}})
        config.update({"perplexity": {}})
        config.update({"distinctn": {}})
        config.update({"selfbleu": {}})
        config.update({"dcr": {"splits": ["train", "condense:train", "test:train"]}})
        config.update({"mauve": {"splits": ["condense:train", "condense:test"]}})

        config_accuracy_lgr = {
            "dataset": {"loader_kwargs": {"batch_size": {"test": 256}, "num_workers": 4}},
            "models": {
                "classifier": {
                    "abbrev": "logistic",
                    "Class": "eval:LogisticRegression",
                    "kwargs": {
                        **metadata_logistic.get_preset()["kwargs"],
                        "input_dim": self.metadata_encoder.get_preset()["embed_dim"],
                        "num_classes": self.metadata_dataset.num_classes,
                    },
                    "load_state_dict": {
                        "f": "./logs/classify/clf-logistic-tfidf-agnews/clf-logistic-tfidf-agnews - run 0.pt",
                        "weights_only": True,
                    },
                }
            },
            "trainer": {
                "fit_with": "clf_trainer",
                "Class": "eval:ClassifierTrainer",
                "kwargs": {
                    "loss_fn": "eval:torch.nn.CrossEntropyLoss()",
                    "num_classes": 4,
                },
                "L_trainer_kw": {
                    "enable_checkpointing": False,
                    "enable_progress_bar": False,
                    "logger": False,
                },
            },
        }
        config.update({"accuracy_lgr": config_accuracy_lgr})

        config_utility_lgr = {
            "dataset": {
                "loader_kwargs": {
                    "batch_size": {"train": 2, "test": 256},
                    "shuffle": {"train": True},
                    "num_workers": 4,
                },
            },
            "models": {
                "classifier": {
                    "abbrev": "logistic",
                    "Class": "eval:LogisticRegression",
                    "kwargs": {
                        **metadata_logistic.get_preset()["kwargs"],
                        "input_dim": self.metadata_encoder.get_preset()["embed_dim"],
                        "num_classes": self.metadata_dataset.num_classes,
                    },
                }
            },
            "trainer": {
                "fit_with": "clf_trainer",
                "Class": "eval:ClassifierTrainer",
                "kwargs": {
                    "loss_fn": "eval:torch.nn.CrossEntropyLoss()",
                    "optimizer_kw": {
                        "Class": "eval:torch.optim.AdamW",
                        "kwargs": {"lr": 0.003, "weight_decay": 0.0005},
                    },
                    "num_classes": 4,
                },
                "L_trainer_kw": {
                    "callbacks": {"printer": {"on_event": "train_epoch_end"}},
                    "check_val_every_n_epoch": 1,
                    "enable_checkpointing": False,
                    "enable_progress_bar": False,
                    "max_epochs": 20,
                    "logger": False,
                },
                "fit_kw": {},
            },
        }
        config.update({"utility_lgr": config_utility_lgr})

        return config

    def override_config(self, config: dict, **kwargs) -> dict:
        keys_avail = []
        for k, v in self.override_args.items():
            assert k in keys_avail, f"Unknown key: {k}"

            if k == "dataset_path":
                config["dataset"]["splits_custom"]["pool"]["from_csv_kwargs"]["path_or_paths"] = v
            elif k == "add_train":
                config["condense"]["add_train"].append(v)
            elif k == "take.num_updates_per_step":
                config["take"]["kwargs_call"]["num_updates_per_step"] = v
            elif k == "n_samples":
                config["condense"]["n_samples"] = v
            elif k == "conditional":
                config["condense"]["conditional"] = v
            elif k == "export_csv":
                config["condense"]["export_csv"] = v
            elif k == "eval.metrics":
                list_metrics = list(v.split(","))
                config["evaluate"]["metrics"] = list_metrics
                for m in list_metrics:
                    if m in config["evaluate"]:
                        config["evaluate"].pop(m)
            elif k == "eval.n_samples":
                config["evaluate"]["n_samples"] = v
            else:
                raise ValueError(f"Unknown key: {k}")

        return config

    def export_config(self, path_config: str | None = None, verbose: bool = True) -> str:
        if path_config is None:
            path = self.metaconfig["path"]
            name = self.metaconfig["name"]
            args_str = ""
            for k, v in self.args.items():
                whitelist = ["base_config", "run", "n_runs"]
                whitelist.extend(["dataset", "llm", "encoder", "influencer", "condense"])
                if k in whitelist:
                    continue
                args_str += f"-{k}={v}"
            path_config = f"{path}/{name}/{name}-config{args_str}.yaml"

        os.makedirs(os.path.dirname(path_config), exist_ok=True)
        with open(path_config, "w") as f:
            yaml.dump(data=self.config, stream=f, sort_keys=False)

        if verbose:
            print(f"Config exported to {path_config}.")

        return path_config


def preprocess_dataset(dataset: DatasetDict, encoder, config: dict) -> DatasetDict:
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

    if config.get("preembed") is True:

        def preembed(batch: dict) -> dict:
            batch["embedding"] = encoder(batch["text"]).cpu()
            return batch

        assert encoder is not None, "Encoder must be provided for pre-embedding the dataset."
        dataset = dataset.map(function=preembed, batched=True, batch_size=4)

    if config.get("add_train") is True:
        dataset["pool"] = concatenate_datasets([dataset["pool"], dataset["train"]])
        dataset["pool"] = dataset["pool"].add_column(name="id", column=range(len(dataset["pool"])))

    return dataset


def shuffle_trim(dataset: Dataset, n_samples: int) -> Dataset:
    idx_shuffle = torch.randperm(n=len(dataset))[0 : min(n_samples, len(dataset))].tolist()
    dataset = dataset.select(indices=idx_shuffle)
    return dataset


def evaluate_dataset(config: dict | None, dataset: DatasetDict, llm, tokenizer) -> dict:
    if config is None:
        return {}

    def tokenize(batch: dict) -> dict:
        batch.update(tokenizer(text=batch["text"]))
        return batch

    dataset = dataset.map(function=tokenize, batched=True)
    dataset_trim = {
        split: shuffle_trim(dataset=dataset[split], n_samples=config["n_samples"])
        for split in dataset.keys()
    }

    # TODO: Add DDP
    metrics = {}
    if "length" in config["metrics"]:
        metrics["length"] = {}
        for split in dataset.keys():
            lengths = torch.tensor([len(tk) for tk in dataset_trim[split]["input_ids"]])
            metrics["length"][split] = lengths.float().mean(dim=0).item()

    if "perplexity" in config["metrics"]:
        ppl = Perplexity(
            model=llm,
            tokenizer=tokenizer,
            data_collator=ClosedEndedCollator(tokenizer=tokenizer, mlm=False),
        )
        results = {}
        for split in dataset.keys():
            results[split] = ppl(dataset_trim[split])
        metrics["perplexity"] = results

    if "distinctn" in config["metrics"]:
        ns = [1, 2, 3]
        if config.get("distinctn", {}).get("n") is not None:
            ns = config["distinctn"]["n"]

        results = {}
        for split in dataset.keys():
            results[split] = {}
            for n in ns:
                distinctn = DistinctN(n=n)
                results_n = distinctn(tokens=[t.tolist() for t in dataset_trim[split]["input_ids"]])
                results[split][f"distinct-{n}"] = {
                    "corpus": results_n["corpus"],
                    "sample_mean": torch.tensor(results_n["sample"]).mean().item(),
                }
        metrics["distinctn"] = results

    if "selfbleu" in config["metrics"]:
        # Can be time-consuming for large datasets - complexity O(N^2)
        kwargs = {}
        if config.get("selfbleu", {}).get("kwargs") is not None:
            kwargs = config["selfbleu"]["kwargs"]
        selfbleu = SelfBLEU(**kwargs)

        results = {}
        for split in dataset.keys():
            tokens = [t.tolist() for t in dataset_trim[split]["input_ids"]]
            results[split] = selfbleu(tokens=tokens)["corpus"]
        metrics["selfbleu"] = results

    if "dcr" in config["metrics"]:
        dcr = DistanceToClosestRecord(distance_fn="euclidean")
        results = {}
        for split in config["dcr"]["splits"]:
            split_list = split.split(":")
            emb1: Tensor = dataset_trim[split_list[0]]["embedding"][:]
            emb2: Tensor | None = None
            if len(split_list) == 2:
                emb2 = dataset_trim[split_list[1]]["embedding"][:]
            results[split] = dcr(input=emb1, other=emb2)
        metrics["dcr"] = results

    if "mauve" in config["metrics"]:
        results = {}
        for split in config["mauve"]["splits"]:
            split_list = split.split(":")
            emb1: Tensor = dataset_trim[split_list[0]]["embedding"][:]
            emb2: Tensor = dataset_trim[split_list[1]]["embedding"][:]
            results[split] = mauve.compute_mauve(p_features=emb1, q_features=emb2).mauve
        metrics["mauve"] = results

    if "accuracy_lgr" in config["metrics"]:
        config_lgr = deepcopy(config["accuracy_lgr"])
        dataset_lgr = {
            split: dataset[split].select_columns(["text", "label"]) for split in dataset.keys()
        }

        dataloader = get_dataloader(dataset=dataset_lgr, **config_lgr["dataset"]["loader_kwargs"])
        classifier: nn.Module = get_classifier(config=config_lgr["models"]["classifier"])
        encoder = get_encoder(config=config_lgr["models"]["encoder"])
        trainer: ClassifierTrainer = config_lgr["trainer"]["Class"](
            classifier=classifier,
            encoder=encoder,
            **config_lgr["trainer"]["kwargs"],
        )

        metrics["accuracy_lgr"] = {}
        for split in dataset.keys():
            results = trainer.evaluate(
                **config_lgr["trainer"]["L_trainer_kw"],
                eval_kw={"dataloaders": dataloader[str(split)], **config_lgr["trainer"]["eval_kw"]},
            )
            metrics["accuracy_lgr"][split] = results

    if "utility_lgr" in config["metrics"]:
        # Accuracy evaluation does not use trimmed dataset
        config_lgr = deepcopy(config["utility_lgr"])
        dataset_lgr = {
            "train": dataset["condense"].select_columns(["text", "label"]),
            "test": dataset["test"].select_columns(["text", "label"]),
        }
        dataloader = get_dataloader(dataset=dataset_lgr, **config_lgr["dataset"]["loader_kwargs"])
        classifier, results = train_classifier(
            config=config_lgr["trainer"],
            classifier=get_classifier(config=config_lgr["models"]["classifier"]),
            encoder=get_encoder(config=config_lgr["models"]["encoder"]),
            dataloader=dataloader,
        )
        metrics["utility_lgr"] = results

    if config.get("verbose"):
        pprint(metrics)
    return metrics


def export_dataset(
    dataset: Dataset,
    metadata: DatasetMetadata,
    path_csv: str,
    remove_columns: list[str] = ["id", "embedding"],
    verbose: bool = True,
):
    for col_name in remove_columns:
        if col_name in dataset.column_names:
            dataset = dataset.remove_columns(column_names=[col_name])
    dataset = dataset.sort(column_names=["label", "text"])

    def map_fn(batch: dict) -> dict:
        batch["label_str"] = metadata.idx_2_label(batch["label"].tolist())
        return batch

    dataset = dataset.map(function=map_fn)
    dataset = dataset.remove_columns(column_names=["label"])
    dataset = dataset.rename_column(original_column_name="label_str", new_column_name="label")
    dataset = dataset.select_columns(["label", "text"])

    if verbose:
        print(f"Condensed dataset class distribution: {Counter(dataset['label'])}")

    dataset.to_csv(path_or_buf=path_csv, index=False)
    if verbose:
        print(f"Dataset exported to {path_csv}")


def expt_cds_kmeans(config: dict, run: int = 0) -> Dataset:
    metadata = DatasetMetadata(dataset=config["dataset"]["abbrev"])
    encoder: nn.Module | None = get_encoder(config=config["models"]["encoder"])

    dataset: DatasetDict = get_dataset(config=config["dataset"])
    dataset = preprocess_dataset(dataset=dataset, encoder=encoder, config=config["dataset"])
    pool = dataset["pool"]

    if config["condense"]["conditional"]:
        cds_datasets: list[Dataset] = []
        n_samples_by_class = balanced_partition(
            total=config["condense"]["n_samples"],
            num_parts=metadata.num_classes,
        )

        for k in range(metadata.num_classes):
            kmeans = KMeansClassifier(K=n_samples_by_class[k])
            idx_k_dtrain = (
                (ensure_tensor(dataset["train"]["label"][:]) == k).nonzero().squeeze(dim=1)
            )
            centroids, _ = kmeans.fit(
                X_train=dataset["train"]["embedding"][idx_k_dtrain],
                **config["condense"]["kmeans_fit_kwargs"],
            )
            idx_k_pool = ensure_tensor(pool["label"][:] == k).nonzero().squeeze(dim=1)
            pool_k = pool.select(indices=idx_k_pool.tolist())
            dist = torch.cdist(pool_k["embedding"][:], centroids, 2)
            idx_closest = dist.argmin(dim=0)
            cds_datasets.append(pool_k.select(indices=idx_closest.tolist()))
        dataset["condense"] = concatenate_datasets(cds_datasets)
    else:
        kmeans = KMeansClassifier(K=config["condense"]["n_samples"])
        centroids, _ = kmeans.fit(
            X_train=ensure_tensor(dataset["train"]["embedding"][:]),
            **config["condense"]["kmeans_fit_kwargs"],
        )
        dist = torch.cdist(x1=ensure_tensor(pool["embedding"][:]), x2=centroids, p=2)
        idx_closest = dist.argmin(dim=0)
        dataset["condense"] = pool.select(indices=idx_closest.tolist())

    export_dataset(
        dataset=dataset["condense"],
        metadata=metadata,
        path_csv=config["condense"]["export_csv"],
    )

    llm = get_llm_model(config=config["models"]["llm"]["model"])
    tokenizer = get_llm_tokenizer(config=config["models"]["llm"]["tokenizer"])
    evaluate_metrics = evaluate_dataset(
        config=config["evaluate"],
        dataset=dataset,
        llm=llm,
        tokenizer=tokenizer,
    )
    pprint(evaluate_metrics)
    return dataset["condense"]


def expt_cds_discreteot(config: dict, run: int = 0):
    metadata = DatasetMetadata(dataset=config["dataset"]["abbrev"])
    encoder: nn.Module | None = get_encoder(config=config["models"]["encoder"])
    dataset: DatasetDict = get_dataset(config=config["dataset"])
    dataset = preprocess_dataset(dataset=dataset, encoder=encoder, config=config["dataset"])
    pool = dataset["pool"]

    # Prepare source and pool embeddings
    # embedding maybe optional, e.g, for BERT
    source_emb = ensure_tensor(dataset["train"]["embedding"][:])
    pool_emb = ensure_tensor(pool["embedding"][:])

    # Compute knowledge values (weights) for source samples
    influencer: nn.Module = get_classifier(config=config["models"]["influencer"])
    take = TrajectoryAwareKnowledgeEstimator(model=influencer, **config["models"]["take"]["kwargs"])
    knowledge = take(
        inputs=source_emb,
        targets=ensure_tensor(dataset["train"]["label"][:]),
        **config["models"]["take"]["kwargs_call"],
    )

    if config["condense"].get("conditional", False):
        cds_datasets: list[Dataset] = []
        n_samples_by_class = balanced_partition(
            total=config["condense"]["n_samples"],
            num_parts=metadata.num_classes,
        )
        for k in range(metadata.num_classes):
            idx_k_train = ensure_tensor(dataset["train"]["label"][:] == k).nonzero().squeeze(dim=1)
            idx_k_pool = ensure_tensor(pool["label"][:] == k).nonzero().squeeze(dim=1)
            source_k = source_emb[idx_k_train]
            pool_k = pool_emb[idx_k_pool]
            knowledge_k = knowledge[idx_k_train]
            distiller = DiscreteOTDistiller(
                source=source_k,
                pool=pool_k,
                k=n_samples_by_class[k],
                source_weights=knowledge_k,
                **config["condense"]["dot_kwargs"],
            )
            idx_cds_k, logs_k = distiller.fit(**config["condense"]["dot_fit_kwargs"])
            cds_datasets.append(pool.select(indices=idx_k_pool[idx_cds_k].tolist()))
        dataset["condense"] = concatenate_datasets(cds_datasets)
    else:
        distiller = DiscreteOTDistiller(
            source=source_emb,
            pool=pool_emb,
            k=config["condense"]["n_samples"],
            source_weights=knowledge,
            **config["condense"]["dot_kwargs"],
        )
        idx_cds, logs = distiller.fit(**config["condense"]["dot_fit_kwargs"])
        dataset["condense"] = pool.select(indices=idx_cds.tolist())

    export_dataset(
        dataset=dataset["condense"],
        metadata=metadata,
        path_csv=config["condense"]["export_csv"],
    )

    llm = get_llm_model(config=config["models"]["llm"]["model"])
    tokenizer = get_llm_tokenizer(config=config["models"]["llm"]["tokenizer"])
    evaluate_metrics = evaluate_dataset(
        config=config["evaluate"],
        dataset=dataset,
        llm=llm,
        tokenizer=tokenizer,
    )
    pprint(evaluate_metrics)
    return dataset["condense"]


def get_expt(expt: str) -> Callable:
    if expt == "expt_kmeans":
        return expt_cds_kmeans
    elif expt == "expt_discreteot":
        return expt_cds_discreteot
    elif expt in ["cds_influence", "cds_residual"]:
        raise NotImplementedError(f"Deprecated expt: {expt}.")
    else:
        raise NotImplementedError(f"Unknown experiment: {expt}")


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
    args_group.add_argument("--dataset", type=str, required=True, choices=DatasetMetadata.supported)
    args_group.add_argument("--dataset_path", type=str)
    args_group = args.add_argument_group("Models arguments")
    args_group.add_argument("--llm", type=str, default="gemma3_270m", choices=LLMMetadata.supported)
    args_group.add_argument("--encoder", type=str, default="minilm", choices=EncoderMetadata.supported)
    args_group.add_argument("--influencer", type=str, default="logistic", choices=ClassifierMetadata.supported)
    args_group.add_argument("--take.num_updates_per_step", type=int)
    args_group = args.add_argument_group("Condense arguments")
    args_group.add_argument("--condense", type=str, default="discreteot", choices=["kmeans", "discreteot"])
    args_group.add_argument("--add_train", type=TypeArgparse.bool_strict)
    args_group.add_argument("--n_samples", type=int)
    args_group.add_argument("--conditional", type=TypeArgparse.bool_strict)
    args_group.add_argument("--export_csv", type=str)
    args_group = args.add_argument_group("Evaluate arguments")
    args_group.add_argument("--eval.metrics", type=str)
    args_group.add_argument("--eval.n_samples", type=TypeArgparse.bool_strict)
    args = args.parse_args()
    custom_args = {k: v for k, v in vars(args).items() if v is not None}
    # fmt: on
    parser = ConfigParser(globals=globals(), locals=locals())
    config_factory = ConfigFactory(**custom_args)
    config = config_factory.get_config()
    config_factory.export_config(verbose=True)
    for run in rename_runs(run=args.run, n_runs=args.n_runs):
        config_run = parser.parse_eval_config(deepcopy(config), parse_flag="eval:")
        pprint(config_run)
        expt = get_expt(expt=config_run["metaconfig"]["expt"])
        expt(config=config_run, run=run)
