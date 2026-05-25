from collections.abc import Callable
from typing import overload


class DatasetMetadata:
    supported = ["agnews", "imdb", "mnlim", "qqp", "sst2", "qnli"]
    supported_cls = ["agnews", "imdb", "sst2"]
    supported_nli = ["mnlim", "qqp", "qnli"]
    requires_unify_map = ["mnlim", "qqp", "sst2", "qnli"]

    def __init__(self, dataset: str = "agnews"):
        assert dataset in self.supported, f"Dataset {dataset} not supported."

        self.dataset = dataset

        preset_metadata = self.get_preset()
        self.task = preset_metadata["task"]
        self.classes = preset_metadata["classes"]
        self.original_splits = preset_metadata["original_splits"]
        self.text_keys = preset_metadata["text_keys"]

        self.num_classes = len(self.classes)
        self.class_2_idx = {k: i for i, k in enumerate(self.classes)}

    def get_preset(self) -> dict:
        # Add load_dataset_kwargs for HuggingFace datasets
        if self.dataset == "agnews":
            preset_metadata = {
                "task": "cls",
                "classes": ["World", "Sports", "Business", "Sci/Tech"],
                "original_splits": {
                    "train": {"features": ["text", "label"], "num_rows": 120000},
                    "test": {"features": ["text", "label"], "num_rows": 7600},
                },
                "text_keys": ["text"],
                "splits_corpus": ["train"],
                "load_dataset_kwargs": {"path": "fancyzhx/ag_news"},
            }

        elif self.dataset == "imdb":
            preset_metadata = {
                "task": "cls",
                "classes": ["neg", "pos"],
                "original_splits": {
                    "train": {"features": ["text", "label"], "num_rows": 25000},
                    "test": {"features": ["text", "label"], "num_rows": 25000},
                    "unsupervised": {"features": ["text", "label"], "num_rows": 50000},
                },
                "text_keys": ["text"],
                "splits_corpus": ["train", "unsupervised"],
                "load_dataset_kwargs": {"path": "stanfordnlp/imdb"},
            }

        elif self.dataset == "mnlim":
            preset_metadata = {
                "task": "nli",
                "classes": ["entailment", "neutral", "contradiction"],
                "original_splits": {
                    "train": {
                        "features": ["premise", "hypothesis", "label", "idx"],
                        "num_rows": 392702,
                    },
                    "validation_matched": {
                        "features": ["premise", "hypothesis", "label", "idx"],
                        "num_rows": 9815,
                    },
                    "validation_mismatched": {
                        "features": ["premise", "hypothesis", "label", "idx"],
                        "num_rows": 9832,
                    },
                    "test_matched": {
                        "features": ["premise", "hypothesis", "label", "idx"],
                        "num_rows": 9796,
                    },
                    "test_mismatched": {
                        "features": ["premise", "hypothesis", "label", "idx"],
                        "num_rows": 9847,
                    },
                },
                "text_keys": ["premise", "hypothesis"],
                "splits_corpus": ["train"],
                "load_dataset_kwargs": {"path": "glue", "name": "mnli"},
            }

        elif self.dataset == "qnli":
            preset_metadata = {
                "task": "nli",
                "classes": ["entailment", "not_entailment"],
                "original_splits": {
                    "train": {
                        "features": ["question", "sentence", "label", "idx"],
                        "num_rows": 104743,
                    },
                    "validation": {
                        "features": ["question", "sentence", "label", "idx"],
                        "num_rows": 5463,
                    },
                    "test": {
                        "features": ["question", "sentence", "label", "idx"],
                        "num_rows": 5463,
                    },
                },
                "text_keys": ["question", "sentence"],
                "splits_corpus": ["train"],
                "load_dataset_kwargs": {"path": "glue", "name": "qnli"},
            }

        elif self.dataset == "qqp":
            preset_metadata = {
                "task": "nli",
                "classes": ["not_duplicate", "duplicate"],
                "original_splits": {
                    "train": {
                        "features": ["question1", "question2", "label", "idx"],
                        "num_rows": 363846,
                    },
                    "validation": {
                        "features": ["question1", "question2", "label", "idx"],
                        "num_rows": 40430,
                    },
                    "test": {
                        "features": ["question1", "question2", "label", "idx"],
                        "num_rows": 390965,
                    },
                },
                "text_keys": ["question1", "question2"],
                "splits_corpus": ["train"],
                "load_dataset_kwargs": {"path": "glue", "name": "qqp"},
            }

        elif self.dataset == "sst2":
            preset_metadata = {
                "task": "cls",
                "classes": ["negative", "positive"],
                "original_splits": {
                    "train": {"features": ["idx", "sentence", "label"], "num_rows": 67349},
                    "validation": {"features": ["idx", "sentence", "label"], "num_rows": 872},
                    "test": {"features": ["idx", "sentence", "label"], "num_rows": 1821},
                },
                "text_keys": ["sentence"],
                "splits_corpus": ["train"],
                "load_dataset_kwargs": {"path": "stanfordnlp/sst2"},
            }

        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

        return preset_metadata

    def get_length_statistics(self) -> dict:
        """Returns token statistics by dataset.
        Obtained with gemma3_270m tokenizer on train split.
        """
        if self.dataset == "agnews":
            stats = {
                "mean": 53.28,
                "std": 18.65,
                "quantiles": {0: 14, 10: 35, 25: 42, 50: 51, 75: 60, 90: 71, 100: 372},
            }
        elif self.dataset == "imdb":
            stats = {
                "mean": 299.16,
                "std": 223.14,
                "quantiles": {0: 12, 10: 118, 25: 160, 50: 222, 75: 364, 90: 589, 100: 3048},
            }
        elif self.dataset == "mnlim":
            stats = {
                "mean": 37.91,
                "std": 19.13,
                "quantiles": {0: 3, 10: 17, 25: 24, 50: 35, 75: 48, 90: 61, 100: 440},
            }
        elif self.dataset == "qnli":
            stats = {
                "mean": 48.85,
                "std": 18.89,
                "quantiles": {0: 10, 10: 29, 25: 36, 50: 46, 75: 58, 90: 72, 100: 521},
            }
        elif self.dataset == "qqp":
            stats = {
                "mean": 28.25,
                "std": 12.89,
                "quantiles": {0: 4, 10: 16, 25: 20, 50: 25, 75: 33, 90: 45, 100: 331},
            }
        elif self.dataset == "sst2":
            stats = {
                "mean": 12.93,
                "std": 9.11,
                "quantiles": {0: 3, 10: 5, 25: 6, 50: 10, 75: 17, 90: 26, 100: 65},
            }
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")
        return stats

    def get_unify_map(self, inverse: bool = False) -> tuple[Callable[[dict], dict], dict]:
        """Returns a map function and kwargs for use with HuggingFace dataset.map to unify text columns.
        - For MNLI (mnlim): {'premise', 'hypothesis'} --> 'text'.
        - For QQP (qqp): {'question1', 'question2'} --> 'text'.
        - For SST-2 (sst2): 'sentence' --> 'text'.
        """

        if self.dataset == "mnlim":
            if not inverse:

                def map_fn(batch: dict) -> dict:
                    batch["text"] = [
                        f"<premise>{p}<hypothesis>{h}"
                        for p, h in zip(batch["premise"], batch["hypothesis"])
                    ]
                    return batch

                map_kwargs = {"batched": True, "remove_columns": ["premise", "hypothesis"]}
            else:

                def map_fn(batch: dict) -> dict:
                    p_raw, _, h = zip(*(t.partition("<hypothesis>") for t in batch["text"]))
                    batch["premise"] = [p.removeprefix("<premise>") for p in p_raw]
                    batch["hypothesis"] = list(h)
                    return batch

                map_kwargs = {"batched": True, "remove_columns": ["text"]}
            return map_fn, map_kwargs

        elif self.dataset == "qnli":
            if not inverse:

                def map_fn(batch: dict) -> dict:
                    batch["text"] = [
                        f"<question>{q1}<sentence>{q2}"
                        for q1, q2 in zip(batch["question"], batch["sentence"])
                    ]
                    return batch

                map_kwargs = {"batched": True, "remove_columns": ["question", "sentence"]}
            else:

                def map_fn(batch: dict) -> dict:
                    q_raw, _, s = zip(*(t.partition("<sentence>") for t in batch["text"]))
                    batch["question"] = [q.removeprefix("<question>") for q in q_raw]
                    batch["sentence"] = list(s)
                    return batch

                map_kwargs = {"batched": True, "remove_columns": ["text"]}
            return map_fn, map_kwargs

        elif self.dataset == "qqp":
            if not inverse:

                def map_fn(batch: dict) -> dict:
                    batch["text"] = [
                        f"<question1>{q1}<question2>{q2}"
                        for q1, q2 in zip(batch["question1"], batch["question2"])
                    ]
                    return batch

                map_kwargs = {"batched": True, "remove_columns": ["question1", "question2"]}
            else:

                def map_fn(batch: dict) -> dict:
                    q1_raw, _, q2 = zip(*(t.partition("<question2>") for t in batch["text"]))
                    batch["question1"] = [q1.removeprefix("<question1>") for q1 in q1_raw]
                    batch["question2"] = list(q2)
                    return batch

                map_kwargs = {"batched": True, "remove_columns": ["text"]}
            return map_fn, map_kwargs

        elif self.dataset == "sst2":
            if not inverse:

                def map_fn(batch: dict) -> dict:
                    batch["text"] = list(batch["sentence"])
                    return batch

                map_kwargs = {"batched": True, "remove_columns": ["sentence"]}
            else:

                def map_fn(batch: dict) -> dict:
                    batch["sentence"] = list(batch["text"])
                    return batch

                map_kwargs = {"batched": True, "remove_columns": ["text"]}
            return map_fn, map_kwargs

        else:
            raise NotImplementedError(f"Dataset {self.dataset} does not require text unification.")

    @overload
    def idx_2_label(self, indices: list[int]) -> list[str]: ...
    @overload
    def idx_2_label(self, indices: int) -> str: ...
    def idx_2_label(self, indices: list | int) -> list | str:
        if isinstance(indices, int):
            return self.classes[indices]
        labels = [self.classes[i] for i in indices]
        return labels

    @overload
    def label_2_idx(self, labels: list[str]) -> list[int]: ...
    @overload
    def label_2_idx(self, labels: str) -> int: ...
    def label_2_idx(self, labels: list | str) -> list | int:
        if isinstance(labels, str):
            return self.class_2_idx[labels]
        indices = [self.class_2_idx[l] for l in labels]
        return indices
