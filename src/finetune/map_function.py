import warnings

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer

from src.metadata import DatasetMetadata, LLMMetadata
from src.utils.typing import Tokenizer


class InstructionFinetuneMapFunction:
    supported_tasks = ["generation", "inference"]

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset: str,
        model: str,
        task: str,
        max_length: int | None = None,
        batched: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.model = model
        self.task = self._validate_args("task", task)
        self.max_length = max_length
        self.batched = batched

        self.dataset_metadata = DatasetMetadata(dataset=dataset)
        self.classes = self.dataset_metadata.classes

        self.model_metadata = LLMMetadata(model=model)
        self.max_length: int = self._validate_args("max_length", self.max_length)
        self.len_label_tokens = self.find_max_length_label_tokens()
        self.len_task_tokens = self.find_max_length_task_tokens()
        # reserve <bos>, <eos>
        self.len_text_tokens = self.max_length - self.len_label_tokens - self.len_task_tokens - 2

        if self.task == "generation":
            self.map_batch = self.map_batch_generation
            self.map_sample = self.map_sample_generation
        elif self.task == "inference":
            self.map_batch = self.map_batch_inference
            self.map_sample = self.map_sample_inference

    def _validate_args(self, arg: str, value):
        if arg == "task":
            if value not in self.supported_tasks:
                raise ValueError("`task` must be one of ['generation', 'inference'].")
            return value

        elif arg == "max_length":
            if value is None:
                return self.model_metadata.context_length
            else:
                if value > self.model_metadata.context_length:
                    warnings.warn(
                        f"max_length excesed model context length of"
                        f"{self.model_metadata.context_length}."
                    )
                return value

    def find_max_length_label_tokens(self) -> int:
        max_len_label_tokens = 0
        for label in self.classes:
            label_str = self.format_label(label)
            label_tok = self.tokenizer(text=label_str, add_special_tokens=False)
            label_len = len(label_tok["input_ids"])
            max_len_label_tokens = max(max_len_label_tokens, label_len)
        return max_len_label_tokens

    def find_max_length_task_tokens(self) -> int:
        max_len_task_tokens = 0
        for task in self.supported_tasks:
            task_str = self.format_task(task)
            task_tok = self.tokenizer(text=task_str, add_special_tokens=False)
            task_len = len(task_tok["input_ids"])
            max_len_task_tokens = max(max_len_task_tokens, task_len)
        return max_len_task_tokens

    def format_task(self, task: str | None = None) -> str:
        task = task if task is not None else self.task
        return f"Task: {task.title()}"

    def format_text(self, text: str) -> str:
        return f"\nText: {text}"

    def format_label(self, label: int | str) -> str:
        if isinstance(label, int):
            label = self.classes[label]
        return f"\nLabel: {label}"

    def __call__(self, sample: dict) -> dict:
        if self.batched:
            return self.map_batch(sample)
        else:
            return self.map_sample(sample)

    def map_sample_generation(self, sample: dict) -> dict:
        # Prepare tokens
        task_str = self.format_task()
        label_str = self.format_label(sample["label"])
        text_str = self.format_text(sample["text"])
        all_tok = self.tokenizer(
            text=self.tokenizer.bos_token + task_str + label_str + text_str,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,  # reserve <eos>
        )
        eos_tok = self.tokenizer(
            text=self.tokenizer.eos_token,
            add_special_tokens=False,
        )
        # Concatenate
        for k in all_tok.keys():
            sample[k] = all_tok[k] + eos_tok[k]
        return sample

    def map_sample_inference(self, sample: dict) -> dict:
        # Prepare tokens
        task_str = self.format_task()
        text_str = self.format_text(sample["text"])
        label_str = self.format_label(sample["label"])
        tasktext_tok = self.tokenizer(
            text=self.tokenizer.bos_token + task_str + text_str,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - self.len_label_tokens - 1,  # reserve label and <eos>
        )
        label_str = self.tokenizer(
            text=label_str + self.tokenizer.eos_token,
            add_special_tokens=False,
        )
        # Concatenate
        for k in tasktext_tok.keys():
            sample[k] = tasktext_tok[k] + label_str[k]
        return sample

    def map_batch_generation(self, batch: dict) -> dict[str, Tensor]:
        task_str = self.format_task()
        labels = batch["label"]
        if isinstance(labels, Tensor):
            labels = labels.tolist()
        # Prepare tokens
        all_tok = self.tokenizer(
            text=[
                self.tokenizer.bos_token + task_str + self.format_label(l) + self.format_text(t)
                for l, t in zip(labels, batch["text"])
            ],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,  # reserve <eos>
        )
        eos_tok = self.tokenizer(
            text=[self.tokenizer.eos_token] * len(batch["text"]),
            add_special_tokens=False,
        )
        # Concatenate
        for k in all_tok.keys():
            batch[k] = []
            for t, l in zip(all_tok[k], eos_tok[k]):
                batch[k].append(t + l)
        return batch

    def map_batch_inference(self, batch: dict) -> dict[str, Tensor]:
        task_str = self.format_task()
        labels = batch["label"]
        if isinstance(labels, Tensor):
            labels = labels.tolist()
        # Prepare tokens
        tasktext_tok = self.tokenizer(
            text=[self.tokenizer.bos_token + task_str + self.format_text(t) for t in batch["text"]],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - self.len_label_tokens - 1,  # reserve label and <eos>
        )
        label_tok = self.tokenizer(
            text=[self.format_label(l) + self.tokenizer.eos_token for l in labels],
            add_special_tokens=False,
        )
        # Concatenate
        for k in tasktext_tok.keys():
            batch[k] = []
            for t, l in zip(tasktext_tok[k], label_tok[k]):
                batch[k].append(t + l)
        return batch
