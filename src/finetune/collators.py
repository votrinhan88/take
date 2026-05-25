import torch
from transformers import DataCollatorForLanguageModeling


class ClosedEndedCollator(DataCollatorForLanguageModeling):
    """
    Custom `DataCollatorForLanguageModeling` for closed-ended tasks, such as structured generation.

    In details, language models are typically trained for open-ended settings, where <eos> tokens
    are set to `-100`, and are ignored during loss computation. This collator sets the first <eos>
    token as trainable so models can learn when to stop generation.
    Link: https://github.com/huggingface/transformers/issues/23530
    """

    ignore_token_id = -100

    def __call__(self, features: dict, *args, **kwargs) -> dict:
        batch: dict = super().__call__(features, *args, **kwargs)
        labels = batch["labels"]

        # Set the very first eos_token as trainable
        mask = (labels == self.ignore_token_id)
        eos_pos = torch.where(
            condition=mask.any(dim=1),
            input=mask.int().argmax(dim=1),
            other=-1,
        )
        labels[torch.arange(labels.shape[0]), eos_pos] = self.tokenizer.eos_token_id

        batch["labels"] = labels
        return batch


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_dataset, Dataset

    dataset: Dataset = load_dataset(
        path="fancyzhx/ag_news",
        cache_dir="./datasets",
        split="train[:1%]",
    )
    from src.utils.typing import Tokenizer
    tokenizer: Tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="gpt2", cache_dir="./pretrained"
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(
        dataset[0:4]["text"],
        truncation=False,
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    collator = ClosedEndedCollator(tokenizer=tokenizer, mlm=False)
    out = collator(tokens)
    print(out)
