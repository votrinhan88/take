from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np


class Perplexity:
    """Compute perplexity for language models using HuggingFace Trainer.

    Args:
    + `model`.
    + `tokenizer`.
    + `training_args`. Defaults to `"auto"`.
    + `data_collator`. Defaults to `"causal_lm"`.

    ---
    Example:
    ```
    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        training_args="auto",
        data_collator="causal_lm",
    )
    # dataset: contains model inputs fields, e.g., "input_ids", "attention_mask"
    perplexity = ppl(dataset)
    ```
    """

    default_training_args = TrainingArguments(
        per_device_eval_batch_size=4,
        report_to="none",
        auto_find_batch_size=True,
    )

    def __init__(
        self,
        model,
        tokenizer,
        training_args: TrainingArguments | str = "auto",
        data_collator = "causal_lm",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = self._validate_args("training_args", training_args)
        self.data_collator = self._validate_args("data_collator", data_collator)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
        )
    
    def _validate_args(self, arg: str, value):
        if arg == "training_args":
            if value == "auto":
                return self.default_training_args
            else:
                return value
        
        elif arg == "data_collator":
            if value == "causal_lm":
                return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
            else:
                return value

    def __call__(self, *args, **kwargs) -> float:
        results = self.trainer.evaluate(*args, **kwargs)
        perplexity = np.exp(results["eval_loss"]).item()
        return perplexity
