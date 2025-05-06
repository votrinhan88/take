from torch import nn, Tensor


class BatchUnpacker(nn.Module):
    def __init__(self, key_inputs: str | None = None, key_targets: str | None = None):
        super().__init__()
        self.key_inputs = key_inputs
        self.key_targets = key_targets

    def forward(self, batch: dict[str, Tensor]):
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch

        if isinstance(batch, dict):
            if self.key_inputs is not None:
                inputs = batch[self.key_inputs]
            elif "text" in batch.keys():
                inputs = batch["text"]
            elif "input_ids" in batch.keys():
                inputs = batch["input_ids"]
            else:
                raise ValueError(f"Unknown key for `inputs` from: {batch.keys()}")

            if self.key_targets is not None:
                targets = batch[self.key_targets]
            elif "label" in batch.keys():
                targets = batch["label"]
            elif "labels" in batch.keys():
                targets = batch["labels"]
            else:
                raise ValueError(f"Unknown key for `targets` from: {batch.keys()}")

        return inputs, targets
