import torch
from torch import nn, Tensor
from transformers import AutoModel


class JinaWrapper(torch.nn.Module):
    def __init__(self, model, task: str = "classification"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(device=self.device)
        self.task = task

    def forward(self, *args, **kwargs) -> Tensor:
        embedding = self.model.encode(*args, **kwargs, task=self.task)
        return embedding

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


if __name__ == "__main__":
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path="jinaai/jina-embeddings-v5-text-nano",
        cache_dir="./models/pretrained/encoders",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    encoder = JinaWrapper(model=model, task="classification")

    input_text = ["Hello, how are you?"]
    embedding = encoder(input_text)
    print(embedding.shape)
