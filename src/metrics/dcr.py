from typing import Callable

import torch
from torch import Tensor


class DistanceToClosestRecord:
    """
    Distance to Closest Record (DCR) for text embeddings. Computes average minimum distance from
    each embedding in one set to the closest embedding in another set.

    Reports average and per-sample distances for:
    - syn-real: synthetic to real
    - syn-syn: synthetic to synthetic (excluding self)
    - real-real: real to real (excluding self)

    Args:
    + `distance_fn`: Function to compute distance between two sets of embeddings. Defaults to
        `"euclidean"`.
    """

    distance_fn: Callable[[Tensor, Tensor], Tensor]

    def __init__(self, distance_fn: str | Callable[[Tensor, Tensor], Tensor] = "euclidean"):
        if isinstance(distance_fn, str):
            if distance_fn not in ["euclidean"]:
                raise ValueError("distance_fn must be 'euclidean'")
            else:
                self.distance_fn = DistanceToClosestRecord.euclidean_distance
        elif callable(distance_fn):
            self.distance_fn = distance_fn
        else:
            raise TypeError("distance_fn must be a string or a callable function")

    @staticmethod
    def euclidean_distance(input: Tensor, other: Tensor) -> Tensor:
        dist = torch.cdist(input, other, p=2)
        return dist

    def __call__(self, input: Tensor, other: Tensor | None = None) -> float:
        if other is None:
            against_self = True
            other = input.clone()
        else:
            against_self = False

        dist = self.distance_fn(input, other)
        if against_self:
            mask = torch.eye(len(input), device=input.device).bool()
            dist = dist.masked_fill(mask, float("inf"))
        dcr = dist.min(dim=1)[0].mean(dim=0).item()
        return dcr


if __name__ == "__main__":
    import os
    import sys

    repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
    assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    from transformers import AutoTokenizer
    from datasets import load_dataset, Dataset
    from sentence_transformers import SentenceTransformer
    from models.encoders.minilm import MiniLMWrapper

    dataset: list[Dataset] = load_dataset(
        path="fancyzhx/ag_news",
        cache_dir="./datasets",
        split=["train[:100]", "test[:100]"],
    )
    encoder = SentenceTransformer(
        model_name_or_path="all-MiniLM-L6-v2",
        cache_folder="./pretrained/encoders/all-MiniLM-L6-v2",
    )
    encoder = MiniLMWrapper(model=encoder)

    def preembed(batch: dict) -> dict:
        batch["embedding"] = encoder(batch["text"]).tolist()
        return batch

    dataset = [d.map(preembed, batched=True) for d in dataset]

    dcr = DistanceToClosestRecord(distance_fn="euclidean")
    results = dcr(
        input=torch.tensor(dataset[0]["embedding"]),
        other=torch.tensor(dataset[1]["embedding"]),
    )
    print("Distance to Closest Record (DCR) results:", results)
