import torch
from torch import nn, Tensor

from typing import Optional


class CosineSimilarity(nn.Module):
    """
    Cosine similarity.
        sim(x1, x2) := cos(x1, x2) = (x1 . x2) / (||x1|| * ||x2||)
        sim(x1, x2) in [-1, 1]
    """

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if x1.is_sparse:
            x1 = x1.to_dense()
        if x2.is_sparse:
            x2 = x2.to_dense()

        x1_norm = x1 / x1.norm(dim=1, keepdim=True)
        x2_norm = x2 / x2.norm(dim=1, keepdim=True)
        sim = torch.mm(x1_norm, x2_norm.t())
        return sim


class ExponentialCosineSimilarity(nn.Module):
    """
    Take an exponential on cosine similarity to avoid negative values.
        sim(x1, x2) := exp(cos(x1, x2) - 1)
        sim(x1, x2) in [1/e^2, 1]
    """

    def __init__(self):
        super().__init__()
        self.cosine_sim = CosineSimilarity()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        sim = torch.exp(self.cosine_sim(x1, x2) - 1)
        return sim


class NormalizedCosineSimilarity(nn.Module):
    """
    Affine cosine similarity to avoid negative values.
        sim(x1, x2) := (cos(x1, x2) + 1)/2
        sim(x1, x2) in [0, 1]
    """

    def __init__(self):
        super().__init__()
        self.cosine_sim = CosineSimilarity()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        sim = (self.cosine_sim(x1, x2) + 1) / 2
        return sim


class InnerProductSimilarity(nn.Module):
    """
    Inner-product similarity.
        sim(x1, x2) := x1 . x2
        sim(x1, x2) in [-inf, inf]
    """

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        sim = torch.mm(x1, x2.t())
        if sim.is_sparse:
            sim = sim.to_dense()
        return sim


class JaccardSimilarity(nn.Module):
    """Jaccard Similarity: Measures fraction of overlapping non-zero indices. Suitable
    for purely lexical sparse embeddings (presence/absence).
        sim(x1, x2) := |intersection(x1, x2)| / |union(x1, x2)|`.
        sim(x1, x2) in [0, 1].

    Args:
      `threshold`: Threshold to binarize to [0, 1] for even sparser inputs. Defaults to\
        `0.5`.
      `epsilon`: Small term to add to denominator for numerical stability. Defaults to \
        `1e-8`.
    """

    def __init__(self, threshold: float = 0.5, epsilon: float = 1e-8):
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if x1.is_sparse:
            x1 = x1.to_dense()
        if x2.is_sparse:
            x2 = x2.to_dense()

        if (x1 < 0).any() or (x2 < 0).any():
            raise ValueError("Inputs must be non-negative for Generalized Jaccard Similarity.")

        x1 = (x1.unsqueeze(dim=1) > self.threshold).float()  # [N1, 1, D]
        x2 = (x2.unsqueeze(dim=0) > self.threshold).float()  # [1, N2, D]
        intersection = torch.min(x1, x2).sum(dim=2)  # [N1, N2]
        union = torch.max(x1, x2).sum(dim=2)  # [N1, N2]
        sim = intersection / (union + self.epsilon)
        return sim


class GeneralizedJaccardSimilarity(nn.Module):
    """Generalized Jaccard Similarity: Suitable for semi-sparse embeddings, such as
    SPLADE.
        sim(x, y) := [sum_{i} min(xi, yi)] / [sum_{i} max(xi, yi)]
        sim(x1, x2) in [0, 1].

    Args:
      `epsilon`: Small positive floatfor numerical stability. Defaults to `1e-8`.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x1: torch.sparse.Tensor, x2: torch.sparse.Tensor) -> torch.Tensor:
        """Use the following optimization trick for efficiency and numerical stability:
            max(a, b) = a + b - min(a, b)
        Thus, at each token position with value v1 from x1 and v2 from x2, we have:
            sim(x1, x2) := sum [min(v1, v2)] / sum [max(v1, v2)]
                        := sum [min(v1, v2)] / (sum(v1) + sum(v2) - sum[min(v1, v2)])
        """
        x1 = x1.coalesce()
        x2 = x2.coalesce()

        # Compute intersections by iterating over x2's non-zeros
        intersects = torch.zeros(
            size=[x1.shape[0], x2.shape[0]],
            device=x1.device,
            dtype=x1.dtype,
        )
        for i in range(x2.shape[0]):
            x2i = x2[i, :].coalesce()  # [V]
            mask_cols = x2i.indices()[0, :]

            # Slice x1 and x2i to only non-zero cols in x2i, shape [N1, V'] and [1, V']
            x1_masked = self.index_cols_coo(x=x1, cols=mask_cols)
            x2i_masked = self.index_cols_coo(x=x2i.unsqueeze(dim=0), cols=mask_cols)
            intersect_N1_i = torch.min(
                x1_masked.to_dense().unsqueeze(dim=1),
                x2i_masked.to_dense().unsqueeze(dim=0),
            )  # [N1, 1, V']

            intersects[:, i] = intersect_N1_i.squeeze(dim=1).sum(dim=1)  # [N1]

        # Compute unions: sum(max(a, b)) = sum(a) + sum(b) - sum(min(a, b))
        x1_sums = x1.sum(dim=1).to_dense()  # [N1]
        x2_sums = x2.sum(dim=1).to_dense()  # [N2]
        unions = x1_sums.unsqueeze(dim=1) + x2_sums.unsqueeze(dim=0) - intersects

        # Compute Jaccard with numerical stability
        sim = (intersects + self.epsilon) / (unions + self.epsilon)
        return sim

    def forward_dense(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Convert x1, x2 to dense then compute sim(x1, x2)."""
        if x1.is_sparse:
            x1 = x1.to_dense()
        if x2.is_sparse:
            x2 = x2.to_dense()

        if (x1 < 0).any() or (x2 < 0).any():
            raise ValueError("Inputs must be non-negative for Generalized Jaccard Similarity.")

        x1 = x1.unsqueeze(dim=1)  # [N1, 1, D]
        x2 = x2.unsqueeze(dim=0)  # [1, N2, D]
        intersection = torch.min(x1, x2).sum(dim=2)  # [N1, N2]
        union = torch.max(x1, x2).sum(dim=2)  # [N1, N2]
        sim = (intersection + self.epsilon) / (union + self.epsilon)
        return sim

    @staticmethod
    def index_cols_coo(x: Tensor, cols: Tensor) -> Tensor:
        """Index columns of a COO sparse tensor."""
        if not x.is_sparse:
            raise ValueError("Input tensor must be sparse.")

        x_coo = x.coalesce()
        mask = torch.isin(x_coo.indices()[1, :], cols)
        indices = x_coo.indices()[:, mask]
        values = x_coo.values()[mask]

        # Slice column indices to match cols
        hashmap = {col.item(): i for i, col in enumerate(cols)}
        indices[1, :] = indices[1, :].cpu().apply_(hashmap.get)

        x_out = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=[x.shape[0], cols.shape[0]],
            device=x_coo.device,
            dtype=x_coo.dtype,
        )
        return x_out


class RBFKernelSimilarity(nn.Module):
    def __init__(
        self,
        sigma: Optional[float] = None,
        embeddings: Optional[Tensor] = None,
    ):
        if (sigma is None) and (embeddings is None):
            raise ValueError("Either 'sigma' or 'embeddings' must be provided.")

        super().__init__()

        if sigma is not None:
            self.sigma = sigma
        elif embeddings is not None:
            self.sigma = self.infer_median(input=embeddings)

    def forward(self, input: Tensor, other: Tensor) -> Tensor:
        dist = torch.cdist(x1=input, x2=other, p=2)
        out = torch.exp(-(dist**2) / (2 * (self.sigma**2)))
        return out

    @staticmethod
    def infer_median(input: Tensor) -> float:
        dist: Tensor = torch.cdist(x1=input, x2=input, p=2)
        median = torch.median(dist.flatten())
        return median.item()


class CosineDissimilarity(nn.Module):
    """Cosine dissimilarity.
    dissim(x1, x2) := 1 - cos(x1, x2) = 1 - (x1 . x2) / (||x1|| * ||x2||)
    dissim(x1, x2) in [-1, 1]
    """

    def __init__(self, norm: bool = True):
        super().__init__()
        self.norm = norm

    def forward(self, x1: Tensor, x2: Tensor, norm: bool | None = None) -> Tensor:
        norm = self.norm if norm is None else norm
        if norm:
            x1 = x1 / (x1.norm(dim=1, keepdim=True) + 1e-9)
            x2 = x2 / (x2.norm(dim=1, keepdim=True) + 1e-9)
        dissim = 1 - torch.mm(x1, x2.t())
        return dissim


class CDist(nn.Module):
    """Matrix multiplication similarity.
    sim(x1, x2) := x1 . x2
    sim(x1, x2) in [-1, 1] if x1, x2 are normalized.
    """

    def __init__(self, p: int | float = 2, power: int = 1):
        super().__init__()
        self.p = p
        self.power = power

    def forward(self, x1: Tensor, x2: Tensor, norm: bool | None = None) -> Tensor:
        dist = torch.cdist(x1, x2, p=self.p) ** self.power
        return dist


if __name__ == "__main__":
    # Change path
    from sentence_transformers import SentenceTransformer, SparseEncoder
    from datasets import load_dataset, Dataset

    def demo_dense(num_samples: int = 5):
        print(" demo_dense ".center(100, "#"))
        dataset = {
            split: (
                load_dataset(
                    path="fancyzhx/ag_news",
                    cache_dir="./datasets",
                    split=f"{split}[:{num_samples}]",
                ).with_format("torch")
            )
            for split in ["train", "test"]
        }

        samples = {split: dataset[split]["text"] for split in dataset.keys()}
        for split, texts in samples.items():
            print(f"Split: {split}")
            for i, t in enumerate(texts):
                print(f"+ [{i}]: {t[0:150]}...")
            print("\n")

        encoder = SentenceTransformer(
            model_name_or_path="all-MiniLM-L6-v2",
            cache_folder="./models/pretrained/encoders/all-MiniLM-L6-v2",
        )
        embeddings: dict[str, Tensor] = {
            split: torch.from_numpy(encoder.encode(texts))
            for split, texts in samples.items()
        }

        simmer1 = CosineSimilarity()
        sim1 = simmer1(embeddings["train"], embeddings["test"])
        print(f"CosineSimilarity:\n{sim1}")

        simmer2 = InnerProductSimilarity()
        sim2 = simmer2(embeddings["train"], embeddings["test"])
        print(f"InnerProductSimilarity:\n{sim2}")

        # Will raise error since dense embeddings can have negative values
        # simmer3 = GeneralizedJaccardSimilarity()
        # sim3 = simmer3(embeddings["train"], embeddings["test"])
        # print(f"GeneralizedJaccardSimilarity:\n{sim3}")

        print(" demo_dense ".center(100, "#"))

    def demo_sparse(num_samples: int = 5):
        print(" demo_sparse ".center(100, "#"))
        dataset: dict[str, Dataset] = load_dataset(
            path="fancyzhx/ag_news",
            cache_dir="./datasets",
            split=[f"train:{num_samples}", f"test:{num_samples}"],
        ).with_format("torch")

        samples = {split: dataset[split]["text"] for split in dataset.keys()}
        for split, texts in samples.items():
            print(f"Split: {split}")
            for i, t in enumerate(texts):
                print(f"+ [{i}]: {t[0:150]}...")
            print("\n")

        encoder = SparseEncoder(
            model_name_or_path="naver/splade-cocondenser-ensembledistil",
            cache_folder="./models/pretrained/encoders/splade",
        )
        embeddings: dict[str, Tensor] = {
            split: encoder.encode(texts) for split, texts in samples.items()
        }

        simmer1 = CosineSimilarity()
        sim1 = simmer1(embeddings["train"], embeddings["test"])
        print(f"CosineSimilarity:\n{sim1}")

        simmer2 = InnerProductSimilarity()
        sim2 = simmer2(embeddings["train"], embeddings["test"])
        print(f"InnerProductSimilarity:\n{sim2}")

        simmer3 = GeneralizedJaccardSimilarity()
        sim3 = simmer3(embeddings["train"], embeddings["test"])
        print(f"GeneralizedJaccardSimilarity:\n{sim3}")

        simmer4 = JaccardSimilarity()
        sim4 = simmer4(embeddings["train"], embeddings["test"])
        print(f"JaccardSimilarity:\n{sim4}")

        print(" demo_sparse ".center(100, "#"))

    demo_dense()
    demo_sparse()
