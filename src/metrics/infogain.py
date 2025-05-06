from typing import Callable, Optional
import torch
from torch import nn, Tensor


class InformationGain(nn.Module):
    def __init__(self, sim_fn: Callable[[Tensor], Tensor]):
        super().__init__()
        self.sim_fn = sim_fn

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class DeterminantalPointProcess(InformationGain):
    """
    Determinantal Point Process.
    Good for dense embeddings.

    Args:
      `sim_fn`: Similarity function.
      `embeddings`: Embeddings of the current dataset. Defaults to `None`.
      `epsilon`: Small positive float for numerical stability. Defaults to `1e-6`.
    """

    def __init__(
        self,
        embeddings: Tensor,
        sim_fn: Callable[[Tensor], Tensor],
        epsilon: float = 1e-6,
    ):
        super().__init__(sim_fn=sim_fn)
        self.embeddings = embeddings
        self.epsilon = epsilon

        if self.embeddings is not None:
            self.embeddings = self.embeddings.to(device=self.device)
            K_DD = self.sim_fn(self.embeddings, self.embeddings)
            K_DD = self.stabilize_matrix(mat=K_DD)
            self.L = torch.linalg.cholesky(K_DD).to(device=self.device)

    def forward(self, input: Tensor) -> float:
        """
        Computes per-sample information gain for each independent x_i in the batch X:
            Delta_i = logdet(K_{D cup {x_i}}) - logdet(K_D)

        Args:
            K_DX: (n, m) cross-similarity between D and new X
            K_XX: (m, m) similarity of new X (only diagonal used)

        Compute conditional variances:
            s_i = K_xx - ||W_i||^2
        Information gain per sample:
            Delta_i = log(s_i)
        """
        input = input.to(device=self.device)
        K_DX: Tensor = self.sim_fn(self.embeddings, input)
        K_XX: Tensor = self.sim_fn(input, input)

        W: Tensor = torch.linalg.solve_triangular(self.L, K_DX, upper=False)
        # Per-sample conditional variances
        S_vec = K_XX.diag() - (W**2).sum(dim=0)  # [M]
        S_vec = S_vec.clamp(min=self.epsilon)  # [M]
        logdet_S = S_vec.log()  # [M]
        return logdet_S

    def update(self, input: Tensor):
        """
        Update the similarity matrix K_DD and Cholesky decomposition L of the current
        dataset in an incremental way (for efficiency) with the new samples K_DX and
        K_XX.
        """
        input = input.to(device=self.device)

        K_DX: Tensor = self.sim_fn(self.embeddings, input)
        K_XX: Tensor = self.sim_fn(input, input)

        W: Tensor = torch.linalg.solve_triangular(self.L, K_DX, upper=False)
        schur = K_XX - W.T @ W
        schur = self.stabilize_matrix(mat=schur)
        L_schur: Tensor = torch.linalg.cholesky(schur)

        n = self.L.shape[0]
        n_new = n + K_XX.shape[0]
        L_new = torch.zeros(
            size=[n_new, n_new],
            dtype=self.L.dtype,
            device=self.L.device,
        )
        L_new[0:n, 0:n] = self.L
        L_new[n:, 0:n] = W.T
        L_new[n:, n:] = L_schur
        self.L = L_new
        self.embeddings = torch.cat([self.embeddings, input], dim=0)

    def forward_batch_gain(self, K_DX: Tensor, K_XX: Tensor) -> Tensor:
        """
        Computes the log-determinant difference (information gain) of new samples
        (K_DX and K_XX) given the current dataset K_DD - using the Schur-complement and
        its Cholesky decomposition L.

        Using the Schur-complement of D in (D cup X), the log-determinant difference
        when adding new points X to existing set D is:
            Delta   = log det(K_{D cup X}) - log det(K_D)
                    = log det(K_XX - K_XD K_DD^{-1} K_DX)

        We want to compute Delta, but taking the inverse K_DD^{-1} is numerically
        unstable and expensive: O(n^3). Instead, we use the Cholesky decomposition of
        K_DD = L L^T, where L is lower-triangular. The inverse can be expressed as:
            K_DD = L L^T --> K_DD^{-1} = (L^T)^{-1} L^{-1}

        Revisiting the Schur-complement:
            S   = K_XX - K_XD K_DD^{-1} K_DX
                = K_XX - K_DX^T (L^T)^{-1} L^{-1} K_DX
                = K_XX - (L^{-1} K_DX)^T (L^{-1} K_DX)
                = K_XX - W^T W, where W = L^{-1} K_DX
        We define W = L^{-1} K_DX, which can be computed efficiently by solving the
        triangular system L W = K_DX.

        Finally, we compute the log-determinant of S. We can compute log det(S)
        directly. However, doing that requires multiplies all eigenvalues or pivots,
        which can be ill-conditioned for matrices that are large, nearly singular, or
        have widely varying scales.

        Thus, we compute log det(S) via its Cholesky decomposition S = L_S L_S^T.
            det(S) = det(L_S)^2
        where L_S is lower-triangular, so det(L_S) is the product of its diagonal
        entries:
            det(L_S) = prod(diag(L_S)).
        Taking the log:
            log det(S) = 2 * sum(log(diag(L_S)))
        """
        W: Tensor = torch.linalg.solve_triangular(self.L, K_DX, upper=False)
        schur = K_XX - W.T @ W
        schur = self.stabilize_matrix(mat=schur)
        L_schur: Tensor = torch.linalg.cholesky(schur)
        logdet_S = 2 * L_schur.diag().log().sum(dim=0)
        return logdet_S

    def stabilize_matrix(self, mat: Tensor) -> Tensor:
        mat = mat.to(device=self.device)
        # Ensure symmetry
        mat = 0.5 * (mat + mat.T)
        # Add epsilon to diagonal
        epsilon_eye = self.epsilon * torch.eye(
            n=mat.shape[0], device=mat.device, dtype=mat.dtype
        )
        mat = mat + epsilon_eye
        return mat


class AverageSimilarityGain(InformationGain):
    def __init__(
        self,
        embeddings: Tensor,
        sim_fn: Callable[[Tensor], Tensor],
    ):
        super().__init__(sim_fn=sim_fn)
        self.embeddings = embeddings

        if self.embeddings is not None:
            self.embeddings = self.embeddings.to(device=self.device)

    def forward(self, input: Tensor) -> Tensor:
        sim_XD: Tensor = self.sim_fn(self.embeddings, input).T
        gain = -sim_XD.mean(dim=1).log()
        return gain


class NearestNeighborDissimilarity(InformationGain):
    def __init__(
        self,
        embeddings: Tensor,
        sim_fn: Callable[[Tensor], Tensor],
        n_neighbors: int = 5,
    ):
        super().__init__(sim_fn=sim_fn)
        self.embeddings = embeddings
        self.n_neighbors = n_neighbors

        if self.embeddings is not None:
            self.embeddings = self.embeddings.to(device=self.device)

    def forward(self, input: Tensor) -> Tensor:
        sim_XD: Tensor = self.sim_fn(self.embeddings, input).T
        sim_XX: Tensor = self.sim_fn(input, input)

        if self.n_neighbors == 1:
            sim_XD_nn = sim_XD.max(dim=1).values
        else:
            sim_XD_nn = sim_XD.sort(dim=1, descending=True).values[
                :, 0 : self.n_neighbors
            ]
            sim_XD_nn = sim_XD_nn.mean(dim=1)

        gain = sim_XX.diag() - sim_XD_nn
        return gain


if __name__ == "__main__":
    # Change path
    import os, sys

    repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
    assert (
        os.path.basename(repo_path) == "textdd"
    ), "Wrong parent folder. Please change to 'textdd'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    from sentence_transformers import SentenceTransformer, SparseEncoder
    from datasets import load_dataset

    from models.diversity.similarity import (
        CosineSimilarity,
        ExponentialCosineSimilarity,
        NormalizedCosineSimilarity,
        GeneralizedJaccardSimilarity,
        InnerProductSimilarity,
        JaccardSimilarity,
        RBFKernelSimilarity,
    )

    def demo_gain_dense():
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = {
            split: (
                load_dataset(
                    path="fancyzhx/ag_news",
                    cache_dir="./datasets",
                    split=f"{split}[:1%]",
                )
                # .map(config["dataset"]["map"])
                .with_format("torch")
            )
            for split in ["train", "test"]
        }
        samples = {
            "old": dataset["train"]["text"],
            "new": dataset["test"]["text"],
        }

        encoder = SentenceTransformer(
            model_name_or_path="all-MiniLM-L6-v2",
            cache_folder="./pretrained/encoders/all-MiniLM-L6-v2",
        )
        embeddings = {
            k: torch.from_numpy(encoder.encode(v)).to(DEVICE)
            for k, v in samples.items()
        }

        # sim_fn = RBFKernelSimilarity(embeddings=embeddings["old"])
        sim_fn = ExponentialCosineSimilarity()

        nndis1 = NearestNeighborDissimilarity(
            embeddings=embeddings["old"],
            sim_fn=sim_fn,
            n_neighbors=1,
        )
        gain_nndis1: Tensor = nndis1(embeddings["new"])
        nndis5 = NearestNeighborDissimilarity(
            embeddings=embeddings["old"],
            sim_fn=sim_fn,
            n_neighbors=5,
        )
        gain_nndis5: Tensor = nndis5(embeddings["new"])
        ags = AverageSimilarityGain(embeddings=embeddings["old"], sim_fn=sim_fn)
        gain_ags: Tensor = ags(embeddings["new"])
        ddp = DeterminantalPointProcess(embeddings=embeddings["old"], sim_fn=sim_fn)
        gain_ddp: Tensor = ddp(embeddings["new"])

        for gain, name in zip(
            [gain_nndis1, gain_nndis5, gain_ags, gain_ddp],
            ["NNDis-1", "NNDis-5", "ASG", "DDP"],
        ):
            print(f" {name} ".center(100, "#"))
            print("Least similar:")
            for i in gain.argsort(descending=True)[0:5]:
                print(
                    f"[{i}, gain={gain[i]:.4f}]: {samples['new'][i.item()][0:150]}..."
                )
            print()
            print("Most similar:")
            for i in gain.argsort(descending=False)[0:5]:
                print(
                    f"[{i}, gain={gain[i]:.4f}]: {samples['new'][i.item()][0:150]}..."
                )
            print()

    def demo_gain_sparse():
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = {
            split: (
                load_dataset(
                    path="fancyzhx/ag_news",
                    cache_dir="./datasets",
                    split=f"{split}[:1%]",
                )
                # .map(config["dataset"]["map"])
                .with_format("torch")
            )
            for split in ["train", "test"]
        }
        samples = {
            "old": dataset["train"]["text"],
            "new": dataset["test"]["text"],
        }

        encoder = SparseEncoder(
            model_name_or_path="naver/splade-cocondenser-ensembledistil",
            cache_folder="./pretrained/encoders/splade",
        )
        embeddings = {k: encoder.encode(v).to(DEVICE) for k, v in samples.items()}

        sim_fn = GeneralizedJaccardSimilarity()

        nndis1 = NearestNeighborDissimilarity(
            embeddings=embeddings["old"],
            sim_fn=sim_fn,
            n_neighbors=1,
        )
        gain_nndis1: Tensor = nndis1(embeddings["new"])
        nndis5 = NearestNeighborDissimilarity(
            embeddings=embeddings["old"],
            sim_fn=sim_fn,
            n_neighbors=5,
        )
        gain_nndis5: Tensor = nndis5(embeddings["new"])
        ags = AverageSimilarityGain(embeddings=embeddings["old"], sim_fn=sim_fn)
        gain_ags: Tensor = ags(embeddings["new"])
        ddp = DeterminantalPointProcess(embeddings=embeddings["old"], sim_fn=sim_fn)
        gain_ddp: Tensor = ddp(embeddings["new"])

        for gain, name in zip(
            [gain_nndis1, gain_nndis5, gain_ags, gain_ddp],
            ["NNDis-1", "NNDis-5", "ASG", "DDP"],
        ):
            print(f" {name} ".center(100, "#"))
            print("Least similar:")
            for i in gain.argsort(descending=True)[0:5]:
                print(
                    f"[{i}, gain={gain[i]:.4f}]: {samples['new'][i.item()][0:150]}..."
                )
            print()
            print("Most similar:")
            for i in gain.argsort(descending=False)[0:5]:
                print(
                    f"[{i}, gain={gain[i]:.4f}]: {samples['new'][i.item()][0:150]}..."
                )
            print()

    demo_gain_sparse()
