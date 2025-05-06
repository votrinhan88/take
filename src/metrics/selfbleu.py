from concurrent.futures import ThreadPoolExecutor

import torch
from torch import Tensor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class SelfBLEU:
    """SelfBLEU computes the 'overlap' of a list of sentences. Self-BLEU ranges between [0, 1],
    where low scores mean diversity and high scores mean more similar outputs or mode collapse.

    Args:
    + `tokenizer`: tokenizer.
    + `n`: The n-gram order (e.g., 4 for Self-BLEU-4). Defaults to `4`.
    + `weights`: Weights for n-gram precisions. Defaults to uniform weights.
    """

    def __init__(self, tokenizer=None, n: int = 4, weights: str | list[float] = "uniform"):
        self.tokenizer = tokenizer
        self.n = n
        self.weights = weights
        self._validate_args()

    def _validate_args(self):
        assert isinstance(self.n, int) and self.n > 0, "n must be positive."
        if isinstance(self.weights, str):
            assert self.weights == "uniform", "Only 'uniform' weights string is supported."
            self.weights = [1.0 / self.n] * self.n
        elif isinstance(self.weights, list):
            assert len(self.weights) == self.n, "weights length must match n."
            assert abs(sum(self.weights) - 1.0) < 1e-6, "weights must sum to 1."

    def _bleu(self, index: int, corpus: list[list[int]]) -> float:
        hypothesis = corpus[index]
        references = corpus[:index] + corpus[index + 1 :]
        smoothing = SmoothingFunction().method1
        return sentence_bleu(
            references, hypothesis, weights=self.weights, smoothing_function=smoothing
        )

    def tokenize(self, texts: list[str] | None) -> list[list[int]]:
        if texts is None:
            raise ValueError("Either tokens or texts must be provided.")
        if len(texts) < 2:
            raise ValueError("Input texts list must contain at least two texts.")

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when texts are used.")
        tokens = self.tokenizer(texts)["input_ids"]
        return tokens

    def __call__(
        self,
        tokens: list[list[int]] | None = None,
        texts: list[str] | None = None,
        workers: int | None = None,
    ) -> dict:
        if tokens is None:
            tokens = self.tokenize(texts)

        indices = range(len(tokens))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scores = torch.tensor(
                data=list(executor.map(lambda i: self._bleu(i, tokens), indices)),
                dtype=torch.float32,
            )

        scores = {
            "corpus": scores.mean(dim=0).item(),
            "sample": scores.tolist(),
        }
        return scores


if __name__ == "__main__":
    import os
    import sys
    from transformers import AutoTokenizer

    repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
    assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="gpt2",
        cache_dir="./pretrained/",
    )
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "A journey of a thousand miles begins with a single step.",
    ]

    tokens = tokenizer(texts)["input_ids"]
    selfbleu = SelfBLEU(n=4)
    result = selfbleu(tokens=tokens)
    print("Self-BLEU:", result)
