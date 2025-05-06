import torch
from torch import Tensor


class DistinctN:
    """Distinct-n measures lexical diversity by computing the ratio of unique n-grams to total
    n-grams.
    Distinct-n = no. of unique n-grams / total no. of n-grams.

    Can be sensitive to text length, so we evaluate at corpus-level and sample-level.
    Distinct-1 is equivalent to Type-Token Ratio (TTR).

    Args:
    + `tokenizer`: tokenizer.
    + `n`: n-gram size. Defaults to `2`.
    """

    def __init__(self, tokenizer=None, n: int = 2):
        self.tokenizer = tokenizer
        self.n = n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize(self, texts: list[str] | str | None) -> list[list[int]]:
        if texts is None:
            raise ValueError("Either tokens or texts must be provided.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when texts are used.")

        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts)["input_ids"]
        return tokens

    def get_ngrams(self, tokens: list[int]) -> list[tuple]:
        if len(tokens) < self.n:
            return []
        return [tuple(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)]

    def __call__(
        self,
        tokens: list[list[int]] | None = None,
        texts: str | list[str] | None = None,
    ) -> dict[str, float | list[float]]:
        if tokens is None:
            tokens = self.tokenize(texts)

        ngram_counts = 0
        distinct_corpus = set()
        distinct_samples = torch.zeros(size=[len(tokens)], device=self.device)

        for i, tk in enumerate(tokens):
            ngrams = self.get_ngrams(tk)
            if len(ngrams) == 0:
                distinct_samples[i] = 0.0
                continue

            ngrams_unique = set(ngrams)
            distinct_samples[i] = len(ngrams_unique) / len(ngrams)
            distinct_corpus.update(ngrams_unique)
            ngram_counts += len(ngrams)

        distinct_corpus_ratio = len(distinct_corpus) / ngram_counts if ngram_counts > 0 else 0.0
        distinctn = {
            "corpus": distinct_corpus_ratio,
            "sample": distinct_samples.cpu().tolist(),
        }
        return distinctn


if __name__ == "__main__":
    import os
    import sys

    repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
    assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    from transformers import AutoTokenizer

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

    for n in [1, 2, 3]:
        distinctn = DistinctN(n=n)
        result = distinctn(tokens=tokens)
        print(f"Distinct-{n}:", result)
