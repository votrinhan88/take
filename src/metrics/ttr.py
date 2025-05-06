import torch
from torch import Tensor


class TypeTokenRatio:
    """Type-Token Ratio = no. of unique tokens / total no. of tokens.
    Can be sensitive to text length, so we evaluate at corpus-level and sample-level.

    Args:
    + `tokenizer`: tokenizer.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, texts: str | list[str]) -> dict[str, float | Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        token_counts = 0
        ttr_corpus = torch.zeros(size=[self.vocab_size], dtype=torch.bool, device=self.device)
        ttr_samples = torch.zeros(size=[len(texts)], device=self.device)
        for i, text in enumerate(texts):
            tokens: list[int] = self.tokenizer(text=text)["input_ids"]
            tokens_unique = set(tokens)
            ttr_samples[i] = len(tokens_unique) / len(tokens)
            ttr_corpus[list(tokens_unique)] = True
            token_counts += len(tokens)

        ttr_corpus = ttr_corpus.sum(dim=0) / token_counts
        ttr = {
            "corpus": ttr_corpus.cpu(),
            "sample": ttr_samples.cpu(),
        }
        return ttr


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

    ttr = TypeTokenRatio(tokenizer=tokenizer)
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
    ]
    result = ttr(texts=texts)
    print("Type-Token Ratio:", result)
