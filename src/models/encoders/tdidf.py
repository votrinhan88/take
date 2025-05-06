import lightning as L
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


class Tfidf(L.LightningModule):
    """Tfidf encoder using scikit-learn's TfidfVectorizer.

    Args:
    + `embed_dim`: Maximum number of features for the vectorizer. Defaults to `None`.
    + `sparse`: Flag to return sparse tensors. Defaults to `False`.

    Attributes:
    + `vectorizer`: The underlying TfidfVectorizer instance.
    + `vocab`: List of feature names in vocabulary.
    + `vocab_size`: Size of vocabulary.

    Methods:
    + `fit`: Fit the vectorizer to a corpus.
    + `forward`: Transform input text to TF-IDF tensor.
    """

    def __init__(self, embed_dim: int | None = None, sparse: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.sparse = sparse
        
        hparams = {"embed_dim": embed_dim, "sparse": sparse}
        self.save_hyperparameters(hparams)

        self.vectorizer = TfidfVectorizer(max_features=embed_dim)

    def fit(self, train_loader: DataLoader | list[str]):
        corpus = []
        for batch in train_loader:
            if isinstance(batch, str):
                corpus.append(batch)
            else:
                corpus.extend(batch["text"])

        embeddings = self.vectorizer.fit_transform(corpus)
        self.embeddings = self.convert_scipy_csr_to_torch_sparse_coo(embeddings)
        self.vocab = self.vectorizer.get_feature_names_out()

    def forward(self, input: list[str] | str) -> Tensor:
        if isinstance(input, str):
            input = [input]
        e = self.vectorizer.transform(input)
        if not self.sparse:
            e = torch.from_numpy(e.todense())
        else:
            e = self.convert_scipy_csr_to_torch_sparse_coo(e)
        e = e.to(device=self.device, dtype=torch.float32)
        return e

    def extra_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self.hparams.items() if v is not None])

    @property
    def vocab_size(self) -> int:
        if self.vocab is None:
            raise ValueError("Vocabulary is not set. Call fit() first.")

        return len(self.vocab)

    @staticmethod
    def convert_scipy_csr_to_torch_sparse_coo(input: csr_matrix) -> Tensor:
        coo = input.tocoo()
        indices = torch.stack(
            tensors=[
                torch.tensor(coo.row, dtype=torch.long),
                torch.tensor(coo.col, dtype=torch.long),
            ],
            dim=0,
        )
        values = torch.tensor(coo.data, dtype=torch.float32)

        x = torch.sparse_coo_tensor(indices=indices, values=values, size=coo.shape)
        x = x.coalesce()
        return x


if __name__ == "__main__":

    def test_1():
        print(" Test 1: Simple corpus ".center(100, "="))
        corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]

        tfidf = Tfidf(embed_dim=1000, sparse=False)
        tfidf.fit(train_loader=corpus)
        print(f"Vocabulary: {tfidf.vocab}")
        print(f"TF-IDF embeddings:\n{tfidf.embeddings}")

        x = ["This is a new document."]
        e = tfidf(x)
        print(f"{x}: {e}")

    def test_2():
        print(" Test 2: IMDB ".center(100, "="))
        # Change path
        import os, sys

        repo_path = os.path.abspath(os.path.join(__file__, "../.."))
        if sys.path[0] != repo_path:
            sys.path.insert(0, repo_path)

        from datasets import load_dataset, DatasetDict
        dataset = load_dataset(
            path="stanfordnlp/imdb", cache_dir="./datasets", split="train[:100]"
        )
        corpus = dataset["text"][:]

        tfidf = Tfidf(embed_dim=1000, sparse=True)
        tfidf.fit(train_loader=corpus)
        print(f"Vocabulary: {tfidf.vocab}")
        print(f"Embeddings:\n{tfidf.embeddings}")

        x = ["This is a new document."]
        e = tfidf(x)
        print(f"{x}: {e}")

    test_1()
    test_2()
