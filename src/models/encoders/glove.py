import numpy as np
from sentence_transformers.models import Pooling
from sentence_transformers.models.tokenizer import WhitespaceTokenizer
import torch
from torch import nn, Tensor


class GloVeEncoder(nn.Module):
    SUPPORTED_DIMS = [50, 100, 200, 300]
    SUPPORTED_EMBED_LEVELS = ["token", "sentence"]

    def __init__(self, embed_dim: int = 50, frozen: bool = True, embed_level: str = "sentence"):
        super().__init__()
        self.embed_dim = self._validate_args("embed_dim", embed_dim)
        self.frozen = frozen
        self.embed_level = self._validate_args("embed_level", embed_level)

        self.load_glove_embeddings()
        self.pool = Pooling(
            word_embedding_dimension=self.embed_dim,
            pooling_mode="mean",
            include_prompt=True,
        )

        if self.frozen:
            for param in self.parameters():
                param.requires_grad = False

    def _validate_args(self, arg: str, value):
        if arg == "embed_dim":
            if value not in self.SUPPORTED_DIMS:
                raise ValueError(f"Unsupported `embed_dim`: {value}. Supported: {self.SUPPORTED_DIMS}.")
            return value

        elif arg == "embed_level":
            if value not in self.SUPPORTED_EMBED_LEVELS:
                raise ValueError(
                    f"Unsupported `embed_level`: {value}. Supported: {self.SUPPORTED_EMBED_LEVELS}."
                )
            return value

    def load_glove_embeddings(self):
        path = f"./models/pretrained/encoders/glove/glove.6B.{self.embed_dim}d.txt"

        self.word_2_idx = {"PADDING_TOKEN": 0}
        self.vocab = ["PADDING_TOKEN"]
        embeddings = torch.zeros(size=[1000, self.embed_dim])
        count = 1

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                # Increase size of embeddings tensor if needed
                if count + 1 > embeddings.shape[0]:
                    print(f"{count}-th sample")
                    embeddings = torch.cat([embeddings, torch.zeros_like(embeddings)], dim=0)

                values = line.strip().split()
                word = values[0]
                embedding = torch.from_numpy(np.array(values[1:], dtype=np.float32))

                self.vocab.append(word)
                self.word_2_idx[word] = count
                embeddings[count] = embedding

                count += 1

        assert count == len(set(self.vocab)), "Vocabulary contains duplicates."
        assert count == len(self.word_2_idx), "word_2_idx mapping is incorrect."
        self.embeddings = embeddings[:count]
        print(f"Loaded GloVe embeddings of shape {self.embeddings.shape}.")
        self.tokenizer = WhitespaceTokenizer(vocab=self.vocab, do_lower_case=False)

    def tokenize(self, texts: list[str], as_tensor: bool = True) -> dict:
        input_ids = [self.tokenizer.tokenize(text) for text in texts]
        sentence_lengths = [len(tokens) for tokens in input_ids]
        attention_mask = [[1] * length for length in sentence_lengths]
        if not as_tensor:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "sentence_lengths": sentence_lengths,
            }

        # Pad and convert to tensors
        max_len = max(sentence_lengths) if sentence_lengths else 0
        input_ids_padded = [tokens + [0] * (max_len - len(tokens)) for tokens in input_ids]
        input_ids = torch.tensor(input_ids_padded, dtype=torch.int64)
        sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.int64)
        attention_mask = torch.where(
            input_ids != 0,
            torch.ones_like(input_ids),
            torch.zeros_like(input_ids),
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentence_lengths": sentence_lengths,
        }

    def embed_tokens(self, batched_tokens: dict[str, Tensor]) -> Tensor:
        token_emb = torch.zeros(
            size=[*batched_tokens["input_ids"].shape, self.embed_dim], dtype=torch.float32
        )
        for uid in batched_tokens["input_ids"].unique():
            indices = (batched_tokens["input_ids"] == uid).nonzero()
            token_emb[indices[:, 0], indices[:, 1]] = self.embeddings[uid]
        return token_emb

    def embed_sentence(self, batched_tokens: dict[str, Tensor], token_emb: Tensor) -> Tensor:
        sentence_emb = self.pool(({"token_embeddings": token_emb}))["sentence_embedding"]
        sentence_emb = sentence_emb * (
            batched_tokens["sentence_lengths"].max()
            / batched_tokens["sentence_lengths"].unsqueeze(dim=1)
        )
        return sentence_emb

    def forward(self, input: list[str], embed_level: str | None = None) -> Tensor:
        if embed_level is None:
            embed_level = self.embed_level

        batched_tokens = self.tokenize(input)
        token_emb = self.embed_tokens(batched_tokens)
        if embed_level == "token":
            return token_emb

        sentence_emb = self.embed_sentence(batched_tokens=batched_tokens, token_emb=token_emb)
        return sentence_emb

    @property
    def hparams(self) -> dict:
        return {"embed_dim": self.embed_dim, "frozen": self.frozen, "embed_level": self.embed_level}

    def extra_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self.hparams.items() if v is not None])


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    def test_glove_all_dims(dims=[50, 100, 200, 300]):
        for dim in dims:
            glove = GloVeEncoder(embed_dim=dim)

    def test_glove_vs_sbert():
        dataset = load_dataset(path="stanfordnlp/imdb", cache_dir="./datasets")
        sentences = [
            "This is an example sentence",
            "Each sentence is converted",
            "The quick brown fox jumps over the lazy dog",
        ] + dataset["train"].select_columns("text")[0:100]["text"]

        glove = GloVeEncoder(embed_dim=300)
        glove_sbert = SentenceTransformer(
            model_name_or_path="sentence-transformers/average_word_embeddings_glove.6B.300d",
            cache_folder="./models/pretrained/encoders/glove/GloVe-300d",
        )

        tokens = glove.tokenize(sentences)
        tokens_2 = glove_sbert.tokenize(sentences)
        for k in ["input_ids", "attention_mask", "sentence_lengths"]:
            assert (tokens[k] == tokens_2[k]).all(), (
                f"Tokens do not match for key {k}: {tokens} != {tokens_2}",
            )

        emb: Tensor = glove(sentences)
        emb_2: Tensor = torch.from_numpy(glove_sbert.encode(sentences))
        print(emb[0:5, 0:10])
        print(emb_2[0:5, 0:10])
        print(f"Exact equal: {(emb == emb_2).sum()}/{np.prod(emb.shape)}")
        print(f"Is close: {emb.isclose(emb_2).sum()}/{np.prod(emb.shape)}")
        print(f"Difference max: {(emb - emb_2).abs().max()}")
        print(f"Difference mean: {(emb - emb_2).abs().mean()}")
        print(f"Mean abs: {emb.abs().mean()}")

    # test_glove_all_dims()
    test_glove_vs_sbert()
