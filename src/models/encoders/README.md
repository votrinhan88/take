# models/encoders/

## Purpose
Defines encoder models for transforming text into embeddings.

## Inputs / Outputs
- **Input:** Raw text (list[str]), config parameters.
- **Output:** Embedding tensors (shape: [batch, dim]).

## Quick Reference
| Symbol                         | Summary                                |
|--------------------------------|----------------------------------------|
| [`GloVeEncoder`](glove.py:8)   | GloVe word embedding encoder           |
| [`MiniLMWrapper`](minilm.py:5) | Wrapper for MiniLM transformer encoder |
| [`Tfidf`](tdidf.py:9)          | TF-IDF vectorizer encoder              |

---

### GloVeEncoder
```python
class GloVeEncoder(nn.Module):
    SUPPORTED_DIMS = [50, 100, 200, 300]
    def __init__(self, embed_dim: int = 50, frozen: bool = True, embed_level: str = "sentence")
    def forward(self, input: list[str]) -> Tensor
    def tokenize(self, texts: Sequence[str]) -> dict[str, Tensor | Sequence[int]]
```
Loads GloVe embeddings and pools to sentence level. See [`glove-download.sh`](glove-download.sh:1) for downloading pretrained vectors.

### Tfidf
```python
class Tfidf(L.LightningModule):
    def __init__(self, embed_dim: int | None = None, sparse: bool = False)
    def fit(self, train_loader: DataLoader|list[str])
    def forward(self, input: list[str]|str) -> Tensor
```
TF-IDF vectorizer encoder using scikit-learn. Supports sparse/dense output.

### MiniLMWrapper
```python
class MiniLMWrapper(nn.Module):
    def __init__(self, model)
    def forward(self, *args, **kwargs) -> Tensor
```
Wrapper for MiniLM transformer encoder. Converts HuggingFace outputs to PyTorch tensors.

---

## Metadata Utility
- [`EncoderMetadata`](metadata.py:1): Validates supported encoder models and provides preset configurations.

## Module Imports
- [`__init__.py`](__init__.py:1): Imports all main encoder classes for easy access.

## Extension Points
- Add new encoder classes by subclassing `nn.Module`.
- Extend existing encoders for custom pooling or tokenization.
