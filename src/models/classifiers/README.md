# models/classifiers/

## Purpose
Defines classifier models for text classification tasks.

## Inputs / Outputs
- **Input:** Tensors (features, embeddings), config parameters.
- **Output:** Predicted class logits or probabilities.

## Quick Reference
| Symbol                  | Summary                                  |
|-------------------------|------------------------------------------|
| `LogisticRegression`    | Linear classifier for multiclass tasks   |
| `NaiveBayes`            | Probabilistic classifier                 |
| `SupportVectorMachine`  | Linear SVM for multiclass classification |
| `MultiClassHingeLoss`   | Hinge loss for SVM                      |
| `TextCNN`               | CNN-based text classifier                |
| `TextRNN`               | RNN-based text classifier                |
| `ClassifierTrainer`     | Training loop for classifiers            |

---

### LogisticRegression
```python
class LogisticRegression(L.LightningModule):
    def __init__(self, input_dim: int, num_classes: int = 2, return_logits: bool = False)
    def forward(self, input: Tensor) -> Tensor
    def extra_repr(self) -> str
```
Linear classifier. Outputs logits or probabilities.

### NaiveBayes
```python
class NaiveBayes(L.LightningModule):
    def __init__(self, input_dim: int, epsilon: float = 1e-8)
    def fit(self, train_loader: DataLoader, encoder: nn.Module | None = None)
```
Multinomial Naive Bayes classifier. Supports fitting from dataloader.

### SupportVectorMachine
```python
class SupportVectorMachine(L.LightningModule):
    def __init__(self, input_dim: int, num_classes: int = 2)
    def forward(self, input: Tensor) -> Tensor
    def extra_repr(self) -> str
```
Linear SVM classifier.

### MultiClassHingeLoss
```python
class MultiClassHingeLoss(L.LightningModule):
    def __init__(self, margin: float = 1.0, slack: float = 0.1, reduction: str = "mean")
    def forward(self, input: Tensor, target: Tensor) -> Tensor
```
Hinge loss for multiclass SVM.

### TextCNN
```python
class TextCNN(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 100, num_channels: int = 100, kernel_sizes: Sequence[int] = [3, 4, 5], dropout: float = 0.5, return_logits: bool = True)
    def forward(self, input: Tensor) -> Tensor
    def extra_repr(self) -> str
```
CNN-based text classifier.

### TextRNN
```python
class TextRNN(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 100, hidden_dim: int = 100, num_layers: int = 2, bidirectional: bool = False, p_dropout: float = 0.5, return_logits: bool = True)
    def forward(self, input: Tensor) -> Tensor
    def extra_repr(self) -> str
```
RNN-based text classifier.

### ClassifierTrainer
```python
class ClassifierTrainer(L.LightningModule):
    def __init__(self, classifier: nn.Module, encoder: Optional[nn.Module], loss_fn, optimizer_kw, num_classes: int | None = None, preembed: bool = False)
    def configure_optimizers(self)
    def training_step(self, batch, batch_idx) -> Tensor
```
Training loop for classifier models.

---

## Dependencies
- Uses modules from [`models/modules/`](../modules/)
- Consumed by experiment scripts in [`expts/`](../../expts/)

## Extension Points
- Add new classifier classes with standard LightningModule or nn.Module interface.
- Extend `ClassifierTrainer` for custom training logic.
