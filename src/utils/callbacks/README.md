# utils/callbacks/

## Purpose
Defines callback utilities for experiment logging and progress reporting.

## Inputs / Outputs
- **Input:** Trainer events, metrics, format functions.
- **Output:** Console logs, formatted progress bars.

## Quick Reference
| Symbol         | Summary                                 |
|---------------|------------------------------------------|
| `PrintCallback` | Prints progress bar and metrics on event |

---

### PrintCallback
```python
class PrintCallback(L.Callback):
    def __init__(self, on_event: str = "train_epoch_end", format_dict: dict | None = None)
    @staticmethod
    def format_default(k: str, v: float) -> str
    def format_multicounter(self, k: str, v: np.ndarray) -> str
    def format_adaptive(self, k: str, v: float | Sequence[int | float]) -> str
    def register_format(self, format_dict: dict | None = None) -> Callable[[str, Any], str
```
Prints metrics and progress bar at specified trainer event.

---

## Dependencies
- Used by trainers in [`expts/`](../../expts/) and [`models/classifiers/`](../../models/classifiers/)

## Extension Points
- Add new callback classes for custom logging or event handling.
