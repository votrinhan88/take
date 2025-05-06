from numpy import ndarray
import torch
from torch import Tensor


def balanced_partition(total: int, num_parts: int) -> list[int]:
    base = torch.full((num_parts,), total // num_parts)
    residual = (torch.randperm(num_parts) < (total % num_parts)).long()
    parts = (base + residual).tolist()
    return parts


def ensure_tensor(x) -> Tensor:
    if isinstance(x, ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, Tensor):
        return x
    elif isinstance(x, list):
        return torch.tensor(x)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def get_batch_indices(num_samples: int, batch_size: int) -> tuple[int, int]:
    """Yields start and end indices for batch processing.

    Args:
    + `num_samples`: Total number of samples.
    + `batch_size`: Size of each batch.

    Yields: `(start_idx, end_idx)`
    """
    for i in range(0, num_samples, batch_size):
        yield i, min(i + batch_size, num_samples)


def set_difference(x1: Tensor, x2: Tensor) -> Tensor:
    """Compute the set difference of two set tensors, each containing unique values.

    Args:
    + `x1`: The first set tensor.
    + `x2`: The second set tensor.

    Returns: The set difference `x1 \\ x2`
    """
    combined = torch.cat(tensors=(x1, x2, x2), dim=0)
    uniques, counts = combined.unique(return_counts=True)
    x1_set_diff_x2 = uniques[counts == 1]
    return x1_set_diff_x2


def set_intersection(x1: Tensor, x2: Tensor) -> Tensor:
    """Compute the set intersection of two set tensors, each containing unique values.

    Args:
    + `x1`: The first set tensor.
    + `x2`: The second set tensor.

    Returns: The set intersection `x1 & x2`
    """
    combined = torch.cat(tensors=(x1, x2), dim=0)
    uniques, counts = combined.unique(return_counts=True)
    x1_intersect_x2 = uniques[counts > 1]
    return x1_intersect_x2
