import torch


def compute_2d_meshgrid(model, bbox: list[int], grid_size: int = 100) -> dict:
    """Compute 2D meshgrid from the prediction of a model.

    Args:
      `model`: Model to give prediction on 2D inputs of shape `[B, 2]`.
      `bbox`: Coordinates of meshgrid in the form of `[x_min, y_min, x_max, y_max]`.
      `grid_size`: Size of the grid per each dimension. Defaults to `100`.

    Returns:
      A dict containing `x1, x2, pred`.
    """
    x1_min, x2_min, x1_max, x2_max = bbox

    plot_x1 = torch.linspace(start=x1_min, end=x1_max, steps=grid_size)
    plot_x2 = torch.linspace(start=x2_min, end=x2_max, steps=grid_size)

    step_x1 = plot_x1[1] - plot_x1[0]
    step_x2 = plot_x2[1] - plot_x2[0]

    x1, x2 = torch.meshgrid([plot_x1 - step_x1 / 2, plot_x2 - step_x2 / 2])
    x = torch.cat(
        tensors=[x1.flatten().unsqueeze(dim=1), x2.flatten().unsqueeze(dim=1)],
        dim=1,
    )
    pred = model(x).reshape([plot_x1.shape[0], plot_x2.shape[0]])
    return {
        "x1": x1,
        "x2": x2,
        "pred": pred,
    }
