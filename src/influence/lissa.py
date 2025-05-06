import warnings

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import tqdm.auto as tqdm

from .batch_unpack import BatchUnpacker


def compute_val_grad(
    model: nn.Module,
    params_inf: list[nn.Parameter] | None,
    valloader: DataLoader,
    loss_fn: nn.Module,
    batch_unpacker: BatchUnpacker,
    device: torch.device,
) -> Tensor:
    r"""Compute the gradient of the validation loss with respect to the (selected) model
    parameters.

    g_val := \nabla_{\theta} L(D_val) = E_{z \in D_val}[ \nabla_{\theta} L(z) ]
    """
    # Backpropagate
    train_state = model.training
    model.eval()
    model.zero_grad()
    num_batches = 0
    for batch in valloader:
        inputs, targets = batch_unpacker(batch)
        inputs = inputs.to(device=device)
        targets = targets.to(device=device)

        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        num_batches += 1
    model.train(mode=train_state)

    # Collect gradients
    if params_inf is None:
        params_inf: list[nn.Parameter] = model.parameters()

    grad_val = []
    for p in params_inf:
        if p.grad is not None:
            grad_val.append(p.grad.clone().detach().flatten() / num_batches)
        else:
            grad_val.append(torch.zeros_like(p).flatten())
    grad_val = torch.cat(tensors=grad_val, dim=0)

    model.zero_grad()
    return grad_val


def compute_inverse_hvp(
    model: nn.Module,
    params_inf: list[nn.Parameter],
    grad_val: Tensor,
    loss_fn: nn.Module,
    trainloader: DataLoader,
    batch_unpacker: BatchUnpacker,
    lissa_depth: int,
    scale: float,
    damping: float,
    device: torch.device,
) -> Tensor:
    """Approximate h_val = H^{-1} . g_val) via iterative update."""
    # Init: h_0 = grad_val
    h = grad_val.clone().detach()
    # Loop: h_t = g_val + h_{t-1} - scale * H . h_{t-1}
    num_steps = min(len(trainloader), lissa_depth)
    if num_steps < lissa_depth:
        warnings.warn(f"Number of steps ({num_steps}) < lissa_depth ({lissa_depth}).")
    pbar = tqdm.tqdm(range(num_steps), desc="Compute IVHP")
    for i in pbar:
        batch = next(iter(trainloader))
        inputs, targets = batch_unpacker(batch)
        inputs = inputs.to(device=device)
        targets = targets.to(device=device)
        Hh_hvp = compute_hvp(
            inputs=inputs,
            targets=targets,
            model=model,
            params_inf=params_inf,
            loss_fn=loss_fn,
            h=h,
            damping=damping,
        )
        h = grad_val + h - scale * Hh_hvp
    # Final: h_val = scale * h_T
    h = scale * h.detach()
    return h


def compute_hvp(
    inputs: Tensor,
    targets: Tensor,
    model: nn.Module,
    params_inf: list[nn.Parameter],
    loss_fn: nn.Module,
    h: Tensor,
    damping: float,
) -> Tensor:
    r"""Use the Pearlmutter Trick (or the "Double Backprop" trick) to compute the
    Hessian-Vector Product H.h directly.

    The Hessian is the derivative of the gradient. By definition:
        H . h = \nabla ((\nabla L)^T h)
    This means the product of a Hessian and a vector is simply the gradient of a
    scalar. Reminder: gradient and Hessian are taken w.r.t the model parameters
    \theta.

    We can first compute the gradient of the loss w.r.t the model parameters:
        g = \nabla L
    Then, we dot-product it with the vector h:
        grad_h = g^T h = (\nabla L)^T h
    Finally, we compute the gradient of this scalar w.r.t the model parameters:
        grad2_h = \nabla grad_h = \nabla ((\nabla L)^T h) = H . h
    We additionally add the damping term, corresponding to adding a small value
    along the diagional of the Hessian (H + \lambda I) h:
        grad2_h += \lambda h
    """
    # 1. Compute loss
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    # 2. Compute gradients \nabla L
    grad = torch.autograd.grad(
        outputs=loss,
        inputs=params_inf,
        create_graph=True,
        retain_graph=True,
    )
    grad = torch.cat(tensors=[g.flatten() for g in grad], dim=0)

    # 3. Compute inner product (\nabla L)^T h
    grad_h = grad.dot(h)

    # 4. Compute gradient of inner product \nabla (g_h)
    grad2_h = torch.autograd.grad(outputs=grad_h, inputs=params_inf)
    grad2_h = torch.cat(tensors=[g.flatten() for g in grad2_h], dim=0)

    return grad2_h + damping * h


class LiSSAInfluenceScorer(nn.Module):
    r"""Computes Influence Functions using the LiSSA (Linear Time Stochastic
    Second-Order Algorithm) approximation.

    Args:
    + `model`: The model to compute influence for.
    + `params_inf`: List of parameters to consider for influence computation. If None,
        uses all parameters requiring grad.
    + `loss_fn`: Loss function.
    + `damping`: Damping factor to add to the Hessian (H + \lambda I) to ensure
        invertibility and numerical stability. Default: 1e-3.
    + `lissa_depth`: Number of iterations (recursion depth) for the LiSSA approximation.
        Higher depth improves accuracy but increases computation time. Default: 10.
    + `scale`: Scaling factor for the Neumann series. Must be small enough such that
        the spectral norm of (I - scale * H) is < 1 for convergence. Default: 0.001.

    ---

    The influence of a training sample z on the validation set (i.e., loss on the
    validation set):
        I(z, D_val) = - \nabla L(D_val)^T  .  H^{-1}  .  \nabla L(z)
                    = - \nabla L(z)^T      .  H^{-1}  .  \nabla L(D_val) (commutative)
    where the gradient g and Hessian H are taken w.r.t the model parameters \theta:
        + g_val := \nabla L(D_val) = \nabla_{\theta} L(D_val) is the average gradient of
            validation loss.
        + H^{-1} := H^{-1}_{\theta} is the inverse Hessian of the training loss.
        + g(z) := \nabla L(z) = \nabla_{\theta} L(z) is the gradient of the training loss at
            the training sample z.

    We factorize the influence into:
        I(z, D_val) = - g(z)^T  .  h_val
    where
        h_val := H^{-1}  .  g_val,
    which represents a fixed quantity only dependent on the validation set D_val, and
    streamline the influence computation for any z.

    However in practice, inverting H is not possible, so we approximate it with an
    iterative update:
        H \approx scale * \sum_{t=0}^{T} (I - scale * H)^t
        where T is the recursion depth (`lissa_depth`).
    Plugging this into h_val:
        h_val/scale \approx \sum_{t=0}^{T} (I - scale * H)^t  .  g_val
    Let M = (I - scale * H), this is equivalent to a Neumann series. We temporarily
    remove the scale factor (will multiply back at the end):
        h_T \approx g_val  .  \sum_{t=0}^{T} M^t
                  = g_val  .  (1 + M + M^2 + ... + M^T)
    To avoid computing to high powers, we factorize to recursion:
        h_t = g_val  +  (I - scale * H) . h_{t-1}
            = g_val  +  h_{t-1}  -  scale * H . h_{t-1}
    Where HVP := H . h_{t-1} is the Hessian-Vector Product, and h is initialized to
    h0 = g_val.
    We finally multiply back the scale factor:
        h_val = scale * h_T
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        params_inf: list[nn.Parameter] | None = None,
        batch_unpacker: BatchUnpacker = BatchUnpacker(),
        damping: float = 1e-3,
        lissa_depth: int = 10,
        scale: float = 0.001,
    ):
        super().__init__()
        self.model = model
        self.params_inf = params_inf
        self.loss_fn = loss_fn
        self.batch_unpacker = batch_unpacker
        self.damping = damping
        self.lissa_depth = lissa_depth
        self.scale = scale

        if self.params_inf is None:
            self.params_inf: list[nn.Parameter] = [p for p in model.parameters()]

        self.relu = nn.ReLU()
        self.device = next(self.model.parameters()).device

    def fit(self, trainloader: DataLoader, valloader: DataLoader):
        # Compute validation gradients g_val = \nabla L(D_val)
        self.grad_val = compute_val_grad(
            model=self.model,
            params_inf=self.params_inf,
            valloader=valloader,
            loss_fn=self.loss_fn,
            batch_unpacker=self.batch_unpacker,
            device=self.device,
        ).to(self.device)
        # Compute h_val = H^{-1} . g_val via IVHP
        self.h_val = compute_inverse_hvp(
            model=self.model,
            params_inf=self.params_inf,
            grad_val=self.grad_val,
            loss_fn=self.loss_fn,
            trainloader=trainloader,
            batch_unpacker=self.batch_unpacker,
            lissa_depth=self.lissa_depth,
            scale=self.scale,
            damping=self.damping,
            device=self.device,
        ).to(self.device)

    def forward(self, batch) -> Tensor:
        inputs, targets = self.batch_unpacker(batch)
        inputs = inputs.to(device=self.device)
        targets = targets.to(device=self.device)
        batch_size = inputs.shape[0]

        train_mode = self.model.training
        self.model.eval()

        influences = []
        for i in range(batch_size):
            x = inputs[[i]]
            y = targets[[i]]

            self.model.zero_grad()
            out = self.model(x)
            loss = self.loss_fn(out, y)

            grad = torch.autograd.grad(outputs=loss, inputs=self.params_inf)
            grad = torch.cat(tensors=[p.flatten() for p in grad], dim=0)
            # I(x, D_val) = - grad(x) . h_val
            influence = self.relu(-grad.dot(self.h_val))
            influences.append(influence)

        self.model.train(train_mode)
        influences = torch.stack(influences)
        return influences


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset, random_split

    class LeNet5(nn.Module):
        def __init__(self):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, 16 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def test_LiSSA():
        print("Running LeNet5 MNIST influence test...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load MNIST
        train_set = torchvision.datasets.MNIST(
            root="./datasets/MNIST/",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        test_set = torchvision.datasets.MNIST(
            root="./datasets/MNIST/",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        # Subsample for speed in this test
        train_subset, _ = random_split(train_set, [1000, len(train_set) - 1000])
        test_subset, _ = random_split(test_set, [100, len(test_set) - 100])
        trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
        testloader = DataLoader(test_subset, batch_size=32, shuffle=False)

        # 2. Train Model (Pretrain)
        print("Training LeNet5...")
        model = LeNet5().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(2):  # Short training
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
        print("Training done.")

        # 3. Compute Influence
        print("Computing Influence...")
        scorer = LiSSAInfluenceScorer(
            model=model,
            loss_fn=loss_fn,
            batch_unpacker=BatchUnpacker(),
            lissa_depth=5,
            scale=0.001,
            damping=0.01,
        )
        scorer.fit(trainloader, testloader)

        print(f"Val grads shape: {scorer.grad_val.shape}")
        print(f"IHVP shape: {scorer.h_val.shape}")

        # Check influence of first batch of training data
        batch = next(iter(trainloader))
        print("Computing influence scores for candidate batch...")
        scores = scorer(batch)
        print("Influence Scores (first 5):", scores[:5])
        print("Done.")

    test_LiSSA()
