import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from .batch_unpack import BatchUnpacker
from .lissa import compute_inverse_hvp, compute_val_grad


class ResidualInfluenceScorer(nn.Module):
    r"""Computes Marginal Influence using a frozen network.

    "Residual" Gap logic:
    1. Target gradient: The gradient the model require to solve the task.
        v = \nabla_{\theta} L(D)
    2. Accumulated gradient: The gradients of current samples in the condensed dataset.
        v^ = \nabla_{\theta} L(D^)
    3. Marginal Gap: The gap between the target gradient and the accumulated gradient.
        u = v - v^
    4. Score for a candidate z:
        Score(z) = u^T . H^{-1} . \nabla_{\theta} L(z)
                 = \nabla_{\theta}^T L(z) . H^{-1} . u
    5. Let h = H^{-1} . u, then
        Score(z) = \nabla_{\theta}^T L(z) . h

    At each iteration, we need to update v^, and correspondingly, u and h.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        params_inf: list[nn.Parameter] | None,
        batch_unpacker: BatchUnpacker,
        damping: float = 1e-3,
        lissa_depth: int = 10,
        scale: float = 0.001,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.batch_unpacker = batch_unpacker
        self.damping = damping
        self.lissa_depth = lissa_depth
        self.scale = scale

        if params_inf is None:
            self.params_inf = [p for p in model.parameters()]
        else:
            self.params_inf = params_inf

        if self.batch_unpacker is None:
            self.batch_unpacker = BatchUnpacker()

        self.relu = nn.ReLU()
        self.device = next(self.model.parameters()).device

        self.v: Tensor = None
        self.vhat: Tensor = None

    def fit(self, dataloader: DataLoader):
        # Step 1: Compute target gradient
        self.v = compute_val_grad(
            model=self.model,
            params_inf=self.params_inf,
            valloader=dataloader,
            loss_fn=self.loss_fn,
            batch_unpacker=self.batch_unpacker,
            device=self.device,
        ).to(self.device)

        # Step 2: Initialize accumulated gradient
        self.vhat = torch.zeros_like(self.v)
        self.h_val = self.recompute_hvp(dataloader).to(device=self.device)

    def update(self, dataloader_condense: DataLoader, dataloader_hessian: DataLoader):
        """Update vhat, u, and h.

        Args:
            dataloader_condense: Dataloader for the condensed dataset (to compute vhat).
            dataloader_hessian: Dataloader for the Hessian approximation (LiSSA).
        """
        # Update vhat (Accumulated gradient)
        self.vhat = compute_val_grad(
            model=self.model,
            params_inf=self.params_inf,
            valloader=dataloader_condense,
            loss_fn=self.loss_fn,
            batch_unpacker=self.batch_unpacker,
            device=self.device,
        ).to(self.device)

        # Update h (which uses the new vhat to compute u = v - vhat)
        self.h_val = self.recompute_hvp(dataloader_hessian).to(device=self.device)

    def recompute_hvp(self, dataloader: DataLoader) -> Tensor:
        """Recompute h_val = H^{-1} (u)."""
        u = self.v - self.vhat
        h = compute_inverse_hvp(
            model=self.model,
            params_inf=self.params_inf,
            grad_val=u,
            loss_fn=self.loss_fn,
            trainloader=dataloader,
            batch_unpacker=self.batch_unpacker,
            lissa_depth=self.lissa_depth,
            scale=self.scale,
            damping=self.damping,
            device=self.device,
        )
        return h

    def forward(self, batch) -> Tensor:
        inputs, targets = self.batch_unpacker(batch)
        inputs = inputs.to(device=self.device)
        targets = targets.to(device=self.device)
        batch_size = inputs.shape[0]

        train_mode = self.model.training
        self.model.eval()

        scores = []
        for i in range(batch_size):
            x = inputs[[i]]
            y = targets[[i]]

            self.model.zero_grad()
            out = self.model(x)
            loss = self.loss_fn(out, y)

            grad = torch.autograd.grad(outputs=loss, inputs=self.params_inf)
            grad = torch.cat(tensors=[p.flatten() for p in grad], dim=0)

            # Dot product
            score = grad.dot(self.h_val)
            scores.append(score)

        self.model.train(train_mode)
        scores = torch.stack(scores)
        return scores


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

    def test_ResidualInfluence():
        print("Running LeNet5 MNIST Residual Influence test...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load MNIST
        dataset = {
            "train": Subset(
                dataset=torchvision.datasets.MNIST(
                    root="./datasets/MNIST/",
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                ),
                indices=range(1000),
            ),
            "test": Subset(
                dataset=torchvision.datasets.MNIST(
                    root="./datasets/MNIST/",
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                ),
                indices=range(1000),
            ),
        }

        # Subsample for speed
        dataloader = {
            "train": DataLoader(dataset["train"], batch_size=32, shuffle=True),
            "test": DataLoader(dataset["test"], batch_size=32, shuffle=False),
        }

        # 2. Train Model (Pretrain)
        print("Training LeNet5...")
        model = LeNet5().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(1):  # Very short training
            for inputs, targets in dataloader["train"]:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
        print("Training done.")

        # 3. Compute Residual Influence
        print("Computing Residual Influence...")
        scorer = ResidualInfluenceScorer(
            model=model,
            params_inf=list(model.fc3.parameters()),
            loss_fn=loss_fn,
            batch_unpacker=BatchUnpacker(),
            lissa_depth=10,
            scale=0.001,
            damping=0.01,
        )
        scorer.fit(dataloader["test"])

        print(f"Val grads (v) shape: {scorer.v.shape}")
        print(f"Residual IHVP (h_val) shape: {scorer.h_val.shape}")

        batch = next(iter(dataloader["train"]))
        print("Computing influence scores for candidate batch...")
        scores = scorer(batch)
        print("Influence Scores (first 5):", scores[:5])

        # Test Update
        print("Updating score with synthetic set...")
        # Create a small loader for the "condensed" set (just the batch)
        # In practice, this would be a real condensed set.

        scorer.update(
            dataloader_condense=dataloader["train"],
            dataloader_hessian=dataloader["test"],
        )
        print(f"Accumulated gradient (vhat) norm: {scorer.vhat.norm()}")

        # Score again
        scores_new = scorer(batch)
        print("New Influence Scores (first 5):", scores_new[:5])
        print("Done.")

    test_ResidualInfluence()
