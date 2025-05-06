import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from .lissa import compute_inverse_hvp, compute_val_grad
from .batch_unpack import BatchUnpacker


class JointMarginalInfluenceScorer(nn.Module):
    def __init__(
        self,
        teacher: nn.Module,
        loss_fn: nn.Module,
        params_inf_T: list[nn.Parameter] | None,
        batch_unpacker: BatchUnpacker = BatchUnpacker(),
        damping: float = 1e-3,
        lissa_depth: int = 10,
        scale: float = 0.001,
    ):
        super().__init__()
        self.teacher = teacher
        self.params_inf_T = params_inf_T
        self.loss_fn = loss_fn
        self.batch_unpacker = batch_unpacker
        self.damping = damping
        self.lissa_depth = lissa_depth
        self.scale = scale

        if self.params_inf_T is None:
            self.params_inf_T: list[nn.Parameter] = [p for p in self.teacher.parameters()]

        self.relu = nn.ReLU()
        self.device = next(self.teacher.parameters()).device

    def update_student(
        self,
        student: nn.Module,
        params_inf_S: list[nn.Parameter] | None,
        trainloader: DataLoader,
        valloader: DataLoader,
    ):
        self.student = student
        self.params_inf_S = params_inf_S

        if self.params_inf_S is None:
            self.params_inf_S: list[nn.Parameter] = [p for p in self.student.parameters()]

        self.fit_student(trainloader=trainloader, valloader=valloader)

    def fit_teacher(self, trainloader: DataLoader, valloader: DataLoader):
        grad_val = compute_val_grad(
            model=self.teacher,
            params_inf=self.params_inf_T,
            valloader=valloader,
            loss_fn=self.loss_fn,
            batch_unpacker=self.batch_unpacker,
            device=self.device,
        ).to(self.device)
        h_val = compute_inverse_hvp(
            model=self.teacher,
            params_inf=self.params_inf_T,
            grad_val=grad_val,
            loss_fn=self.loss_fn,
            trainloader=trainloader,
            batch_unpacker=self.batch_unpacker,
            lissa_depth=self.lissa_depth,
            scale=self.scale,
            damping=self.damping,
            device=self.device,
        ).to(self.device)

        self.grad_val_T = grad_val
        self.h_val_T = h_val

    def fit_student(self, trainloader: DataLoader, valloader: DataLoader):
        grad_val = compute_val_grad(
            model=self.student,
            params_inf=self.params_inf_S,
            valloader=valloader,
            loss_fn=self.loss_fn,
            batch_unpacker=self.batch_unpacker,
            device=self.device,
        ).to(self.device)
        h_val = compute_inverse_hvp(
            model=self.student,
            params_inf=self.params_inf_S,
            grad_val=grad_val,
            loss_fn=self.loss_fn,
            trainloader=trainloader,
            batch_unpacker=self.batch_unpacker,
            lissa_depth=self.lissa_depth,
            scale=self.scale,
            damping=self.damping,
            device=self.device,
        ).to(self.device)

        self.grad_val_S = grad_val
        self.h_val_S = h_val

    def forward(self, batch) -> Tensor:
        # Use both teacher and student influence
        # I(z, D_val) = I_teacher * I_student
        inputs, targets = self.batch_unpacker(batch)
        inputs = inputs.to(device=self.device)
        targets = targets.to(device=self.device)
        batch_size = inputs.shape[0]

        train_mode_T = self.teacher.training
        train_mode_S = self.student.training
        self.teacher.eval()
        self.student.eval()

        influences = []
        for i in range(batch_size):
            x = inputs[[i]]
            y = targets[[i]]

            # Teacher Influence
            self.teacher.zero_grad()
            out_T = self.teacher(x)
            loss_T = self.loss_fn(out_T, y)
            grad_T = torch.autograd.grad(outputs=loss_T, inputs=self.params_inf_T)
            grad_T = torch.cat(tensors=[p.flatten() for p in grad_T], dim=0)
            influence_T = self.relu(-grad_T.dot(self.h_val_T))

            # Student Influence
            self.student.zero_grad()
            out_S = self.student(x)
            loss_S = self.loss_fn(out_S, y)
            grad_S = torch.autograd.grad(outputs=loss_S, inputs=self.params_inf_S)
            grad_S = torch.cat(tensors=[p.flatten() for p in grad_S], dim=0)
            influence_S = self.relu(-grad_S.dot(self.h_val_S))

            influences.append(influence_T * influence_S)

        self.teacher.train(train_mode_T)
        self.student.train(train_mode_S)
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

    def test_JMI():
        print("Running LeNet5 MNIST JMI test...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load MNIST
        dataset = {
            "train_T": Subset(
                dataset=torchvision.datasets.MNIST(
                    root="./datasets/MNIST/",
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                ),
                indices=range(1000),
            ),
            "train_S": Subset(
                dataset=torchvision.datasets.MNIST(
                    root="./datasets/MNIST/",
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                ),
                indices=range(100),
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
        dataloader = {
            "train_T": DataLoader(dataset["train_T"], batch_size=20, shuffle=True),
            "train_S": DataLoader(dataset["train_S"], batch_size=20, shuffle=True),
            "test": DataLoader(dataset["test"], batch_size=20, shuffle=False),
        }

        # 2. Train Model (Pretrain)
        print("Training LeNet5...")
        loss_fn = nn.CrossEntropyLoss()

        net_T = LeNet5().to(device)
        opt_T = torch.optim.Adam(net_T.parameters(), lr=0.001)
        net_T.train()
        for epoch in range(2):
            for inputs, targets in dataloader["train_T"]:
                inputs, targets = inputs.to(device), targets.to(device)
                opt_T.zero_grad()
                outputs = net_T(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                opt_T.step()
        print("Training teacher done.")

        net_S = LeNet5().to(device)
        opt_S = torch.optim.Adam(net_S.parameters(), lr=0.001)
        net_S.train()
        for epoch in range(2):
            for inputs, targets in dataloader["train_S"]:
                inputs, targets = inputs.to(device), targets.to(device)
                opt_S.zero_grad()
                outputs = net_S(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                opt_S.step()
        print("Training student done.")

        # 3. Compute Influence
        scorer = JointMarginalInfluenceScorer(
            teacher=net_T,
            loss_fn=loss_fn,
            params_inf_T=None,
            batch_unpacker=BatchUnpacker(),
            lissa_depth=5,
            scale=0.001,
            damping=0.01,
        )
        print("Computing Influence...")
        scorer.fit_teacher(
            trainloader=dataloader["train_T"],
            valloader=dataloader["test"],
        )
        print(f"Val grads shape: {scorer.grad_val_T.shape}")
        print(f"IHVP shape: {scorer.h_val_T.shape}")

        scorer.update_student(
            student=net_S,
            params_inf_S=None,
            trainloader=dataloader["train_S"],
            valloader=dataloader["test"],
        )
        print(f"Val grads shape: {scorer.grad_val_S.shape}")
        print(f"IHVP shape: {scorer.h_val_S.shape}")

        # Check influence of first batch of training data
        batch = next(iter(dataloader["train_S"]))
        print("Computing influence scores for candidate batch...")
        scores = scorer(batch)
        print("Influence Scores (first 5):", scores[:5])
        print("Done.")

    test_JMI()
