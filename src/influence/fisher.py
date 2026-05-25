import torch
from torch import nn, Tensor
import torch
import torch.nn as nn
from torch.func import vmap, grad, functional_call


class FisherScorer:
    """Computes Fisher information proxy (gradient norm) for each sample.

    Args:
        model: Inference model.
        params_inf: Layers of model to compute influence.
        batch_size: Size of batches for processing.
        loss_fn: Loss function (currently supports "ce").
        use_goodfellow: If True, uses the Goodfellow trick. Otherwise, uses vmap.

    ---
    Note on the Goodfellow Trick:
        This method computes per-sample gradient norms (Fisher Information proxies)
        efficiently by leveraging the chain rule structure of Linear layers.

        Instead of performing a costly backward pass for each individual sample, we:
        1. Use forward hooks to capture the input activations (a).
        2. Use backward hooks to capture the output gradients (g) from a single
        aggregated batch backward pass.
        3. Recover the per-sample gradient norm for sample 'i' by calculating the
        product of the norms: ||grad_i|| = ||a_i|| * ||g_i||.

        This avoids the OOM (Out-of-Memory) risks associated with `torch.vmap` while
        remaining significantly faster than looping through the batch.
        Note that if the loss function uses 'mean' reduction, 'g' must be rescaled
        by the batch size to represent the true individual gradients.
    """

    supported_losses = (nn.CrossEntropyLoss, nn.MSELoss, nn.BCEWithLogitsLoss)

    def __init__(
        self,
        model: nn.Module,
        params_inf: nn.Module | None = None,
        batch_size: int = 1024,
        loss_fn: str | nn.Module = "ce",
        use_goodfellow_trick: bool = True,
    ):
        self.model = model
        self.params_inf = params_inf if params_inf is not None else model
        self.batch_size = batch_size
        self._loss_fn: nn.Module = nn.CrossEntropyLoss(reduction="mean") if loss_fn == "ce" else loss_fn
        self.use_goodfellow_trick = use_goodfellow_trick

        self.params_dict = dict(self.params_inf.named_parameters())

        if self.use_goodfellow_trick:
            # Goodfellow trick strictly requires Linear layers to use the norm-product identity
            self.target_layers = []
            for m in self.params_inf.modules():
                assert isinstance(m, nn.Linear), (
                    "Goodfellow trick currently only supports nn.Linear layers."
                )
                self.target_layers.append(m)
            assert len(self.target_layers) > 0, (
                "Goodfellow trick requires at least one nn.Linear layer."
            )

            # Goodfellow trick strictly requires a loss function where the per-sample gradient can
            # be expressed as an outer product of the input and output gradients.
            assert isinstance(self._loss_fn, self.supported_losses), (
                f"Goodfellow trick only supports {self.supported_losses}."
            )
            # Mean reduction is required to ensure the output gradients are properly scaled for this identity to hold.
            assert self._loss_fn.reduction == "mean", (
                "Goodfellow trick requires 'mean' reduction for loss function."
            )

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self.use_goodfellow_trick:
            return self._call_with_goodfellow_trick(inputs, targets)
        else:
            return self._call_with_vmap(inputs, targets)

    def _call_with_goodfellow_trick(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Efficient per-sample gradients via outer-product norm identity."""
        self.model.eval()
        device = next(self.model.parameters()).device
        all_norms = []

        hooks = []
        for layer in self.target_layers:
            hooks.append(layer.register_forward_pre_hook(self._save_input))
            hooks.append(layer.register_full_backward_hook(self._save_grad_output))

        for i in range(0, inputs.shape[0], self.batch_size):
            bx, by = (
                inputs[i : i + self.batch_size].to(device),
                targets[i : i + self.batch_size].to(device),
            )
            bs = bx.shape[0]

            self.model.zero_grad()
            loss = self._loss_fn(self.model(bx), by)
            loss.backward()

            b_sq_norms = torch.zeros(bs, device=device)
            for layer in self.target_layers:
                # Rescale gradients because 'mean' reduction was used
                res = layer.grad_res * bs
                b_sq_norms += layer.input_act.pow(2).sum(1) * res.pow(2).sum(1)
                if layer.bias is not None:
                    b_sq_norms += res.pow(2).sum(1)

            all_norms.append(b_sq_norms.sqrt())

        for h in hooks:
            h.remove()
        return torch.cat(all_norms)

    def _call_with_vmap(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Per-sample gradients via vectorized autograd."""
        self.model.eval()
        device = next(self.model.parameters()).device
        all_norms = []

        def compute_loss(params, x, y):
            out = functional_call(self.model, params, x.unsqueeze(0))
            return self._loss_fn(out, y.unsqueeze(0))

        grad_fn = grad(compute_loss, argnums=0)
        batched_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))

        for i in range(0, inputs.shape[0], self.batch_size):
            bx, by = (
                inputs[i : i + self.batch_size].to(device),
                targets[i : i + self.batch_size].to(device),
            )
            per_sample_grads = batched_grad_fn(self.params_dict, bx, by)

            b_sq_norms = torch.zeros(bx.shape[0], device=device)
            for g in per_sample_grads.values():
                b_sq_norms += g.reshape(bx.shape[0], -1).pow(2).sum(dim=1)

            all_norms.append(b_sq_norms.sqrt())

        return torch.cat(all_norms)

    @staticmethod
    def _save_input(m, i):
        m.input_act = i[0].detach()

    @staticmethod
    def _save_grad_output(m, gi, go):
        m.grad_res = go[0].detach()


if __name__ == "__main__":

    def verify_all_fisher_methods(model, inputs, targets):
        """
        Verifies numerical consistency between:
        1. Ground Truth (Sequential Loop)
        2. Goodfellow Trick (Hook-based)
        3. vmap (Vectorized Autograd)
        """

        def start_timer():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return start, end

        def stop_timer(start, end):
            end.record()
            torch.cuda.synchronize()  # Wait for GPU to finish
            return start.elapsed_time(end)  # Returns milliseconds

        model.eval()
        device = next(model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.shape[0]
        loss_fn = nn.CrossEntropyLoss(reduction="mean")

        # 1. Initialize Scorer for both methods
        scorer_gf = FisherScorer(model, loss_fn=loss_fn, use_goodfellow_trick=True)
        scorer_vm = FisherScorer(model, loss_fn=loss_fn, use_goodfellow_trick=False)

        # --- METHOD 1: Goodfellow Trick ---
        s, e = start_timer()
        norms_gf = scorer_gf(inputs, targets).cpu()
        time_gf = stop_timer(s, e)

        # --- METHOD 2: Vmap ---
        s, e = start_timer()
        norms_vm = scorer_vm(inputs, targets).cpu()
        time_vm = stop_timer(s, e)

        # --- METHOD 3: Ground Truth (Slow Loop) ---
        s, e = start_timer()
        norms_slow = []
        for i in range(batch_size):
            model.zero_grad()
            x, y = inputs[i : i + 1], targets[i : i + 1]
            loss = loss_fn(model(x), y)
            loss.backward()
            sample_sq_norm = torch.tensor(0.0, device=device)
            # Match against Linear layers tracked by Scorer
            for layer in scorer_gf.target_layers:
                for p in layer.parameters():
                    if p.grad is not None:
                        sample_sq_norm += p.grad.detach().pow(2).sum()
            norms_slow.append(sample_sq_norm.sqrt())
        norms_slow = torch.stack(norms_slow).cpu()
        time_slow = stop_timer(s, e)

        # --- COMPARISON ---
        diff_gf_slow = torch.abs(norms_gf - norms_slow).max().item()
        diff_vm_slow = torch.abs(norms_vm - norms_slow).max().item()
        diff_gf_vm = torch.abs(norms_gf - norms_vm).max().item()

        print(f"\n--- Verification Report [{loss_fn.__class__.__name__.upper()}] ---")
        print(f"Goodfellow vs. Slow: {diff_gf_slow:.2e}")
        print(f"Vmap vs. Slow:       {diff_vm_slow:.2e}")
        print(f"Goodfellow vs. Vmap: {diff_gf_vm:.2e}")
        print(f"\nTiming:")
        print(f"Goodfellow Time: {time_gf:.2f} ms")
        print(f"Vmap Time:       {time_vm:.2f} ms")
        print(f"Slow Loop Time:  {time_slow:.2f} ms")

        # Threshold 1e-5 for float32 precision
        success = all(d < 1e-5 for d in [diff_gf_slow, diff_vm_slow, diff_gf_vm])

        if success:
            print("All methods are numerically consistent.")
        else:
            print("Discrepancy detected. Check reduction scaling or layer filtering.")

        return success

    verify_all_fisher_methods(
        model=nn.Linear(10, 5),
        inputs=torch.normal(mean=0, std=1, size=[10000, 10]),
        targets=torch.randint(low=0, high=5, size=[10000]),
    )
