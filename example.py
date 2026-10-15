"""Run BEANet on synthetic inputs; no dataset or checkpoint is required."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from beanet import BEANet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--backward", action="store_true", help="Also run one synthetic gradient step"
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model = BEANet().to(device)
    h = torch.randn(2, 300, 16, 16, device=device)
    f = torch.randn_like(h)

    model.eval()
    with torch.no_grad():
        output = model(h, f, return_diagnostics=True)
    assert output["logits"].shape == (2, 24)
    assert torch.isfinite(output["logits"]).all()
    for name in ("gates", "late_gates"):
        weights = output[name]
        assert torch.allclose(
            weights.sum(dim=1), torch.ones(2, device=device), atol=1e-6
        )
    print(f"Logits: {tuple(output['logits'].shape)}")
    print(f"Expert weights: {tuple(output['gates'].shape)}")
    print(f"Source weights: {tuple(output['late_gates'].shape)}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    if args.backward:
        model.train()
        labels = torch.tensor([0, 23], dtype=torch.long, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.CrossEntropyLoss()(model(h, f), labels)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        optimizer.step()
        print(f"Synthetic training loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
