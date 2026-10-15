from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BioOpticalDescriptor(nn.Module):
    """Project 20 within-path statistics and three cross-path statistics."""

    def __init__(self, out_dim: int = 32, outlier_z: float = 3.5):
        super().__init__()
        self.outlier_z = float(outlier_z)
        self.project = nn.Sequential(
            nn.Linear(23, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _safe_std(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[dim] <= 1:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return x.std(dim=dim, unbiased=False)

    def _path_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.finfo(x.dtype).eps
        profile = x.mean(dim=(-1, -2))
        centered = profile - profile.mean(dim=1, keepdim=True)
        d1 = profile.diff(dim=1)
        d2 = d1.diff(dim=1) if d1.shape[1] > 1 else d1
        freq = torch.fft.rfft(centered.float(), dim=1).abs().to(profile.dtype)
        if freq.shape[1] > 2:
            cut = max(1, freq.shape[1] // 4)
            low = freq[:, 1 : 1 + cut].pow(2).mean(dim=1)
            high = freq[:, -cut:].pow(2).mean(dim=1)
        else:
            low = freq.pow(2).mean(dim=1)
            high = low
        abs_x = x.abs()
        stats = torch.stack(
            [
                profile.mean(dim=1),
                self._safe_std(profile, dim=1),
                d1.abs().mean(dim=1),
                self._safe_std(d1, dim=1),
                d2.abs().mean(dim=1),
                d2.pow(2).mean(dim=1) / (d1.pow(2).mean(dim=1) + eps),
                low,
                high / (low + eps),
                (x < -self.outlier_z).float().mean(dim=(1, 2, 3)),
                (abs_x > self.outlier_z).float().mean(dim=(1, 2, 3)),
            ],
            dim=1,
        )
        return (stats, profile)

    def forward(self, h: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(h.dtype).eps
        h_stats, h_profile = self._path_stats(h)
        f_stats, f_profile = self._path_stats(f)
        h_norm = F.normalize(h_profile, dim=1)
        f_norm = F.normalize(f_profile, dim=1)
        cross = torch.stack(
            [
                (h_profile - f_profile).abs().mean(dim=1),
                (h_norm * f_norm).sum(dim=1),
                h_profile.abs().mean(dim=1) / (f_profile.abs().mean(dim=1) + eps),
            ],
            dim=1,
        )
        return self.project(torch.cat([h_stats, f_stats, cross], dim=1))


class SpectralStateEncoder(nn.Module):
    """Summarize mean spectra with gated depthwise one-dimensional convolutions."""

    def __init__(self, out_dim: int = 32, hidden_dim: int = 48, kernel_size: int = 9):
        super().__init__()
        padding = kernel_size // 2
        self.in_proj = nn.Conv1d(2, hidden_dim * 2, kernel_size=1)
        self.dw_scan = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
            bias=False,
        )
        self.gate_scan = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
            bias=False,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.GELU()
        )

    def forward(self, h: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        h_profile = h.mean(dim=(-1, -2))
        f_profile = f.mean(dim=(-1, -2))
        x = torch.stack([h_profile, f_profile], dim=1)
        x = (x - x.mean(dim=-1, keepdim=True)) / (
            x.std(dim=-1, keepdim=True, unbiased=False) + 1e-06
        )
        value, gate = self.in_proj(x).chunk(2, dim=1)
        value = F.silu(self.dw_scan(value))
        gate = torch.sigmoid(self.gate_scan(gate))
        state = value * gate
        pooled = torch.cat([state.mean(dim=-1), state.amax(dim=-1)], dim=1)
        return self.out_proj(pooled)


class InteractionFusion(nn.Module):
    """Route between global, regional, reflectance, and fluorescence experts."""

    def __init__(
        self,
        halogen_ch: int,
        fluorescence_ch: int,
        hidden_dim: int = 64,
        num_classes: int = 10,
        qc_dim: int = 32,
        spectral_state_dim: int = 32,
    ):
        super().__init__()
        self.local_fusion = nn.Sequential(
            nn.Conv2d(halogen_ch + fluorescence_ch, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.mod_H = nn.Parameter(torch.empty(1, halogen_ch, 16, 16))
        self.mod_F = nn.Parameter(torch.empty(1, fluorescence_ch, 16, 16))
        nn.init.orthogonal_(self.mod_H)
        nn.init.orthogonal_(self.mod_F)
        self.expert_pool1 = nn.Linear(hidden_dim, num_classes)
        self.expert_pool2 = nn.Linear(hidden_dim * 4, num_classes)
        self.exp_H = nn.Linear(halogen_ch, num_classes)
        self.exp_F = nn.Linear(fluorescence_ch, num_classes)
        self.qc_token = BioOpticalDescriptor(out_dim=qc_dim)
        self.spectral_state = SpectralStateEncoder(out_dim=spectral_state_dim)
        router_in = (
            hidden_dim + halogen_ch + fluorescence_ch + qc_dim + spectral_state_dim
        )
        router_hidden = max(hidden_dim, router_in // 3)
        self.router = nn.Sequential(
            nn.Linear(router_in, router_hidden),
            nn.LayerNorm(router_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(router_hidden, 4),
        )
        self.quality_bias = nn.Sequential(
            nn.Linear(
                qc_dim + spectral_state_dim, max(16, (qc_dim + spectral_state_dim) // 2)
            ),
            nn.GELU(),
            nn.Linear(max(16, (qc_dim + spectral_state_dim) // 2), 4),
        )

    def forward(
        self,
        H_feat: torch.Tensor,
        F_feat: torch.Tensor,
        H_raw: torch.Tensor,
        F_raw: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        z = torch.cat([H_feat + self.mod_H, F_feat + self.mod_F], dim=1)
        z = self.local_fusion(z)
        z_pooled1 = self.pool1(z).flatten(1)
        z_pooled2 = self.pool2(z).flatten(1)
        h_global = H_feat.flatten(2).mean(dim=-1)
        f_global = F_feat.flatten(2).mean(dim=-1)
        qc = self.qc_token(H_raw, F_raw)
        spectral_state = self.spectral_state(H_raw, F_raw)
        route_context = torch.cat([qc, spectral_state], dim=1)
        gate_input = torch.cat([h_global, f_global, z_pooled1, route_context], dim=1)
        gate_logits = self.router(gate_input) + 0.25 * self.quality_bias(route_context)
        gates = F.softmax(gate_logits, dim=-1)
        pool1_logits = self.expert_pool1(z_pooled1)
        pool2_logits = self.expert_pool2(z_pooled2)
        h_logits = self.exp_H(h_global)
        f_logits = self.exp_F(f_global)
        pred = (
            gates[:, 0:1] * pool1_logits
            + gates[:, 1:2] * pool2_logits
            + gates[:, 2:3] * h_logits
            + gates[:, 3:4] * f_logits
        )
        embedding = torch.cat(
            [z_pooled1, h_global, f_global, qc, spectral_state], dim=1
        )
        if return_diagnostics:
            return {
                "logits": pred,
                "gates": gates,
                "embedding": embedding,
                "qc_token": qc,
                "spectral_state": spectral_state,
                "expert_logits": {
                    "fusion": pool1_logits,
                    "pool2": pool2_logits,
                    "h": h_logits,
                    "f": f_logits,
                },
            }
        return pred
