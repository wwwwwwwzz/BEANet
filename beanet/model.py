"""Three-source classification with bio-optical reliability conditioning."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ._heads import SPRNStyleRawHead
from .fluorescence import FluorescenceBranch
from .fusion import BioOpticalDescriptor, InteractionFusion
from .reflectance import ReflectanceBranch


class EncoderFusion(nn.Module):
    """Modality encoders followed by four-expert interaction fusion."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.halogenbranch = ReflectanceBranch(in_channels, 16)
        self.fluorescencebranch = FluorescenceBranch(in_channels)
        self.MoE2F = InteractionFusion(
            halogen_ch=8,
            fluorescence_ch=32,
            hidden_dim=64,
            num_classes=num_classes,
            qc_dim=32,
            spectral_state_dim=32,
        )

    def forward(
        self, h: torch.Tensor, f: torch.Tensor, return_diagnostics: bool = False
    ):
        h_features = self.halogenbranch(h)
        f_features = self.fluorescencebranch(f)
        return self.MoE2F(
            h_features, f_features, h, f, return_diagnostics=return_diagnostics
        )


class BEANet(nn.Module):
    """Classify preprocessed, spatially paired 16 x 16 hyperspectral patches.

    The default output has 24 logits. A joint label uses species-major order:
    ``joint_label = species_index * 3 + burden_index``.
    """

    def __init__(self, in_channels: int = 300, num_classes: int = 24):
        super().__init__()
        if in_channels < 3:
            raise ValueError("in_channels must be at least 3 for spectral derivatives")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.core = EncoderFusion(in_channels, num_classes)
        self.h_raw_head = SPRNStyleRawHead(in_ch=in_channels, num_classes=num_classes)
        self.f_raw_head = SPRNStyleRawHead(in_ch=in_channels, num_classes=num_classes)
        self.qc_token = BioOpticalDescriptor(out_dim=32)
        self.late_gate = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.late_gate[-1].weight)
        nn.init.zeros_(self.late_gate[-1].bias)

    def forward(
        self, h: torch.Tensor, f: torch.Tensor, return_diagnostics: bool = False
    ):
        """Return logits, or logits and routing diagnostics when requested."""
        if h.ndim != 4 or f.ndim != 4 or h.shape != f.shape:
            raise ValueError(
                "Both inputs must have the same [batch, bands, height, width] shape"
            )
        if h.shape[1:] != (self.in_channels, 16, 16):
            raise ValueError(
                f"Expected inputs with shape [batch, {self.in_channels}, 16, 16]"
            )
        if not h.is_floating_point() or not f.is_floating_point():
            raise ValueError("Inputs must be floating-point tensors")
        if h.device != f.device or h.dtype != f.dtype:
            raise ValueError("Inputs must use the same device and dtype")

        interaction = self.core(h, f, return_diagnostics=return_diagnostics)
        interaction_logits = (
            interaction["logits"] if return_diagnostics else interaction
        )
        h_logits = self.h_raw_head(h)
        f_logits = self.f_raw_head(f)
        qc = self.qc_token(h, f)
        source_weights = F.softmax(self.late_gate(qc), dim=-1)
        logits = (
            source_weights[:, 0:1] * interaction_logits
            + source_weights[:, 1:2] * h_logits
            + source_weights[:, 2:3] * f_logits
        )
        if not return_diagnostics:
            return logits

        expert_logits = dict(interaction["expert_logits"])
        expert_logits.update(
            {"core": interaction_logits, "raw_h": h_logits, "raw_f": f_logits}
        )
        return {
            "logits": logits,
            "gates": interaction["gates"],
            "embedding": interaction["embedding"],
            "late_gates": source_weights,
            "classwise_late_gates": None,
            "qc_token": qc,
            "expert_logits": expert_logits,
        }
