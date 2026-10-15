from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


from .fluorescence import NNAD


def make_bior11_analysis_kernels_base(dtype=torch.float32, device="cpu"):
    LL = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype, device=device)
    LH = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=dtype, device=device)
    HL = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=dtype, device=device)
    HH = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=dtype, device=device)
    return torch.stack([LL, LH, HL, HH], dim=0)


def make_analysis_kernels(in_ch, dtype=torch.float32, device="cpu"):
    base = make_bior11_analysis_kernels_base(dtype=dtype, device=device)
    kernels = base.unsqueeze(1).repeat(in_ch, 1, 1, 1).reshape(in_ch * 4, 1, 2, 2)
    return kernels


def make_bior11_synthesis_kernels_base(dtype=torch.float32, device="cpu"):
    LL = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype, device=device)
    LH = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=dtype, device=device)
    HL = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=dtype, device=device)
    HH = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=dtype, device=device)
    return torch.stack([LL, LH, HL, HH], dim=0)


def make_synthesis_kernels(in_ch, dtype=torch.float32, device="cpu"):
    base = make_bior11_synthesis_kernels_base(dtype=dtype, device=device)
    kernels = base.unsqueeze(1).repeat(1, in_ch, 1, 1, 1).view(4, in_ch, 1, 2, 2)
    return kernels


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=kernel,
                padding=padding,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x):
        if x.shape[-1] < 3 or x.shape[-2] < 3:
            return torch.sigmoid(torch.mean(x, dim=1, keepdim=True))
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class ChannelAttention(nn.Module):

    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(ch, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch, bias=False),
            nn.Sigmoid(),
        )
        self.eps = 1e-06

    def forward(self, x):
        b, c = x.shape[:2]
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = self.fc(y).view(b, c, 1, 1)
        w = w / (w.mean(dim=1, keepdim=True) + self.eps)
        return w


class SubbandProcessor(nn.Module):
    """Process one subband with spatial attention followed by channel attention."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch, out_ch)
        self.satt = SpatialAttention()
        self.catt = ChannelAttention(out_ch, 16)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        feat = self.conv(x)
        SA = self.satt(feat)
        feat_sa = feat * SA
        CA = self.catt(feat_sa)
        feat_ca = feat_sa * CA
        out = feat_ca + self.skip(x)
        out = self.bn(out)
        return F.relu(out, inplace=True)


class MWCSFA(nn.Module):
    """Aggregate wavelet subbands using constrained, learned interactions."""

    def __init__(
        self,
        in_ch=300,
        subband_ch=64,
        levels=3,
        learnable_dwt=False,
        learnable_idwt=False,
        per_level_W=False,
        default_allowed_mask=None,
        off_limit=0.5,
        gamma_schedule=None,
    ):
        super().__init__()
        assert levels >= 1
        self.in_ch = in_ch
        self.subband_ch = subband_ch
        self.levels = levels
        self.learnable_dwt = learnable_dwt
        self.learnable_idwt = learnable_idwt
        self.per_level_W = per_level_W
        self.off_limit = float(off_limit)
        ak = make_analysis_kernels(in_ch)
        if learnable_dwt:
            self.analysis_kernels = nn.Parameter(ak)
        else:
            self.register_buffer("analysis_kernels", ak)
        sk = make_synthesis_kernels(in_ch)
        if learnable_idwt:
            self.synth_kernels = nn.Parameter(sk.view(4 * in_ch, 1, 2, 2))
        else:
            self.register_buffer("synth_kernels", sk)
        self.level_processors = nn.ModuleList()
        for _ in range(levels):
            procs = nn.ModuleList([SubbandProcessor(in_ch, in_ch) for _ in range(4)])
            self.level_processors.append(procs)
        self.deeper_to_pseudos = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False)
                        for _ in range(4)
                    ]
                )
                for _ in range(levels)
            ]
        )
        self.final_reduce = nn.Sequential(
            nn.Conv2d(in_ch, subband_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(subband_ch),
            nn.ReLU(inplace=True),
        )
        if self.per_level_W:
            self.W_raw = nn.Parameter(torch.randn(levels, 4, 4) * 0.01)
        else:
            self.W_raw = nn.Parameter(torch.randn(1, 4, 4) * 0.01)
        if default_allowed_mask is None:
            default_allowed_mask = torch.tensor(
                [
                    [0.0, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.3, 0.2],
                    [0.0, 0.3, 0.0, 0.2],
                    [0.0, 0.2, 0.2, 0.0],
                ],
                dtype=torch.float32,
            )
        self.register_buffer("mix_allowed_mask", default_allowed_mask.clone())
        self.M_param = nn.Parameter(torch.zeros_like(default_allowed_mask))
        if gamma_schedule is None:
            if levels == 1:
                gamma_schedule = [0.1]
            elif levels == 2:
                gamma_schedule = [0.1, 0.3]
            else:
                gamma_schedule = [0.1 + 0.3 * i for i in range(levels)]
        assert len(gamma_schedule) == levels
        self.register_buffer(
            "base_gamma", torch.tensor(gamma_schedule, dtype=torch.float32)
        )
        self.gamma_param = nn.Parameter(torch.zeros(levels))
        self.eps = 1e-06

    def _dwt_grouped(self, x):
        B, C, H, W = x.shape
        if H < 2 or W < 2:
            return [x, x, x, x]
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(
                x,
                (0, 1 if W % 2 == 1 else 0, 0, 1 if H % 2 == 1 else 0),
                mode="reflect",
            )
        kernels = self.analysis_kernels.to(x.device).type(x.dtype)
        out = F.conv2d(x, weight=kernels, bias=None, stride=2, padding=0, groups=C)
        h2, w2 = (out.shape[-2], out.shape[-1])
        out = out.view(B, C, 4, h2, w2).permute(2, 0, 1, 3, 4)
        return [out[i].contiguous() for i in range(4)]

    def _idwt_grouped(self, LL, LH, HL, HH):
        device = LL.device
        dtype = LL.dtype
        C = LL.shape[1]
        if self.learnable_idwt:
            sk = self.synth_kernels.view(4, C, 1, 2, 2).to(device).type(dtype)
        else:
            sk = self.synth_kernels.to(device).type(dtype)
        out_LL = F.conv_transpose2d(
            LL, sk[0].view(C, 1, 2, 2), bias=None, stride=2, padding=0, groups=C
        )
        out_LH = F.conv_transpose2d(
            LH, sk[1].view(C, 1, 2, 2), bias=None, stride=2, padding=0, groups=C
        )
        out_HL = F.conv_transpose2d(
            HL, sk[2].view(C, 1, 2, 2), bias=None, stride=2, padding=0, groups=C
        )
        out_HH = F.conv_transpose2d(
            HH, sk[3].view(C, 1, 2, 2), bias=None, stride=2, padding=0, groups=C
        )
        return out_LL + out_LH + out_HL + out_HH

    def _compute_M_final(self, lvl, device, dtype):
        if self.per_level_W:
            W_raw = self.W_raw[lvl]
        else:
            W_raw = self.W_raw[0]
        W_raw = W_raw.to(device).type(dtype)
        base = self.mix_allowed_mask.to(device).type(dtype)
        M_param = self.M_param.to(device).type(dtype)
        learn_allowed = base * torch.sigmoid(M_param)
        allowed = learn_allowed
        off_indicator = (allowed > 0.0).to(dtype)
        off_logits = W_raw.clone()
        off_logits = off_logits * (1.0 - torch.eye(4, device=device, dtype=dtype))
        off_logits = off_logits * off_indicator
        row_soft = F.softmax(off_logits, dim=1)
        off_scaled = row_soft * allowed
        row_sums = off_scaled.sum(dim=1, keepdim=True)
        scale = torch.clamp(self.off_limit / (row_sums + self.eps), max=1.0)
        off_scaled = off_scaled * scale
        M_allowed = off_scaled.clone()
        M_allowed = M_allowed * (1.0 - torch.eye(4, device=device, dtype=dtype))
        M_allowed = M_allowed + torch.eye(4, device=device, dtype=dtype)
        gamma_raw = self.gamma_param[lvl]
        gamma_l = self.base_gamma[lvl] * torch.sigmoid(gamma_raw)
        gamma_l = gamma_l.to(device).type(dtype)
        I = torch.eye(4, device=device, dtype=dtype)
        M_final = (1.0 - gamma_l) * I + gamma_l * M_allowed
        return M_final

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        raw_subbands = []
        cur = x
        for lvl in range(self.levels):
            LL, LH, HL, HH = self._dwt_grouped(cur)
            raw_subbands.append((LL, LH, HL, HH))
            cur = LL
        processed_per_level = [None] * self.levels
        deeper_rec = None
        for lvl in reversed(range(self.levels)):
            LL, LH, HL, HH = raw_subbands[lvl]
            procs = self.level_processors[lvl]
            if deeper_rec is not None:
                convs = self.deeper_to_pseudos[lvl]
                pseudo_LL = convs[0](deeper_rec)
                pseudo_LH = convs[1](deeper_rec)
                pseudo_HL = convs[2](deeper_rec)
                pseudo_HH = convs[3](deeper_rec)
                if pseudo_LL.shape[-2:] != LL.shape[-2:]:
                    size = LL.shape[-2:]
                    pseudo_LL = F.interpolate(pseudo_LL, size=size, mode="area")
                    pseudo_LH = F.interpolate(pseudo_LH, size=size, mode="area")
                    pseudo_HL = F.interpolate(pseudo_HL, size=size, mode="area")
                    pseudo_HH = F.interpolate(pseudo_HH, size=size, mode="area")
                LL = LL + pseudo_LL
                LH = LH + pseudo_LH
                HL = HL + pseudo_HL
                HH = HH + pseudo_HH
            pLL = procs[0](LL)
            pLH = procs[1](LH)
            pHL = procs[2](HL)
            pHH = procs[3](HH)
            processed_per_level[lvl] = (pLL, pLH, pHL, pHH)
            X_stack = torch.stack([pLL, pLH, pHL, pHH], dim=0)
            M_final = self._compute_M_final(lvl, device, dtype)
            Y = torch.einsum("ij,jbchw->ibchw", M_final, X_stack)
            mLL, mLH, mHL, mHH = [Y[i].contiguous() for i in range(4)]
            deeper_rec = self._idwt_grouped(mLL, mLH, mHL, mHH)
        top_pLL, top_pLH, top_pHL, top_pHH = processed_per_level[0]
        X_top = torch.stack([top_pLL, top_pLH, top_pHL, top_pHH], dim=0)
        M_top = self._compute_M_final(0, device, dtype)
        Y_top = torch.einsum("ij,jbchw->ibchw", M_top, X_top)
        top_mLL, top_mLH, top_mHL, top_mHH = [Y_top[i].contiguous() for i in range(4)]
        recon = self._idwt_grouped(top_mLL, top_mLH, top_mHL, top_mHH)
        orig_LL, orig_LH, orig_HL, orig_HH = raw_subbands[0]
        shallow_rec = self._idwt_grouped(orig_LL, orig_LH, orig_HL, orig_HH)
        recon = recon + shallow_rec
        out = self.final_reduce(recon)
        return out


class ReflectanceBranch(nn.Module):

    def __init__(self, in_ch=300, out_ch=32):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, 2 * out_ch, 1, 1, bias=False)
        self.MLWCSFA = MWCSFA(in_ch=2 * out_ch, subband_ch=out_ch, levels=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = NNAD(out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch // 2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch // 2)
        self.act2 = NNAD(out_ch // 2)

    def forward(self, x):
        x1 = self.reduce(x)
        x2 = self.MLWCSFA(x1)
        x2 = self.bn1(x2)
        x2 = self.act1(x2)
        x3 = self.conv(x2)
        x3 = self.bn2(x3)
        x3 = self.act2(x3)
        return x3
