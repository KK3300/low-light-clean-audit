"""Approximately 1M-parameter curve enhancer and preregistered controls.

The seven-stage topology follows the spatially compact Zero-DCE++ curve
predictor, but the default Stage 2 backbone uses ordinary 3x3 convolutions
with width 128.  This is a new experiment package; legacy HAM-DCE code and
checkpoints are not modified or loaded here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = {
    "plain",
    "gap_gate",
    "variance_gate",
    "bright_gate",
    "curve_residual",
    "rcm",
    "output_residual",
    "rcm_output",
    "budgeted_output",
    "ms_refine",
    "ms_refine_uncond",
    "ms_refine_dark",
    "ms_refine_reliability",
}


class CurveBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, conv_mode: str = "standard"):
        super().__init__()
        mode = conv_mode.lower()
        if mode not in {"standard", "depthwise"}:
            raise ValueError(f"unknown conv_mode={conv_mode}")
        if mode == "standard":
            self.op = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=True)
        else:
            self.op = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1, groups=in_ch, bias=True),
                nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ChannelGate(nn.Module):
    """Four-parameter gate used for ordinary-GAP and variance controls."""

    def __init__(self, mode: str):
        super().__init__()
        if mode not in {"gap", "variance"}:
            raise ValueError(mode)
        self.mode = mode
        self.channel_filter = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, _darkness: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "gap":
            z = x.mean(dim=(2, 3), keepdim=True)
        else:
            mean = x.mean(dim=(2, 3), keepdim=True)
            z = (x - mean).square().mean(dim=(2, 3), keepdim=True)
        z = z.squeeze(-1).transpose(1, 2)  # B,1,C
        response = self.channel_filter(z).transpose(1, 2).unsqueeze(-1)
        return x * (1.0 + self.alpha * torch.tanh(response))


class BrightReferenceGate(ChannelGate):
    """GAP-like control weighted toward reliable bright input regions."""

    def __init__(self):
        super().__init__("gap")

    def forward(self, x: torch.Tensor, darkness: Optional[torch.Tensor] = None) -> torch.Tensor:
        if darkness is None:
            return super().forward(x, darkness)
        bright = (1.0 - darkness).clamp(0.0, 1.0)
        denom = bright.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        z = (x * bright).sum(dim=(2, 3), keepdim=True) / denom
        z = z.squeeze(-1).transpose(1, 2)
        response = self.channel_filter(z).transpose(1, 2).unsqueeze(-1)
        return x * (1.0 + self.alpha * torch.tanh(response))


class CurveResidual(nn.Module):
    """40 trainable parameters: depthwise 3x3 + pointwise 1x1 + scalar."""

    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        self.pointwise = nn.Conv2d(3, 3, 1, bias=True)
        self.eta = nn.Parameter(torch.zeros(()))

    def forward(self, logits: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        delta = self.pointwise(self.depthwise(logits))
        return logits + self.eta * condition * delta


class OutputResidual(nn.Module):
    """Same small residual capacity as CurveResidual, but applied in RGB."""

    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        self.pointwise = nn.Conv2d(3, 3, 1, bias=True)
        self.eta = nn.Parameter(torch.zeros(()))

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        delta = self.pointwise(self.depthwise(image))
        return image + self.eta * condition * delta


class BudgetedOutputResidual(OutputResidual):
    """Reliability-gated residual with a fixed per-image perturbation budget.

    The normalization prevents a large raw residual in a small bright region
    from dominating the whole image.  ``budget`` is a protocol constant, not a
    learned quality target; the module remains a 40-parameter control.
    """

    def __init__(self, budget: float = 0.04):
        super().__init__()
        self.budget = float(budget)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        delta = self.pointwise(self.depthwise(image))
        weighted = (delta.abs() * condition).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        scale = self.budget * torch.tanh(self.eta) / weighted
        return image + delta * condition * scale


class MultiScaleRefinement(nn.Module):
    """Small quarter-resolution RGB refinement branch."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1, bias=True),
        )
        # Start as an exact no-op; this separates refinement learning from
        # the base curve predictor during attribution.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        # The final convolution is zero-initialized, so the branch is still an
        # exact no-op at initialization.  A nonzero scalar keeps gradients
        # flowing into that final convolution on the first update.
        self.eta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        h = max(1, (x.shape[-2] + 3) // 4)
        w = max(1, (x.shape[-1] + 3) // 4)
        low = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        residual = self.body(low)
        residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # Keep a small floor so bright, reliable edges are not completely
        # frozen while still down-weighting highly uncertain dark regions.
        gate = 0.25 + 0.75 * reliability
        return x + torch.tanh(self.eta) * gate * residual


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "plain"
    width: int = 128
    conv_mode: str = "standard"
    scale_factor: int = 12
    curve_steps: int = 8


def _local_variance(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    lum = x.mean(dim=1, keepdim=True)
    mean = F.avg_pool2d(lum, kernel, stride=1, padding=kernel // 2)
    mean2 = F.avg_pool2d(lum.square(), kernel, stride=1, padding=kernel // 2)
    return (mean2 - mean.square()).clamp_min(0.0)


class RCMCurveNet(nn.Module):
    """Curve enhancer with preregistered Stage 2 variants.

    ``rcm`` uses a deterministic reliability map derived from darkness,
    local variance and saturation risk.  It has no learned reliability
    estimator; the only learned residual is the 40-parameter curve residual.
    ``curve_residual`` uses an all-ones condition and is the capacity control.
    """

    def __init__(
        self,
        variant: str = "plain",
        width: int = 128,
        conv_mode: str = "standard",
        scale_factor: int = 12,
        curve_steps: int = 8,
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        self.cfg = ModelConfig(variant, int(width), conv_mode.lower(), int(scale_factor), int(curve_steps))
        n = self.cfg.width
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = CurveBlock(3, n, self.cfg.conv_mode)
        self.e_conv2 = CurveBlock(n, n, self.cfg.conv_mode)
        self.e_conv3 = CurveBlock(n, n, self.cfg.conv_mode)
        self.e_conv4 = CurveBlock(n, n, self.cfg.conv_mode)
        self.e_conv5 = CurveBlock(n * 2, n, self.cfg.conv_mode)
        self.e_conv6 = CurveBlock(n * 2, n, self.cfg.conv_mode)
        self.e_conv7 = CurveBlock(n * 2, 3, self.cfg.conv_mode)

        self.gate4: Optional[nn.Module] = None
        self.gate6: Optional[nn.Module] = None
        if variant == "gap_gate":
            self.gate4 = ChannelGate("gap")
            self.gate6 = ChannelGate("gap")
        elif variant == "variance_gate":
            self.gate4 = ChannelGate("variance")
            self.gate6 = ChannelGate("variance")
        elif variant == "bright_gate":
            self.gate4 = BrightReferenceGate()
            self.gate6 = BrightReferenceGate()

        self.curve_residual: Optional[nn.Module] = None
        self.output_residual: Optional[nn.Module] = None
        if variant in {"curve_residual", "rcm"}:
            self.curve_residual = CurveResidual()
        if variant in {"output_residual", "rcm_output"}:
            self.output_residual = OutputResidual()
        elif variant == "budgeted_output":
            self.output_residual = BudgetedOutputResidual()
        # Refinement attribution variants deliberately share the exact same
        # branch (and therefore parameter count) as ``ms_refine``.  Only the
        # deterministic spatial gate differs:
        #   * uncond       : gate is identically one (capacity control)
        #   * dark         : darkness proxy only
        #   * reliability  : darkness × variance × chroma reliability map
        self.multiscale_refinement: Optional[nn.Module] = (
            MultiScaleRefinement(32)
            if variant in {"ms_refine", "ms_refine_uncond", "ms_refine_dark", "ms_refine_reliability"}
            else None
        )

    @staticmethod
    def darkness_proxy(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x.mean(dim=1, keepdim=True)).clamp(0.0, 1.0)

    @staticmethod
    def reliability_map(x: torch.Tensor) -> torch.Tensor:
        """Input-derived reliability: dark but not saturated/noisy regions.

        This is a deliberately transparent proxy, not a physical illumination
        map or a learned segmentation.  The normalization constants are fixed
        for the pilot and must be treated as protocol values.
        """
        lum = x.mean(dim=1, keepdim=True)
        darkness = (1.0 - lum).clamp(0.0, 1.0)
        var = _local_variance(x).sqrt().clamp(0.0, 0.5) / 0.5
        sat = ((x.max(dim=1, keepdim=True).values - x.min(dim=1, keepdim=True).values).clamp(0.0, 1.0))
        # High local variance and strong chroma can be unreliable for a curve
        # correction; keep a nonzero floor to avoid discontinuous masks.
        return (darkness * (1.0 - 0.5 * var) * (1.0 - 0.25 * sat)).clamp(0.0, 1.0)

    def enhance(self, x: torch.Tensor, curve: torch.Tensor) -> torch.Tensor:
        out = x
        for _ in range(self.cfg.curve_steps):
            out = out + curve * (out.square() - out)
        return out.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if self.cfg.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1.0 / self.cfg.scale_factor, mode="bilinear", align_corners=False)
        p_down = self.darkness_proxy(x_down)
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        if self.gate4 is not None:
            x4 = self.gate4(x4, p_down)
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        if self.gate6 is not None:
            x6 = self.gate6(x6, p_down)
        logits = self.e_conv7(torch.cat([x1, x6], dim=1))
        cond = torch.ones_like(p_down)
        if self.cfg.variant in {"rcm", "rcm_output", "budgeted_output"}:
            cond = self.reliability_map(x_down)
        if self.curve_residual is not None:
            logits = self.curve_residual(logits, cond)
        curve = torch.tanh(logits)
        if self.cfg.scale_factor != 1:
            curve = F.interpolate(curve, size=x.shape[-2:], mode="bilinear", align_corners=False)
        enhanced = self.enhance(x, curve)
        if self.output_residual is not None:
            out_cond = F.interpolate(cond, size=enhanced.shape[-2:], mode="bilinear", align_corners=False)
            enhanced = (enhanced + self.output_residual(enhanced, out_cond) - enhanced).clamp(0.0, 1.0)
        if self.multiscale_refinement is not None:
            # Keep the ablation masks input-derived and parameter-free.  The
            # branch itself is identical across all four variants, so any
            # difference can be attributed to the gate rather than capacity.
            if self.cfg.variant == "ms_refine_uncond":
                refine_gate = torch.ones_like(p_down)
            elif self.cfg.variant == "ms_refine_dark":
                refine_gate = p_down
            else:  # ms_refine / ms_refine_reliability
                refine_gate = self.reliability_map(x)
            refine_gate = F.interpolate(refine_gate, size=enhanced.shape[-2:], mode="bilinear", align_corners=False)
            enhanced = self.multiscale_refinement(enhanced, refine_gate).clamp(0.0, 1.0)
        if return_aux:
            return enhanced, curve, {
                "darkness": F.interpolate(p_down, size=x.shape[-2:], mode="bilinear", align_corners=False),
                "reliability": F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False),
                "logits": logits,
            }
        return enhanced, curve


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_signature(model: RCMCurveNet) -> Dict[str, object]:
    return {
        "architecture": "RCMCurveNet",
        **asdict(model.cfg),
        "parameters": count_parameters(model),
    }
