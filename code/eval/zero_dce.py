"""Protocol-matched reimplementations of Zero-DCE and Zero-DCE++.

Why this file exists
--------------------
The manuscript had no quantitative comparison against a published learned
method, which is the most likely single reason for rejection at a Q3 venue.
Downloading the third-party repositories proved slow and unreliable in this
environment, so both baselines are reimplemented here and then trained on *our*
clean training split and scored with *our* metric implementation.  The result is
a protocol-matched comparison rather than a cross-paper number copy.

Architecture provenance
-----------------------
The layer topology, channel widths and skip-connection pattern follow the
official reference definitions fetched from the authors' repositories (see
`third_party/official_ref/`, retained for provenance):

  * Zero-DCE  (CVPR 2020): number_f = 32; conv1..conv6 are 3x3 with widths
    32,32,32,32,32,32 -> e_conv5 takes cat[x3,x4], e_conv6 takes cat[x2,x5],
    e_conv7 takes cat[x1,x6] and outputs 24 = 8 iterations x 3 channels.
  * Zero-DCE++ (TPAMI 2022): identical topology but every convolution is
    depthwise-separable, e_conv7 outputs **3** channels (one curve reused for
    all eight iterations -- this is what makes it ~10K parameters), and the
    curve predictor runs on a 1/scale_factor downscaled input.

The forward passes, curve application order and output splitting are written
here rather than copied, and every model is validated against the published
parameter counts before use:

    Zero-DCE     target ~79,416
    Zero-DCE++   target ~10,561
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDN(nn.Module):
    """Depthwise-separable 3x3 convolution (official CSDN_Tem equivalent)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, groups=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class DCENet(nn.Module):
    """Zero-DCE curve predictor: 7 standard 3x3 convs, 8 curve iterations."""

    def __init__(self, number_f: int = 32, curve_steps: int = 8):
        super().__init__()
        self.curve_steps = curve_steps
        self.relu = nn.ReLU(inplace=True)
        f = number_f
        self.e_conv1 = nn.Conv2d(3, f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(f * 2, f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(f * 2, f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(f * 2, 3 * curve_steps, 3, 1, 1, bias=True)

    def forward(self, x, return_maps: bool = False):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        r = torch.split(x_r, 3, dim=1)
        out = x
        for i in range(self.curve_steps):
            out = out + r[i] * (out * out - out)
        out = out.clamp(0.0, 1.0)
        if return_maps:
            return out, x_r
        return out


class DCENetPlusPlus(nn.Module):
    """Zero-DCE++: depthwise-separable, single 3-channel curve, downscaled input."""

    def __init__(self, number_f: int = 32, curve_steps: int = 8, scale_factor: int = 12):
        super().__init__()
        self.curve_steps = curve_steps
        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        f = number_f
        self.e_conv1 = CSDN(3, f)
        self.e_conv2 = CSDN(f, f)
        self.e_conv3 = CSDN(f, f)
        self.e_conv4 = CSDN(f, f)
        self.e_conv5 = CSDN(f * 2, f)
        self.e_conv6 = CSDN(f * 2, f)
        self.e_conv7 = CSDN(f * 2, 3)          # one curve, reused for all iterations

    def forward(self, x, return_maps: bool = False):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1.0 / self.scale_factor, mode="bilinear")

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        # Interpolate to the exact input geometry.  The fixed-factor
        # UpsamplingBilinear2d used in the reference code can land one pixel
        # short when the input is not divisible by scale_factor.
        if self.scale_factor != 1 and x_r.shape[-2:] != x.shape[-2:]:
            x_r = F.interpolate(x_r, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = x
        for _ in range(self.curve_steps):
            out = out + x_r * (out * out - out)
        out = out.clamp(0.0, 1.0)
        if return_maps:
            return out, x_r
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# published reference counts used as an implementation check
TARGETS = {"Zero-DCE": 79416, "Zero-DCE++": 10561}


def validate() -> bool:
    a = count_params(DCENet(32, 8))
    b = count_params(DCENetPlusPlus(32, 8, 12))
    ok = True
    print("Zero-DCE    params = %6d   (published ~%d)  %s"
          % (a, TARGETS["Zero-DCE"], "MATCH" if abs(a - TARGETS["Zero-DCE"]) < 200 else "MISMATCH"))
    print("Zero-DCE++  params = %6d   (published ~%d)  %s"
          % (b, TARGETS["Zero-DCE++"], "MATCH" if abs(b - TARGETS["Zero-DCE++"]) < 200 else "MISMATCH"))
    if abs(a - TARGETS["Zero-DCE"]) >= 200:
        ok = False
    if abs(b - TARGETS["Zero-DCE++"]) >= 200:
        ok = False

    x = torch.rand(2, 3, 64, 64)
    for name, m in [("Zero-DCE", DCENet()), ("Zero-DCE++", DCENetPlusPlus())]:
        y = m(x)
        rng = (y.min().item(), y.max().item())
        print("  %-12s out=%s range=(%.3f, %.3f)" % (name, tuple(y.shape), *rng))
        if not (0.0 <= rng[0] and rng[1] <= 1.0):
            ok = False
    return ok


if __name__ == "__main__":
    good = validate()
    print("\nIMPLEMENTATION CHECK:", "PASS" if good else "FAIL")
    raise SystemExit(0 if good else 1)
