"""Frame channel: the exposure-averaged intensity that anchors absolute photometry.

A sharp exposure is itself the low-resolution observation of the latent.  A blurred exposure is
first referred to its centre by the event double integral (EDI), with the same contrast
threshold c as the event channel; consecutive exposures (mEDI) determine c.
"""
import numpy as np
import torch
import torch.nn as nn

from ._constants import SENSOR
from .ops import bounded_softplus, l2_loss, l1_loss, down_sampling, time_window
from .edi import (ExposureEDI, bin_events_pos_neg, build_continuous_video, edge_correlation_loss,
                  temporal_transfer_loss, temporal_variation, total_variation)


class FrameModel(nn.Module):
    """One exposure and the residual of a latent against it.

    edi=True: I_blur = mean_t I(t) with I(t) = I_c exp(E(t)) from the exposure's own events;
    the EDI latent I_c at the exposure centre shares c with the event model, and the pooled
    reconstruction plus a sensor-fixed background must match it.  edi=False: the exposure is
    short enough to be sharp, and the frame itself is the anchor.  domain="amplitude" compares
    square roots, as for a hologram.  bg_beta is the width of the background's sparsity prior.
    """

    def __init__(self, frame, xs, ys, ts, ps, win_frame, raw_c, sensor=SENSOR, num_bins=32,
                 align_exposure=False, edi=True, domain="intensity", bg_beta=1e-5):
        super().__init__()
        if domain not in ("intensity", "amplitude"):
            raise ValueError("frame domain must be intensity or amplitude")
        self.frame, self.domain = frame, domain
        self.bg_beta = float(bg_beta)
        self.win_frame = int(win_frame)
        if self.win_frame <= 0 or num_bins <= 0 or num_bins % 2:
            raise ValueError("EDI needs a positive exposure and a positive even bin count")
        self.edi = None
        if edi:
            xe, ye, te, pe = time_window(xs, ys, ts, ps, s=0, win=int(win_frame))
            origin = 0 if align_exposure or len(te) == 0 else te.min()
            pos, neg = bin_events_pos_neg(te - origin, xe, ye, pe, sensor, num_bins=num_bins,
                                          duration=float(win_frame), device=frame.device)
            self.edi = ExposureEDI(pos, neg, shared_contrast=True).to(frame.device)
            self.edi.raw_c = raw_c
        self.background = nn.Parameter(torch.zeros_like(frame).float())

    def latent(self):
        """The photometric anchor: the frame, or its EDI latent at the exposure centre."""
        return self.frame if self.edi is None else self.edi(self.frame)[0]

    def _compare(self, a, b):
        if self.domain == "amplitude":
            a, b = torch.sqrt(a + 1e-8), torch.sqrt(b + 1e-8)
        return l2_loss(a - b)

    def residual(self, I, magnification, latent=None):
        """Mean squared difference between the pixel-aperture average of I plus the background
        and the anchor."""
        pooled = I if magnification == 1 else down_sampling(I, magnification)
        return self._compare(pooled + self.background, self.latent() if latent is None else latent)

    def prior(self):
        return l1_loss(self.background, beta=self.bg_beta)


class MultiFrameEDI(nn.Module):
    """mEDI: photometric transfer between the centres of consecutive exposures.

    frames are linear intensities in [0, 1]; exposures and events use one clock in microseconds.
    Each exposure is inverted to its centre by EDI.  Carrying one centre latent to the next
    through the events between them constrains the shared threshold c: reblurring one
    analytically inverted exposure alone is an identity and cannot calibrate c.
    loss() adds three priors on the EDI video: edge, the Sobel correlation of each exposure
    stack with its event edges, and tv / temporal_tv on the exposure stacks joined across the
    gaps between exposures.  Zero weights leave the transfer residual alone.
    """

    def __init__(self, frames, exposures, xs, ys, ts, ps, raw_c, num_bins=32, tv=1e-3, temporal_tv=1e-4,
                 edge=1e-4):
        super().__init__()
        times = np.asarray(exposures, dtype=np.float64)
        if frames.ndim != 3 or len(frames) < 2 or times.shape != (len(frames), 2):
            raise ValueError("mEDI needs >=2 frames [F,H,W] and exposure intervals [F,2]")
        centers = times.mean(1)
        if np.any(times[:, 1] <= times[:, 0]) or np.any(np.diff(centers) <= 0):
            raise ValueError("Exposure durations and successive centre differences must be positive")
        self.raw_c = raw_c
        self.tv, self.temporal_tv, self.edge = float(tv), float(temporal_tv), float(edge)
        self.exposures = times.tolist()
        self.windows = nn.ModuleList([
            FrameModel(frames[i], xs, ys, np.asarray(ts) - a, ps, int(b - a), raw_c,
                       frames.shape[-2:], num_bins=num_bins, align_exposure=True)
            for i, (a, b) in enumerate(times)])
        for window in self.windows:
            window.background.requires_grad_(False)
        pos, neg = [], []
        for a, b in zip(centers[:-1], centers[1:]):
            take = (np.asarray(ts) > a) & (np.asarray(ts) <= b)
            p, n = bin_events_pos_neg(np.asarray(ts)[take] - a, np.asarray(xs)[take],
                                     np.asarray(ys)[take], np.asarray(ps)[take], frames.shape[-2:],
                                     num_bins=1, duration=b - a, device=frames.device)
            pos.append(p[0]); neg.append(n[0])
        self.register_buffer("pos_transfer", torch.stack(pos))
        self.register_buffer("neg_transfer", torch.stack(neg))
        self.blind = []
        for (a0, a), (b, _) in zip(times[:-1], times[1:]):
            xb, yb, tb, pb = time_window(*map(np.asarray, (xs, ys, ts, ps)), s=int(a), win=int(b - a))
            if b <= a or len(tb) == 0:
                self.blind.append(None)
                continue
            bins = max(1, int(round((b - a) / max((a - a0) / num_bins, 1.0))))
            p, n = bin_events_pos_neg(tb - int(a), xb, yb, pb, frames.shape[-2:], num_bins=bins,
                                      duration=int(b - a), device=frames.device)
            self.blind.append(dict(pos=p, neg=n))
        self.history = []

    def forward(self):
        return torch.stack([window.latent() for window in self.windows])

    def transfer(self, latents=None):
        I = self() if latents is None else latents
        c = bounded_softplus(self.raw_c)
        return torch.stack([temporal_transfer_loss(I[k], I[k + 1], p, n, c, c)
                            for k, (p, n) in enumerate(zip(self.pos_transfer, self.neg_transfer))]).mean()

    def loss(self):
        stacks = [window.edi(window.frame) for window in self.windows]
        loss = self.transfer(torch.stack([s[0] for s in stacks]))
        if self.edge:
            loss = loss + self.edge * torch.stack([edge_correlation_loss(s[1], window.edi.event_edge_reference())
                                                   for s, window in zip(stacks, self.windows)]).mean()
        if self.tv or self.temporal_tv:
            c = bounded_softplus(self.raw_c)
            video = build_continuous_video(stacks, self.blind, c, c)
            loss = loss + self.tv * total_variation(video) + self.temporal_tv * temporal_variation(video)
        return loss
