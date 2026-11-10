"""Sharp exposure in amplitude space, pooled by the original box operator."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._constants import *
from .ops import bounded_softplus, l2_loss, l1_loss, down_sampling, lin_log, time_window
from .edi import (ExposureEDI, bin_events_pos_neg, build_continuous_video, edge_correlation_loss,
                  temporal_transfer_loss, temporal_variation, total_variation)

class FrameModel(nn.Module):
    """I_blur = mean_t I(t) with I(t) = I_c exp(E(t)) from the exposure's own events; the MEDI
    latent I_c shares c with the event model, and the pooled reconstruction plus a background
    must match it.  edi=False: the exposure is short enough to be sharp, and the frame itself
    is the low-resolution observation of the latent.  domain="amplitude" compares square roots,
    as for a hologram, "log" lin_log(255 x) / log 255.  edi_loss="huber" compares with the
    latent by l1_loss_huber.

    reblur weights the APS term: the latent
    blurred by the exposure's kernel of the path, pooled, plus the background, against the frame
    itself; the residual is (1 - reblur) times the latent term plus reblur times that one (0.5 is
    the average, 1 the reblur term alone).  bg_beta is the
    background prior's l1 width."""

    def __init__(self, frame, xs, ys, ts, ps, win_frame, raw_c, sensor=SENSOR, num_bins=32,
                 align_exposure=False, edi=True, domain="intensity", reblur=0.0, edi_loss="l2", bg_beta=1e-5):
        super().__init__()
        if domain not in ("intensity", "amplitude", "log"):
            raise ValueError("frame domain must be intensity, amplitude or log")
        if not 0 <= reblur <= 1 or edi_loss not in ("l2", "huber"):
            raise ValueError("reblur is a weight in [0, 1] and edi_loss l2 or huber")
        self.frame, self.domain = frame, domain
        self.reblur, self.edi_loss, self.bg_beta = float(reblur), edi_loss, float(bg_beta)
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
        return self.frame if self.edi is None else self.edi(self.frame)[0]


    def _compare(self, a, b, loss="l2"):
        if self.domain == "amplitude":
            a, b = torch.sqrt(a + 1e-8), torch.sqrt(b + 1e-8)
        elif self.domain == "log":
            a, b = lin_log(a * 255) / np.log(255), lin_log(b * 255) / np.log(255)
        return l2_loss(a - b) if loss == "l2" else l1_loss_huber(a - b)

    def residual(self, I, magnification, latent=None, kernel=None):
        pooled = I if magnification == 1 else down_sampling(I, magnification)
        if self.reblur == 1:
            loss = 0.0
        else:
            loss = self._compare(pooled + self.background, self.latent() if latent is None else latent, self.edi_loss)
        if not self.reblur:
            return loss
        if kernel is None:
            raise ValueError("reblur needs the exposure kernel, FrameModel.kernel")
        blurred = blur_frame(I, kernel, conv_type="conv", device=I.device)
        blurred = blurred if magnification == 1 else down_sampling(blurred, magnification)
        return (1 - self.reblur) * loss + self.reblur * self._compare(blurred + self.background, self.frame)

    def prior(self):
        return l1_loss(self.background, beta=self.bg_beta)


class MultiFrameEDI(nn.Module):
    """mEDI centre-to-centre photometric transfer using the legacy EDI and transfer operators.

    frames are linear intensities in [0,1]; exposures and events use one clock in microseconds.
    This flow-free branch shares raw_c with EventModel. Its transfer residual constrains c;
    reblurring each analytically inverted exposure alone is an identity and cannot calibrate c.
    loss() adds the mEDI_debug priors on the EDI video: edge, the Sobel correlation of each
    exposure stack with its event edges, and tv/temporal_tv on the exposure stacks joined by
    blind-gap bridges binned at the exposure cadence from the gap start.  Zero weights leave
    the transfer residual alone.
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
