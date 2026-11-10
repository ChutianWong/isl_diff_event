"""Differential sensing with normalized, per-segment finite-step IWEs.

Internal signatures preserve source defaults. Only the configured dh path is supported.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._constants import *
from .ops import (bounded_softplus, raw_from_bounded_softplus, lin_log, events_to_image_torch,
    iwe_neg_var_loss, Rot_Flow_Gau_events_to_image_torch_sr_segments, segment_iwe_from_log_image, l2_loss)

class EventModel(nn.Module):
    """L = lin_log(255 I); over one segment an event pixel accumulates -grad L . D, and the
    measurement is c times the flow-shaped splat of the warped events of that segment.
    flow_scale divides both sides by the largest displacement of the sensor-grid field.
    support_density (events per pixel) scales each pixel's
    residual by min(rho / support_density, 1), rho the support_box mean of the warped event count,
    so that pixels without events are left to the frame; None keeps the plain residual.  The last
    target rendered at M supplies the support weight for residuals at M.

    fidelity="pairs" is the event-wise route: each event and the previous
    event at its sensor pixel, both warped to t0, must differ in L by p c, with L bilinear at M x
    (pair_centre: M x + (M - 1) / 2, the pooling lattice); huber is the Huber width,
    None the L2.  prf_fill > 0 averages the latent over a centred
    aperture of that linear width in pitches before L at M > 1 (1 is the full
    pitch)."""

    def __init__(self, xt, yt, tt, pt, sensor=SENSOR, sigma=SIGMA, contrast=CONTRAST_0, thre=THRE,
                 min_flow_mag=MIN_FLOW_MAG, stencil="forward", half_shift=True, polar=True,
                 forward_model="finite", fidelity="contrast", flow_scale=False, support_density=None,
                 support_box=15, prf_fill=0.0, huber=None, pair_centre=False):
        super().__init__()
        self.xt, self.yt, self.tt, self.pt = xt, yt, tt, pt
        self.sensor = tuple(int(v) for v in sensor)
        self.sigma = float(sigma)
        self.thre = thre
        self.min_flow_mag = float(min_flow_mag)
        self.stencil = stencil
        self.half_shift = bool(half_shift)
        self.polar = bool(polar)
        self.flow_scale = bool(flow_scale)
        if forward_model not in ("finite", "eklt") or fidelity not in ("contrast", "normalized", "pairs"):
            raise ValueError("event model must be finite/eklt and fidelity contrast/normalized/pairs")
        self.forward_model, self.fidelity = forward_model, fidelity
        if support_density is not None and (support_density <= 0 or support_box < 1 or support_box % 2 == 0):
            raise ValueError("support_density must be positive and support_box a positive odd width")
        if fidelity != "pairs" and (huber is not None or pair_centre):
            raise ValueError("huber and pair_centre belong to the pairs fidelity")
        if fidelity == "pairs" and (support_density is not None or flow_scale):
            raise ValueError("the pairs fidelity has no support weight or flow scale")
        if not 0 <= prf_fill <= 1:
            raise ValueError("prf_fill is a linear width in [0, 1] pitches")
        self.support_density = None if support_density is None else float(support_density)
        self.support_box = int(support_box)
        self.prf_fill = float(prf_fill)
        self.huber = None if huber is None else float(huber)
        self.pair_centre = bool(pair_centre)
        self.raw_c = nn.Parameter(raw_from_bounded_softplus(contrast).to(xt.device))
        self._border, self._support, self._focus_original = {}, {}, None

    @property
    def device(self):
        return self.xt.device

    @property
    def contrast(self):
        return bounded_softplus(self.raw_c)

    def log(self, I):
        return lin_log(I * 255, self.thre)

    def response(self, I, magnification):
        """Linear intensity seen by an event pixel on the M-times finer grid: a centred aperture of active
        width prf_fill pitches, weights its overlap with each fine cell (prf_fill = 1 is the flat
        M-tap mean, <= 1 / M the point sample).
        The point sample at M = 1 or prf_fill = 0."""
        M = int(magnification)
        if not self.prf_fill or M == 1:
            return I
        a = self.prf_fill * M
        r = int(np.ceil((a - 1.0) / 2.0)) if a > 1.0 else 0
        k = torch.arange(-r, r + 1, device=I.device, dtype=I.dtype)
        half = torch.as_tensor(a / 2.0, device=I.device, dtype=I.dtype)
        w = torch.clamp(torch.minimum(k + 0.5, half) - torch.maximum(k - 0.5, -half), min=0.0)
        kernel = torch.outer(w / w.sum(), w / w.sum())[None, None]
        return F.conv2d(F.pad(I[None, None], (r,) * 4, mode="replicate"), kernel)[0, 0]


    def border(self, magnification):
        if magnification not in self._border:
            mask = torch.zeros(self.sensor[0] * magnification, self.sensor[1] * magnification, device=self.device)
            mask[1:-1, 1:-1] = 1
            self._border[magnification] = mask
        return self._border[magnification]

    def count(self, xw, yw, signed=False):
        """Bilinear event image of the warped events on the sensor grid."""
        return events_to_image_torch(xw, yw, self.pt, self.sensor, device=self.device, polar=signed)

    def focus(self, xw, yw):
        """Contrast maximisation: minus the variance of the count image."""
        return iwe_neg_var_loss(self.count(xw, yw))


    def local_segments(self, flow):
        """Displacement of each event's segment, sampled at the source pixel, [N,2]."""
        b = flow.segment_boundaries.to(self.tt.dtype)
        idx = torch.bucketize(self.tt, b[1:-1], right=True).clamp(0, flow.K - 1)
        if flow.uniform:
            return flow.segments()[idx]
        return torch.einsum("nk,nkc->nc", flow.A[idx], flow.sample(self.xt, self.yt))

    def direction(self, flow):
        """Unit direction of each event's own segment at its own pixel, [N, 2], detached."""
        D = self.local_segments(flow).detach()
        return D / D.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def support(self, xw, yw, magnification=1):
        """Residual weight min(rho / support_density, 1) on the M-times finer grid, rho the support_box
        mean of the sensor-grid count of the warped events; None without support_density."""
        if self.support_density is None:
            return None
        with torch.no_grad():
            rho = F.avg_pool2d(self.count(xw.detach(), yw.detach())[None, None], self.support_box, stride=1,
                               padding=self.support_box // 2, count_include_pad=False)[0, 0]
            weight = (rho / self.support_density).clamp(max=1.0)
        M = int(magnification)
        return weight.repeat_interleave(M, 0).repeat_interleave(M, 1) if M > 1 else weight

    def target(self, xw, yw, flow, magnification, sigma=None):
        """Per-segment flow-shaped IWE on the pooling lattice, [K, M H, M W].

        Fine pixel M x + (M - 1) / 2 is the centre of sensor pixel x, so each event is rendered at
        x + (M - 1) / (2 M).  The finite forward stencil L(x) - L(x + u) is centred at x + u / 2, so
        with it and half_shift the event moves by a further -u / (2 M), u its segment's unit
        direction.  The support weight of the same warped events is kept for residual at this M.
        The pairs fidelity needs no image: its target is the warped positions, [2, N] in sensor px.
        """
        M = int(magnification)
        self._support[M] = self.support(xw, yw, M)
        sigma = self.sigma if sigma is None else float(sigma)
        xw, yw = xw + (M - 1) / (2 * M), yw + (M - 1) / (2 * M)
        if self.forward_model == "finite" and self.half_shift and self.stencil == "forward":
            u = self.direction(flow)
            xw, yw = xw - 0.5 * u[:, 0] / M, yw - 0.5 * u[:, 1] / M
        if flow.uniform:
            stack = Rot_Flow_Gau_events_to_image_torch_sr_segments(
                xw, yw, self.tt, self.pt, self.sensor, flow.segment_boundaries,
                M=M, Dxy_pc=flow.segments(), sigma=sigma, device=self.device, polar=self.polar)
        return stack * self.border(M)[None]


    def predict(self, L, segments):
        """Per-segment prediction [K, H, W] from the log latent L [H, W], and its mask.

        forward |D_k| [L(x) - L(x + u_k)], centered |D_k| [L(x - u_k / 2) - L(x + u_k / 2)], eklt
        -grad L . D_k, with D_k [K, 2] or [K, 2, H, W] in sensor px and u_k = D_k / |D_k|; the
        finite step and the forward gradient take one pixel of the current grid.  The mask drops
        |D_k| <= min_flow_mag and the one-pixel border."""
        return segment_iwe_from_log_image(L, segments, self.min_flow_mag, stencil=self.stencil)

    def residual(self, I, flow, magnification, target, contrast=None):
        """sum_k || pred_k - c IWE_k ||^2 scaled as the mean over pixels times K.
        normalized: || pred / |pred| - IWE / |IWE| ||^2 over the whole stack, summed, so the
        weight does not change with M.  With
        support_density both sides carry the support weight that target recorded at this M.
        pairs: the pair residual, with target the warped positions."""
        pred, mask = self.predict(self.log(self.response(I, magnification)), flow.segments(magnification))
        a, b = pred, target * mask
        if self.support_density is not None:
            if self._support.get(int(magnification)) is None:
                raise RuntimeError("support_density needs the target rendered by EventModel.target at this M")
            weight = self._support[int(magnification)]
            a, b = a * weight, b * weight
        if self.flow_scale:
            scale = torch.linalg.vector_norm(flow.segments(1), dim=1).amax() + 1e-9
            a, b = a / scale, b / scale
        if self.fidelity == "normalized":
            return (a / a.norm().clamp_min(1e-8) - b / b.norm().clamp_min(1e-8)).square().sum(), pred, mask
        c = self.contrast if contrast is None else contrast
        return l2_loss(a - b * c) * flow.K, pred, mask


def previous_event_indices(xt, yt, image_size):
    """For each event, return the previous event at the same sensor pixel."""
    _, width = image_size
    pixel_id = yt.long() * width + xt.long()

    order = torch.argsort(pixel_id, stable=True)
    pixel_id_sorted = pixel_id[order]
    same_pixel = pixel_id_sorted[1:] == pixel_id_sorted[:-1]

    prev_idx = torch.full_like(order, -1)
    event_idx = order[1:][same_pixel]
    prev_idx[event_idx] = order[:-1][same_pixel]
    return prev_idx


def sample_image_at_events(I, xt, yt):
    H, W = I.shape

    xt = xt.to(device=I.device, dtype=I.dtype)
    yt = yt.to(device=I.device, dtype=I.dtype)

    gx = xt / (W - 1) * 2.0 - 1.0
    gy = yt / (H - 1) * 2.0 - 1.0

    grid = torch.stack([gx, gy], dim=-1)
    grid = grid.view(1, 1, -1, 2)

    sampled = F.grid_sample(
        I[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    return sampled.view(-1)
