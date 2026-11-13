"""Event channel: differential sensing of the motion-projected log-intensity gradient.

An event marks a step of +-c in L = log I at one pixel.  While the scene moves by D_k over
segment k, a pixel accumulates -grad L . D_k, so c times the image of warped events (IWE) of
that segment is a measurement of the transport-projected gradient of the latent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._constants import CONTRAST_0, MIN_FLOW_MAG, SENSOR, SIGMA, THRE
from .ops import (bounded_softplus, raw_from_bounded_softplus, lin_log, events_to_image_torch,
                  iwe_neg_var_loss, Rot_Flow_Gau_events_to_image_torch_sr_segments,
                  segment_iwe_from_log_image, l2_loss)


class EventModel(nn.Module):
    """The events of one window, their images and the residual against a latent intensity.

    L = lin_log(255 I).  The measurement of segment k is c times the transport-shaped splat of
    its warped events; the prediction is the finite step |D_k| [L(x) - L(x + u_k)] along the
    unit direction u_k = D_k / |D_k|.

    fidelity="contrast" compares the prediction with c IWE.  fidelity="normalized" compares
    the two after l2 normalization of each over the whole stack, so that their common
    multiplicative contrast scale cancels and c is not needed.
    """

    def __init__(self, xt, yt, tt, pt, sensor=SENSOR, sigma=SIGMA, contrast=CONTRAST_0, thre=THRE,
                 min_flow_mag=MIN_FLOW_MAG, fidelity="contrast"):
        super().__init__()
        self.xt, self.yt, self.tt, self.pt = xt, yt, tt, pt
        self.sensor = tuple(int(v) for v in sensor)
        self.sigma = float(sigma)
        self.thre = thre
        self.min_flow_mag = float(min_flow_mag)
        if fidelity not in ("contrast", "normalized"):
            raise ValueError("fidelity must be contrast or normalized")
        self.fidelity = fidelity
        self.raw_c = nn.Parameter(raw_from_bounded_softplus(contrast).to(xt.device))
        self._border = {}

    @property
    def device(self):
        return self.xt.device

    @property
    def contrast(self):
        return bounded_softplus(self.raw_c)

    def log(self, I):
        return lin_log(I * 255, self.thre)

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
        """Displacement of each event's segment, [N, 2]."""
        b = flow.segment_boundaries.to(self.tt.dtype)
        idx = torch.bucketize(self.tt, b[1:-1], right=True).clamp(0, flow.K - 1)
        return flow.segments()[idx]

    def direction(self, flow):
        """Unit direction of each event's own segment, [N, 2], detached."""
        D = self.local_segments(flow).detach()
        return D / D.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def target(self, xw, yw, flow, magnification, sigma=None):
        """Per-segment transport-shaped IWE on the pooling lattice, [K, M H, M W].

        Fine pixel M x + (M - 1) / 2 is the centre of sensor pixel x, so each event is rendered
        at x + (M - 1) / (2 M).  The finite forward step L(x) - L(x + u) is centred at
        x + u / 2, so the event moves by a further -u / (2 M), u its segment's unit direction.
        """
        M = int(magnification)
        sigma = self.sigma if sigma is None else float(sigma)
        xw, yw = xw + (M - 1) / (2 * M), yw + (M - 1) / (2 * M)
        u = self.direction(flow)
        xw, yw = xw - 0.5 * u[:, 0] / M, yw - 0.5 * u[:, 1] / M
        stack = Rot_Flow_Gau_events_to_image_torch_sr_segments(
            xw, yw, self.tt, self.pt, self.sensor, flow.segment_boundaries,
            M=M, Dxy_pc=flow.segments(), sigma=sigma, device=self.device, polar=True)
        return stack * self.border(M)[None]

    def predict(self, L, segments):
        """Per-segment prediction [K, H, W] from the log latent L [H, W], and its mask.

        |D_k| [L(x) - L(x + u_k)], with D_k [K, 2] in sensor px and u_k = D_k / |D_k|; the
        step takes one pixel of the current grid.  The mask drops |D_k| <= min_flow_mag and
        the one-pixel border."""
        return segment_iwe_from_log_image(L, segments, self.min_flow_mag)

    def residual(self, I, flow, target, contrast=None):
        """contrast: sum_k || pred_k - c IWE_k ||^2, scaled as the mean over pixels times K.
        normalized: || pred / |pred| - IWE / |IWE| ||^2 over the whole stack, summed, so the
        weight does not change with M."""
        pred, mask = self.predict(self.log(I), flow.segments())
        a, b = pred, target * mask
        if self.fidelity == "normalized":
            return (a / a.norm().clamp_min(1e-8) - b / b.norm().clamp_min(1e-8)).square().sum(), pred, mask
        c = self.contrast if contrast is None else contrast
        return l2_loss(a - b * c) * flow.K, pred, mask


def previous_event_indices(xt, yt, image_size):
    """For each event, the index of the previous event at the same sensor pixel, or -1."""
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
    """Bilinear samples of image I at the continuous positions (xt, yt), in pixels of I."""
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
