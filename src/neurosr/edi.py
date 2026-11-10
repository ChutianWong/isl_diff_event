"""Exposure-centred EDI and the original mEDI transfer and video priors."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import bounded_softplus, raw_from_bounded_softplus


def bin_events_pos_neg(ts_w, xs_w, ys_w, ps_w, sensor_size, num_bins, duration,
                       device=None):
    """Bilinearly splat events into temporal bins -> pos/neg count volumes.

    Differentiable bilinear splat (same architecture as
    utils.utility.interpolate_to_image): each event spreads its unit weight to
    the 4 surrounding pixels by sub-pixel weights, so gradients flow to the event
    coordinates (needed once events are warped by a learnable flow).  For integer
    DAVIS coordinates it reduces EXACTLY to nearest-pixel counting.

    Returns pos_bins, neg_bins : (num_bins, H, W) float tensors on `device`.
    """
    H, W = sensor_size
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(ts_w) == 0:
        z = torch.zeros(num_bins, H, W, device=device)
        return z, z.clone()

    t = torch.as_tensor(np.asarray(ts_w), dtype=torch.float32, device=device)
    x = torch.as_tensor(np.asarray(xs_w), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(ys_w), dtype=torch.float32, device=device)
    p = torch.as_tensor(np.asarray(ps_w), dtype=torch.float32, device=device)

    bin_idx = torch.clamp((t / max(float(duration), 1.0) * num_bins).long(),
                          0, num_bins - 1)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    dx = x - x0
    dy = y - y0
    x0 = x0.long()
    y0 = y0.long()

    pos_w = (p > 0).float()
    neg_w = 1.0 - pos_w

    pos_flat = torch.zeros(num_bins * H * W, device=device)
    neg_flat = torch.zeros(num_bins * H * W, device=device)
    for ox, oy, wgt in ((0, 0, (1 - dx) * (1 - dy)),
                        (1, 0, dx * (1 - dy)),
                        (0, 1, (1 - dx) * dy),
                        (1, 1, dx * dy)):
        xi = x0 + ox
        yi = y0 + oy
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        flat = (bin_idx * (H * W) + yi * W + xi)[valid]
        pos_flat.index_add_(0, flat, (wgt * pos_w)[valid])
        neg_flat.index_add_(0, flat, (wgt * neg_w)[valid])

    return pos_flat.view(num_bins, H, W), neg_flat.view(num_bins, H, W)


def edi_centered_double_integral(bii):
    """Centered double integral of brightness increments (MEDI core).

    bii  (2N, H, W) per-bin increments c_pos*E+ - c_neg*E-.
    Returns E (2N+1, H, W): log-intensity change RELATIVE TO THE EXPOSURE CENTRE
    (t = T/2).  Frame N is the reference (all-zero); left frames integrate
    backward, right frames forward.  Puts the latent reference at the middle of
    the exposure, which is numerically better conditioned than integrating from 0.
    """
    two_n = bii.shape[0]
    assert two_n % 2 == 0, "num_bins must be even for the centered double integral"
    n = two_n // 2
    left = [-bii[i:n].sum(dim=0) for i in range(n)]          # t < centre
    center = [torch.zeros_like(bii[0])]                      # t = centre
    right = [bii[n:n + 1 + i].sum(dim=0) for i in range(n)]  # t > centre
    return torch.stack(left + center + right, dim=0)


class ExposureEDI(nn.Module):
    """Differentiable MEDI model for one APS exposure window.

        E(t)   = centered double integral of  c_pos*E+(t) - c_neg*E-(t)
        I(t)   = I_center * exp(E(t))
        I_blur = mean_t I(t)   =>   I_center = I_blur / mean_t exp(E(t))

    The forward is ALWAYS the plain "rect" mean (optimisation path); smoothing is
    a render-only concern handled outside this class.  With shared_contrast=True a
    single threshold c is used (c_pos == c_neg), the paper's global-c model.
    """

    def __init__(self, pos_bins, neg_bins, init_c_pos=0.2, init_c_neg=0.2,
                 shared_contrast=True, edge_alpha=1.0, eps=1e-8):
        super().__init__()
        self.shared_contrast = bool(shared_contrast)
        self.edge_alpha = float(edge_alpha)
        self.eps = eps
        if self.shared_contrast:
            init_c = 0.5 * (float(init_c_pos) + float(init_c_neg))
            self.raw_c = nn.Parameter(raw_from_bounded_softplus(init_c))
        else:
            self.raw_c_pos = nn.Parameter(raw_from_bounded_softplus(init_c_pos))
            self.raw_c_neg = nn.Parameter(raw_from_bounded_softplus(init_c_neg))
        self.register_buffer('pos_bins', pos_bins.float())
        self.register_buffer('neg_bins', neg_bins.float())

    @property
    def c_pos(self):
        return bounded_softplus(self.raw_c if self.shared_contrast else self.raw_c_pos)

    @property
    def c_neg(self):
        return bounded_softplus(self.raw_c if self.shared_contrast else self.raw_c_neg)

    def event_integral(self):
        bii = self.c_pos * self.pos_bins - self.c_neg * self.neg_bins
        return edi_centered_double_integral(bii)

    def event_edge_reference(self):
        """Soft, temporally-weighted event-edge reference for the Sobel loss."""
        event_counts = self.pos_bins + self.neg_bins          # (2N, H, W)
        num_event_bins = event_counts.shape[0]
        num_video_frames = num_event_bins + 1
        frame_idx = torch.arange(num_video_frames, device=event_counts.device,
                                 dtype=event_counts.dtype)
        event_idx = torch.arange(num_event_bins, device=event_counts.device,
                                 dtype=event_counts.dtype) + 0.5
        weights = torch.exp(-self.edge_alpha *
                            torch.abs(frame_idx[:, None] - event_idx[None, :]))
        return torch.einsum("tb,bhw->thw", weights, event_counts)

    def forward(self, blurry):
        """Return (I_center, video, blur_pred) via the plain rect EDI inversion."""
        integral = self.event_integral().clamp(-20.0, 20.0)   # (2N+1, H, W)
        exp_integral = torch.exp(integral)
        I_center = (blurry / (exp_integral.mean(dim=0) + self.eps)).clamp_min(0.0)
        video = I_center.unsqueeze(0) * exp_integral
        blur_pred = video.mean(dim=0)                         # plain mean == blurry
        return I_center, video, blur_pred


def total_variation(image, eps=1e-6):
    if image.ndim == 2:
        image = image.unsqueeze(0)
    dy = image[..., 1:, :] - image[..., :-1, :]
    dx = image[..., :, 1:] - image[..., :, :-1]
    return torch.sqrt(dy.square() + eps).mean() + torch.sqrt(dx.square() + eps).mean()


def temporal_variation(video, eps=1e-6):
    dt = video[1:] - video[:-1]
    return torch.sqrt(dt.square() + eps).mean()


def sobel_magnitude(image, eps=1e-6):
    squeeze_time = image.ndim == 2
    if squeeze_time:
        image = image.unsqueeze(0)
    kx = image.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    ky = image.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    x = image.unsqueeze(1)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + eps).squeeze(1)
    return edge[0] if squeeze_time else edge


def edge_correlation_loss(video, event_edge_reference):
    image_edges = sobel_magnitude(video)
    event_edges = sobel_magnitude(event_edge_reference).detach()
    return -(image_edges * event_edges).mean()


def temporal_transfer_loss(I_ref_src, I_ref_dst, pos_transfer, neg_transfer, c_pos, c_neg):
    """Symmetric consistency loss that transports I_ref_src to the next window's
    centre and matches it against I_ref_dst.  This term is what fixes c.  Do NOT
    detach I_ref_dst -- both endpoints depend on c, and detaching biases the
    gradient's stationary point (verified ~1.7% off)."""
    integral = (c_pos * pos_transfer - c_neg * neg_transfer).clamp(-15.0, 15.0)
    I_transported = I_ref_src * torch.exp(integral)
    return torch.mean((I_transported - I_ref_dst) ** 2)


def render_blind_bridge(start_latent, end_latent, pos_blind, neg_blind, c_pos, c_neg):
    """Bridge a blind (inter-exposure) gap by overlap-blending a forward
    integration from window k's last frame and a backward integration from window
    k+1's first frame, so both junctions are continuous by construction."""
    bii = c_pos * pos_blind - c_neg * neg_blind
    E = torch.cumsum(bii, dim=0).clamp(-20.0, 20.0)          # (Bb, H, W)
    E_total = E[-1:]
    forward = start_latent.unsqueeze(0) * torch.exp(E)
    backward = end_latent.unsqueeze(0) * torch.exp(E - E_total)
    bb = bii.shape[0]
    w = torch.linspace(0.0, 1.0, bb + 2, device=bii.device)[1:-1].view(bb, 1, 1)
    return (1.0 - w) * forward + w * backward


def build_continuous_video(results, blind_pairs, c_pos, c_neg):
    """Concatenate per-exposure latent stacks + blind bridges into one gap-free,
    step-free video."""
    chunks = []
    for k, (I_center, video, blur_pred) in enumerate(results):
        chunks.append(video)
        if k < len(results) - 1 and blind_pairs[k] is not None:
            bridge = render_blind_bridge(
                video[-1], results[k + 1][1][0],
                blind_pairs[k]['pos'], blind_pairs[k]['neg'], c_pos, c_neg)
            chunks.append(bridge)
    return torch.cat(chunks, dim=0)
