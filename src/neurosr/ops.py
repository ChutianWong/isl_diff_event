"""Numerical building blocks: event images, the log response, losses and priors, registration.

Arrays are indexed [y, x]; event coordinates are (x, y) in sensor pixels; times are microseconds.
Functions derived from third-party code are marked where they are defined; see
THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import gc
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from skimage.registration import phase_cross_correlation

C_MIN, C_MAX = 1e-3, 0.5      # admissible range of the contrast threshold c


def time_window(xs, ys, ts, ps, s, win):
    """The events with s <= t <= s + win."""
    loc = np.where((ts <= s + win) & (ts >= s))
    ts, xs, ys, ps = ts[loc], xs[loc], ys[loc], ps[loc]
    return xs, ys, ts, ps


def np_to_torch(xs, ys, ts, ps, device):
    """Float tensors on device; the time axis restarts at the first event."""
    xt, yt, tt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ts), torch.from_numpy(ps)
    xt = xt.float().to(device)
    yt = yt.float().to(device)
    tt0 = tt[0]
    tt = (tt[:] - tt0).float().to(device)
    pt = pt.float().to(device)
    return xt, yt, tt, pt


# Derived from TimoStoff/event_utils (MIT), locally modified; see THIRD_PARTY_NOTICES.md
def interpolate_to_image(pxs, pys, dxs, dys, interp_weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img_h, img_w = img.shape  # Get image dimensions
    valid_mask = (pxs >= 0) & (pxs < img_w-1) & (pys >= 0) & (pys < img_h-1)
    pxs = pxs[valid_mask]
    pys = pys[valid_mask]
    dxs = dxs[valid_mask]
    dys = dys[valid_mask]
    interp_weights = interp_weights[valid_mask]

    img.index_put_((pys, pxs), interp_weights * (1.0 - dxs) * (1.0 - dys), accumulate=True)
    img.index_put_((pys, pxs + 1), interp_weights * dxs * (1.0 - dys), accumulate=True)
    img.index_put_((pys + 1, pxs), interp_weights * (1.0 - dxs) * dys, accumulate=True)
    img.index_put_((pys + 1, pxs + 1), interp_weights * dxs * dys, accumulate=True)

    return img


def gaussian_interpolate_to_image_rot_flow(
    pxs, pys,
    dxs, dys,
    interp_weights,
    img,
    sigma_along,
    sigma_across,
    flow_dir,
    kernel_size=(7, 7),
):
    """
    Accumulate events into img with a Gaussian whose axes follow the flow (per-event normalized).

    Each event at integer pixel (pxs, pys) with sub-pixel offset (dxs, dys) spreads its weight
    interp_weights over a kernel_size window.  sigma_along is the standard deviation along
    flow_dir and sigma_across the one perpendicular to it, both in pixels of img.  flow_dir is a
    length-2 tensor (x, y); its norm and sign do not matter, and a zero vector falls back to the
    x axis.  Differentiable in dxs, dys, interp_weights, the two widths and flow_dir.
    Contributions outside the image are clipped without renormalization.  The window is square
    in image axes, so an oblique ellipse is truncated less than an axis-aligned one.
    """
    device = pxs.device
    img_h, img_w = img.shape

    # 1. Kernel offsets (integer for indexing, real for the weights)
    kx, ky = kernel_size
    rx = (kx - 1) // 2
    ry = (ky - 1) // 2
    offsets_xi = torch.arange(-rx, rx + 1, device=device)  # [Kx]
    offsets_yi = torch.arange(-ry, ry + 1, device=device)  # [Ky]
    offsets_x = offsets_xi.to(dxs.dtype)
    offsets_y = offsets_yi.to(dys.dtype)
    Kx = offsets_xi.numel()
    Ky = offsets_yi.numel()
    K2 = Kx * Ky

    # 2. Rotated Gaussian kernel (per-event), axes along / across the flow
    u = torch.as_tensor(flow_dir).to(device=device, dtype=dxs.dtype).reshape(2)
    if float(torch.linalg.norm(u.detach())) <= 1e-12:
        u = torch.tensor([1.0, 0.0], device=device, dtype=dxs.dtype)
    u = u / (torch.linalg.norm(u) + 1e-12)

    ddx = (offsets_x[None, :] - dxs[:, None])[:, :, None]   # pixel minus event, x  [N, Kx, 1]
    ddy = (offsets_y[None, :] - dys[:, None])[:, None, :]   # pixel minus event, y  [N, 1, Ky]
    along = ddx * u[0] + ddy * u[1]
    across = -ddx * u[1] + ddy * u[0]
    w = torch.exp(
        -0.5 * (along / sigma_along) ** 2
        - 0.5 * (across / sigma_across) ** 2
    )                                                       # [N, Kx, Ky]

    # Per-event normalization in the kernel domain.
    w = w / (w.sum(dim=(1, 2), keepdim=True) + 1e-12)

    # 3. Pixel coordinates (expand to event-pixel domain)
    pxs_b = pxs[:, None] + offsets_xi[None, :]     # [N, Kx]
    pys_b = pys[:, None] + offsets_yi[None, :]     # [N, Ky]

    pxs_b = pxs_b[:, :, None].expand(-1, -1, Ky).reshape(-1)  # [N*K2]
    pys_b = pys_b[:, None, :].expand(-1, Kx, -1).reshape(-1)  # [N*K2]

    w = w.reshape(-1)                              # [N*K2]
    interp_w = interp_weights.repeat_interleave(K2)

    # 4. Boundary mask (simple clipping)
    valid = (
        (pxs_b >= 0) & (pxs_b < img_w) &
        (pys_b >= 0) & (pys_b < img_h)
    )

    pxs_b = pxs_b[valid]
    pys_b = pys_b[valid]
    w = w[valid]
    interp_w = interp_w[valid]

    # 5. Accumulate
    img.index_put_(
        (pys_b, pxs_b),
        interp_w * w,
        accumulate=True
    )

    return img


def window_event_to_img(xs, ys, ts, ps, sensor_size, s, win, polar=True):
    """Event image of the events with s <= t <= s + win, as a numpy array."""
    xs_win, ys_win, _, ps_win = time_window(xs, ys, ts, ps, s, win)
    img = events_to_image(xs_win, ys_win, ps_win, sensor_size, polar=polar)
    return img


# Wrapper of the event_utils-derived bilinear interpolation; see THIRD_PARTY_NOTICES.md
def events_to_image_torch(xt, yt, pt, image_size, polar=False, device=None):
    """Bilinear event image: counts, or with polar the signed sum of polarities (0 counts as -1)."""
    if device is None:
        device = xt.device
    img_size = (image_size[0], image_size[1])
    img = torch.zeros(img_size).to(device).float()
    pxs = (xt.floor()).float()
    pys = (yt.floor()).float()
    dxs = (xt - pxs).float()
    dys = (yt - pys).float()
    pxs = pxs.long()
    pys = pys.long()

    if polar == False:
        interp_weights = torch.ones_like(pt)
    else:
        npt = pt.clone()
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    img = interpolate_to_image(pxs, pys, dxs, dys, interp_weights, img)
    return img


# Wrapper of the event_utils-derived bilinear interpolation; see THIRD_PARTY_NOTICES.md
def events_to_image(xs, ys, ps, sensor_size, polar=False):
    """Bilinear event image of numpy events; its brightest pixel is then set to zero."""
    xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
    xt, yt, pt = xt.float(), yt.float(), pt.float()
    img = events_to_image_torch(xt, yt, pt, sensor_size, polar)
    img = img.numpy()
    img[np.argmax(img) // sensor_size[1], np.argmax(img) % sensor_size[1]] = 0
    img = img.reshape(sensor_size)
    return img


def Rot_Flow_Gau_events_to_image_torch_sr(xt, yt, pt, image_size, M, flow_xy=torch.tensor([1.0, 0.0]), sigma=1,
                                          device=None, polar=False, min_flow_mag=0.1):
    """Event image on the M-times finer grid, splatted with a Gaussian shaped by the transport.

    Sub-pixel sampling is gained along the motion only, so the kernel is narrow along flow_xy
    (sigma / 2 fine px) and wide across it (M * sigma / 2 fine px).  flow_xy is the (x, y)
    displacement of the segment in sensor px; only its direction is used.  A displacement
    shorter than min_flow_mag carries no usable direction and falls back to the isotropic
    M * sigma / 2 kernel.  The image is scaled by M.  Differentiable in xt, yt (through the
    sub-pixel offsets), pt and flow_xy.
    """
    if device is None:
        device = xt.device
    flow_xy = torch.as_tensor(flow_xy).to(device)
    img_size = np.asarray([int(image_size[0] * M), int(image_size[1] * M)])
    img = torch.zeros(img_size[0], img_size[1]).float().to(device)
    xt = xt * M
    yt = yt * M
    pxs = (xt.round()).float()
    pys = (yt.round()).float()
    dxs = (xt - pxs).float()
    dys = (yt - pys).float()
    pxs = pxs.long()
    pys = pys.long()
    pt = pt.float()
    if polar == False:
        interp_weights = pt.abs()
    else:
        npt = pt.clone()
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    sigma_across = M * sigma / 2
    if float(flow_xy.detach().float().norm()) < min_flow_mag:   # direction below min_flow_mag px is noise
        sigma_along = sigma_across                       # isotropic
        flow_dir = torch.tensor([1.0, 0.0], device=device)
    else:
        sigma_along = sigma / 2
        flow_dir = flow_xy
    kernel_sz = int(2 * sigma * M + 1) | 1
    img = gaussian_interpolate_to_image_rot_flow(
        pxs, pys, dxs, dys,
        interp_weights,
        img,
        sigma_along, sigma_across, flow_dir,
        kernel_size=[kernel_sz, kernel_sz]
    ) * M
    return img


def Rot_Flow_Gau_events_to_image_torch_sr_segments(
        xt, yt, tt, pt, image_size, segment_boundaries, M, Dxy_pc,
        sigma=1, device=None, polar=False, min_flow_mag=0.1):
    """One transport-shaped event image per trajectory segment, [K, M H, M W]; Dxy_pc [K, 2]
    holds the segment displacements in sensor px."""
    boundaries = segment_boundaries.to(device=tt.device, dtype=tt.dtype)
    segment_iwes = []
    for i in range(boundaries.numel() - 1):
        event_mask = (tt >= boundaries[i]) & (tt < boundaries[i + 1])
        if i == boundaries.numel() - 2:
            event_mask = event_mask | (tt == boundaries[-1])
        segment_iwes.append(
            Rot_Flow_Gau_events_to_image_torch_sr(
                xt[event_mask], yt[event_mask], pt[event_mask], image_size,
                M=M, flow_xy=Dxy_pc[i], sigma=sigma,
                device=device, polar=polar, min_flow_mag=min_flow_mag,
            )
        )
    return torch.stack(segment_iwes)


# Derived from SensorsINI/v2e (MIT), locally modified; see THIRD_PARTY_NOTICES.md
def lin_log(x, threshold=1+1e-6):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255 (8 bit)
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping

    Returns: the log value
    """
    # converting x into np.float64.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()
    rounding = 1e-8
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x * f, torch.log(x+rounding))
    return y


def plot_ax_roi_norm(ax, img, cmap="gray", axis_ind="off", roi=None, M=1, _norm=None):
    """Show img restricted to roi = [[x0, x1], [y_bottom, y_top]], given in native pixels.

    Native pixel k spans k to k + 1 and covers M x M pixels of an image at M times the native
    sampling, so panels at different M show the same region."""
    cax = ax.imshow(img, cmap=cmap, norm=_norm)
    if roi is None:
        roi = [[0, img.shape[1] / M], [img.shape[0] / M, 0]]
    ax.set_xlim(np.asarray(roi[0]) * M - 0.5)
    ax.set_ylim(np.asarray(roi[1]) * M - 0.5)
    ax.axis(axis_ind)
    return cax


def add_roi(ax, roi, M=1, color='red'):
    """Outline roi (native pixels) on an image shown at M times the native sampling."""
    roi_x = np.asarray(roi[0]) * M - 0.5
    roi_y = np.asarray(roi[1]) * M - 0.5
    ax.plot([roi_x[0], roi_x[0], roi_x[1], roi_x[1], roi_x[0]],
            [roi_y[0], roi_y[1], roi_y[1], roi_y[0], roi_y[0]],
            color=color, linestyle='--', linewidth=1.5)


def bounded_softplus(raw, c_min=C_MIN, c_max=C_MAX):
    """Map an unconstrained parameter into (c_min, c_max)."""
    positive = F.softplus(raw)
    normalized = positive / (1.0 + positive)
    return c_min + (c_max - c_min) * normalized


def raw_from_bounded_softplus(value, c_min=C_MIN, c_max=C_MAX):
    """Inverse of bounded_softplus: the raw parameter of a target c."""
    v = min(max(float(value), c_min + 1e-6), c_max - 1e-6)
    normalized = (v - c_min) / (c_max - c_min)
    positive = normalized / (1.0 - normalized)
    if positive > 20.0:
        return torch.tensor(float(positive))
    return torch.tensor(float(math.log(math.expm1(positive))))


# Derived from clovaai/AdamP (MIT), with complex-valued modifications by Ni Chen for inverse
# diffraction imaging; see THIRD_PARTY_NOTICES.md
class AdamP(Optimizer):
    '''
        https://clovaai.github.io/AdamP/
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.reshape(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        # Modified by Ni Chen: the similarity of the magnitudes, for complex parameters
        return F.cosine_similarity(x.abs(), y.abs(), dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss


def l2_loss(x):
    y = torch.mean(torch.square(x))
    return y


def iwe_neg_var_loss(iwe_refs):
    """Contrast-maximisation loss: minus the variance of an image of warped events.

    A list, tuple or stacked tensor of images averages the same objective over them.
    """
    if isinstance(iwe_refs, torch.Tensor):
        refs = [iwe_refs] if iwe_refs.ndim == 2 else list(iwe_refs)
    else:
        refs = list(iwe_refs)
    if not refs:
        raise ValueError("iwe_refs must contain at least one IWE")
    return torch.stack([-torch.var(iwe) for iwe in refs]).mean()


def l1_loss(x, beta=0.5):
    """Mean smooth-L1 magnitude of x; beta is the width of the quadratic zone."""
    smooth_loss = torch.nn.SmoothL1Loss(beta = beta, reduction='mean')
    return smooth_loss(x, torch.zeros_like(x))


def tv_loss(x, tv_tau=1e-4, iso=True, spatial_scale=1.0):
    """First-order total variation of a real or complex image, averaged over its pixels.

    iso sums sqrt(|d_x|^2 + |d_y|^2 + tv_tau); otherwise |d_x| + |d_y|.  spatial_scale multiplies
    the differences, so that one weight means the same on the native and on a finer grid.
    """
    arr_size = 1
    for idx in range(len(x.shape)):
        arr_size = arr_size * x.shape[idx]

    x_d = x - torch.roll(x, -1, dims=1)
    y_d = x - torch.roll(x, -1, dims=0)

    x_d = x_d * spatial_scale
    y_d = y_d * spatial_scale
    x_d[:, -1:] = torch.tensor(0.0, device=x.device)
    y_d[-1:, :] = torch.tensor(0.0, device=x.device)

    if iso:
        TV_amp = torch.sqrt(x_d.abs() ** 2 + y_d.abs() ** 2 + tv_tau).sum()
    else:
        TV_amp = (x_d.abs() + y_d.abs()).sum()

    return TV_amp / arr_size


def flow_regularization_weights(Dxy_pc, eps=1e-8, min_flow_mag=0.0):
    """Direction weights B with B.T B = I - (D.T D) / lambda_max, from segment displacements D.

    The events constrain the gradient along the transport, so this prior acts across it.
    Rank-one motion gives the transverse projection.  Independent motion in the second
    direction reduces its weight, equal coverage gives zero penalty, and no motion gives
    ordinary isotropic TV.  The geometry is detached so that the image prior cannot steer the
    transport, and segments below min_flow_mag are excluded as they are from the event term.
    """
    D = Dxy_pc.detach()
    if D.ndim != 2 or D.shape[1] != 2:
        raise ValueError("Dxy_pc must have shape [num_segments, 2]")
    D = D * (torch.linalg.vector_norm(D, dim=1) > min_flow_mag)[:, None]
    eigenvalues, axes = torch.linalg.eigh(D.T @ D)
    largest = eigenvalues[-1].clamp_min(0)
    if largest <= eps:
        return torch.eye(2, device=D.device, dtype=D.dtype)
    weights = (1.0 - eigenvalues.clamp_min(0) / largest).clamp(0, 1).sqrt()
    return weights[:, None] * axes.T


def tv_loss_flow(x, direction_weights, tv_tau=1e-4, spatial_scale=1.0):
    """Isotropic total variation of the gradient projected by direction_weights (2 x 2).

    The differences are expressed in sensor-pixel coordinates through spatial_scale, so one
    weight means the same at M = 1 and at M > 1.
    """
    arr_size = 1
    for idx in range(len(x.shape)):
        arr_size = arr_size * x.shape[idx]

    x_d = x - torch.roll(x, -1, dims=1)
    y_d = x - torch.roll(x, -1, dims=0)

    x_d = x_d * spatial_scale
    y_d = y_d * spatial_scale

    x_d[:, -1:] = torch.tensor(0.0, device=x.device)
    y_d[-1:, :] = torch.tensor(0.0, device=x.device)
    B = direction_weights.detach().to(device=x.device, dtype=x.dtype)
    projected = B @ torch.stack((x_d, y_d)).reshape(2, -1)
    d_ver_sq = projected[0] ** 2 + projected[1] ** 2
    TV_amp = torch.sqrt(d_ver_sq + tv_tau).sum()
    return TV_amp / arr_size


def down_sampling(x, sigma):
    """Pixel-aperture average and decimation: the mean over non-overlapping sigma x sigma blocks."""
    patches = x.unfold(0, sigma, sigma).unfold(1, sigma, sigma)
    y = patches.mean(dim=-1).mean(dim=-1)

    return y


def reset_optimizer_and_params(optimizer, param_groups):
    """Close a stage: drop the gradients of its parameters and the optimizer state, free caches."""
    all_params = []
    for group in param_groups:
        params = group.get('params', None)
        if params is None:
            continue
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        all_params.extend(params)

    for p in all_params:
        if hasattr(p, "grad") and p.grad is not None:
            p.grad = None

    optimizer.state.clear()
    torch.cuda.empty_cache()
    gc.collect()


def dense_flow_grid(flow_xy):
    """grid_sample coordinates of x + flow for a per-pixel step flow_xy [2, H, W] in pixels."""
    _, h, w = flow_xy.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow_xy.device, dtype=flow_xy.dtype),
        torch.arange(w, device=flow_xy.device, dtype=flow_xy.dtype),
        indexing="ij",
    )
    sample_x = xx + flow_xy[0]
    sample_y = yy + flow_xy[1]
    grid_x = sample_x / max(w - 1, 1) * 2.0 - 1.0
    grid_y = sample_y / max(h - 1, 1) * 2.0 - 1.0
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def unit_flow_mag_mask(flow_xy, eps=1e-1, border=1):
    """Unit direction, magnitude and mask of a step field [2, H, W].

    The mask keeps finite magnitudes above eps away from the image border; the direction is
    zero outside it."""
    if flow_xy.ndim != 3 or flow_xy.shape[0] != 2:
        raise ValueError("flow_xy must have shape [2,H,W]")
    mag = torch.linalg.vector_norm(flow_xy, dim=0)
    safe_mag = torch.where(mag > float(eps), mag, torch.ones_like(mag))
    mask = torch.isfinite(mag) & (mag > float(eps))
    if border > 0:
        border_mask = torch.zeros_like(mask)
        border_mask[border:-border, border:-border] = True
        mask = mask & border_mask
    mask = mask.float()
    unit_flow = flow_xy / safe_mag.unsqueeze(0)
    unit_flow = unit_flow * mask.unsqueeze(0)
    return unit_flow.float(), safe_mag.float(), mask


def warp_iwe_from_log_image(log_image, flow_map):
    """L(x) - L(x + u) for a per-pixel step u = flow_map [2, H, W], sampled bilinearly with zeros
    outside the image.  It equals -grad L . u for affine L, and is centred half a step
    downstream of x."""
    grid = dense_flow_grid(flow_map).to(device=log_image.device, dtype=log_image.dtype)
    shifted = F.grid_sample(log_image[None, None], grid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)[0, 0]
    return log_image - shifted


def segment_iwe_from_log_image(log_image, segment_displacements, min_flow_mag):
    """Finite-step prediction of the per-segment event image from the log latent.

    Each D_i of segment_displacements [K, 2] is in sensor pixels, and u_i = D_i / |D_i| is a
    one-pixel step on the grid of log_image (one fine pixel at M > 1).  The prediction is
    |D_i| [L(x) - L(x + u_i)], which equals -grad L . D_i for affine L.  Both |D_i| and u_i
    stay differentiable.  The mask drops segments with |D_i| <= min_flow_mag and the image
    border, so that both samples are inside the image; apply it to the measured image too.
    """
    if log_image.ndim != 2 or segment_displacements.ndim != 2 or segment_displacements.shape[1] != 2:
        raise ValueError("Expected L [H, W] and displacements [K, 2]")
    if segment_displacements.shape[0] == 0:
        raise ValueError("At least one segment is required")
    predictions, masks = [], []
    for displacement in segment_displacements:
        flow = displacement.to(device=log_image.device, dtype=log_image.dtype)
        flow = flow[:, None, None].expand(2, *log_image.shape)
        unit_flow, magnitude, mask = unit_flow_mag_mask(flow, eps=min_flow_mag, border=1)
        predictions.append(warp_iwe_from_log_image(log_image, unit_flow) * magnitude * mask)
        masks.append(mask)
    return torch.stack(predictions), torch.stack(masks)


def track_position(img_seq):
    """Shift (row, column) of each image against the first, by phase correlation at 1/100 px.

    A ten-pixel border is ignored.  Returns a [2, N] numpy array whose first column is zero.
    """
    img_ref = img_seq[0]
    img_num = img_seq.size(0)
    pos = np.zeros((2, img_num))
    border = 10
    reference = img_ref[border:-border, border:-border].cpu().numpy()

    for idx in range(1, img_num):
        image = img_seq[idx, border:-border, border:-border].cpu().numpy()
        pos[:, idx] = phase_cross_correlation(reference, image, upsample_factor=100, space='real',
                                              normalization=None)[0]

    return pos
