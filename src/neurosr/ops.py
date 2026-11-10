"""Low-level operations: event images, intensity mapping, optimizers, registration and propagation.

Array axes are y,x; event coordinates are x,y; times are microseconds.
Ni Chen's optimizer header and complex-valued AdamP modifications are retained.
"""
from __future__ import annotations

import gc
import math
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from skimage.registration import phase_cross_correlation

ArrayLike = Union[np.ndarray, torch.Tensor]
C_MIN, C_MAX = 1e-3, 0.5

# optimizer for inverse diffraction imaging.
# @author: Ni Chen

def  time_window(xs, ys,ts, ps ,s , win ):
    loc = np.where((ts <= s + win) & (ts >= s))
    ts, xs, ys, ps = ts[loc], xs[loc], ys[loc], ps[loc]
    return xs, ys, ts, ps


def np_to_torch(xs, ys, ts, ps,device):
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
    Flow-ROTATED anisotropic Gaussian splatting (per-event normalized).

    Same contract as gaussian_interpolate_to_image_flow, but the kernel's
    principal axes follow the flow instead of the image axes: sigma_along is
    the std along flow_dir and sigma_across the std perpendicular to it, both
    in pixels of img.  With flow_dir = (1, 0) it equals
    gaussian_interpolate_to_image_flow(..., sigma_xy=(sigma_along, sigma_across)),
    with flow_dir = (0, 1) it equals sigma_xy=(sigma_across, sigma_along).
    flow_dir is a length-2 tensor (x, y); its norm and sign do not matter
    (a zero vector falls back to the x axis).
    Differentiable in dxs, dys, interp_weights, sigma_along, sigma_across and
    flow_dir.  The rotated kernel is not separable, so the [N, Kx, Ky]
    weights are formed directly (same memory order as the separable version).
    Boundary handling: simple clipping (no post-clip renormalization).
    The window is square in image axes, so an oblique ellipse is truncated
    less than an axis-aligned one: with the shared (2*sigma*M+1) window the
    rendered across std is 1.40 HR px axis-aligned and 1.48 HR px at 45 deg
    for a nominal 1.5 (the isotropic reference renders 1.40).
    """
    device = pxs.device
    img_h, img_w = img.shape

    # ------------------------------------------------------------------
    # 1. Kernel offsets (integer for indexing, real for the weights)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Rotated Gaussian kernel (per-event), axes along / across the flow
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Pixel coordinates (expand to event-pixel domain)
    # ------------------------------------------------------------------
    pxs_b = pxs[:, None] + offsets_xi[None, :]     # [N, Kx]
    pys_b = pys[:, None] + offsets_yi[None, :]     # [N, Ky]

    pxs_b = pxs_b[:, :, None].expand(-1, -1, Ky).reshape(-1)  # [N*K2]
    pys_b = pys_b[:, None, :].expand(-1, Kx, -1).reshape(-1)  # [N*K2]

    w = w.reshape(-1)                              # [N*K2]
    interp_w = interp_weights.repeat_interleave(K2)

    # ------------------------------------------------------------------
    # 4. Boundary mask (simple clipping)
    # ------------------------------------------------------------------
    valid = (
        (pxs_b >= 0) & (pxs_b < img_w) &
        (pys_b >= 0) & (pys_b < img_h)
    )

    pxs_b = pxs_b[valid]
    pys_b = pys_b[valid]
    w = w[valid]
    interp_w = interp_w[valid]

    # ------------------------------------------------------------------
    # 5. Accumulate
    # ------------------------------------------------------------------
    img.index_put_(
        (pys_b, pxs_b),
        interp_w * w,
        accumulate=True
    )

    return img


def window_event_to_img(xs, ys, ts, ps,sensor_size,s=int(1 / 60 * 12 * 1e6), win=int(1 / 60 * 3 * 1e6), polar = True):
    xs_win, ys_win, ts_win, ps_win = time_window(xs, ys, ts, ps, s,
                                                 win)
    img = events_to_image(xs_win, ys_win, ps_win,
                          sensor_size, interpolation='bilinear', polar = polar)
    return img


# Wrapper of the event_utils-derived bilinear interpolation; see THIRD_PARTY_NOTICES.md
def events_to_image_torch(xt, yt, pt, image_size,polar= False,
                          device=None):
    if device is None:
        device = xt.device
    img_size = (image_size[0] , image_size[1])
    # mask = ((xt <= clipx) & (xt > 0)) & ((yt <= clipy) | (yt > 0))
    img = torch.zeros(img_size).to(device).float()
    pxs = (xt.floor()).float()
    pys = (yt.floor()).float()
    dxs = (xt - pxs).float()#[mask]
    dys = (yt - pys).float()#[mask]
    pxs = pxs.long()#[mask]
    pys = pys.long()#[mask]

    if polar == False:
        interp_weights = torch.ones_like(pt)
    else:
        npt = pt.clone()#[mask]
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    img=interpolate_to_image(pxs, pys, dxs, dys, interp_weights , img)
    return img


# Wrapper of the event_utils-derived bilinear interpolation; see THIRD_PARTY_NOTICES.md
def events_to_image(xs, ys, ps, sensor_size=(180, 240), interpolation='bilinear', polar =False,padding=False):
    """
    Place events into an image using numpy
    """
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
        xt, yt, pt = xt.float(), yt.float(), pt.float()
        img = events_to_image_torch(xt, yt, pt,sensor_size, polar)
        img = img.numpy()
        img[np.argmax(img) // sensor_size[1], np.argmax(img) % sensor_size[1]] = 0
    else:
        coords = np.stack((ys, xs))
        try:
            abs_coords = np.ravel_multi_index(coords, sensor_size)
        except ValueError:
            print("Issue with input arrays! coords.shape={}, sum(coords)={}, sensor_size={}. \n minx={}, maxx={}, miny={}, maxy={}".format(coords.shape, np.sum(coords), sensor_size, np.min(xs), np.max(xs), np.min(ys), np.max(ys)))
            raise ValueError
        if polar == True:
            img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0] * sensor_size[1])
        else:
            img = np.bincount(abs_coords, weights=np.ones(ps.size), minlength=sensor_size[0]*sensor_size[1])
    img = img.reshape(sensor_size)
    return img


def Rot_Flow_Gau_events_to_image_torch_sr(xt, yt, pt, image_size, M, flow_xy=torch.tensor([1.0, 0.0]), sigma=1,
                                          device=None, polar=False, min_flow_mag=0.1):
    """Flow-shaped Gaussian splat whose narrow axis follows the SIGNED flow vector.

    Same call as Flow_Gau_events_to_image_torch_sr, but flow_xy is the signed
    (x, y) displacement (its direction matters, its sign and norm do not) and
    the anisotropy is rotated to it: sigma / 2 HR px along the flow (the width
    the per-axis version gives its dominant axis) and M * sigma / 2 HR px
    across the flow.  Identical to Flow_Gau_events_to_image_torch_sr when
    flow_xy is axis aligned.  For oblique flow the per-axis version narrows
    BOTH axes (45 deg motion gets (sigma/2, sigma/2) and renders as a nearest
    splat); this one keeps the across-flow width.  A displacement shorter than
    min_flow_mag sensor px (default 0.1) carries no usable direction and no
    sub-pixel sampling along it, so it falls back to the isotropic
    M * sigma / 2 kernel (the per-axis version does the same only at 1e-5).
    Differentiable in xt, yt (through the sub-pixel offsets), pt and flow_xy.
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
        sigma_along = sigma_across                       # isotropic, like Gau_events_to_image_torch_sr
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
    """Rot_Flow_Gau_events_to_image_torch_sr per trajectory segment, flow_xy = Dxy_pc[i] (signed)."""
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
    #y = torch.where(x <= threshold, x*f, torch.log(x))
    y = torch.where(x <= threshold, x * f, torch.log(x+rounding))
    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    # rounding = 1e8
    # y = torch.round(y*rounding)/rounding

    #return y.float()
    return y


def plot_ax_roi_norm(ax, img, cmap = "gray", axis_ind = "off", roi = None, Dxy_ori =np.asarray([[0,0],[0,0]]), M = 1,_norm= None):
    cax = ax.imshow(img, cmap=cmap, norm=_norm)
    if roi is None:
        roi = [[0,img.shape[1]],[img.shape[0], 0]]
    ax.set_xlim(np.asarray(roi[0] * M) + np.asarray(Dxy_ori[0]));
    ax.set_ylim(np.asarray(roi[1] * M) + np.asarray(Dxy_ori[1]));
    ax.axis(axis_ind)
    return cax


def add_roi(ax, roi, M = 1, Dxy_ori = np.asarray([[0,0],[0,0]]), color='red'):
    roi_x = np.asarray(roi[0]) * M + Dxy_ori[0][0]
    roi_y = np.asarray(roi[1]) * M + Dxy_ori[1][0]
    ax.plot([roi_x[0], roi_x[0], roi_x[1], roi_x[1], roi_x[0]],
            [roi_y[0], roi_y[1], roi_y[1], roi_y[0], roi_y[0]],
            color=color, linestyle='--', linewidth=2)


def bounded_softplus(raw, c_min=C_MIN, c_max=C_MAX):
    positive = F.softplus(raw)
    normalized = positive / (1.0 + positive)
    return c_min + (c_max - c_min) * normalized


def raw_from_bounded_softplus(value, c_min=C_MIN, c_max=C_MAX):
    """Inverse of bounded_softplus so we can initialise raw_c from a target c."""
    v = min(max(float(value), c_min + 1e-6), c_max - 1e-6)
    normalized = (v - c_min) / (c_max - c_min)
    positive = normalized / (1.0 - normalized)
    if positive > 20.0:
        return torch.tensor(float(positive))
    return torch.tensor(float(math.log(math.expm1(positive))))


# Derived from clovaai/AdamP (MIT), with complex-valued modifications by Ni Chen; see THIRD_PARTY_NOTICES.md
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
        return x.reshape(1, -1)#x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        # return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()
        # Modified by Ni Chen
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
    """CMax sharpness loss for one or more reference-time IWEs.

    A single 2-D tensor reproduces the original ``-torch.var(iwe)`` objective.
    A list/tuple (or a stacked tensor) averages the same objective over the
    requested reference times, so changing ``num_t_ref`` does not rescale the
    loss or change the trajectory segmentation.
    """
    if isinstance(iwe_refs, torch.Tensor):
        refs = [iwe_refs] if iwe_refs.ndim == 2 else list(iwe_refs)
    else:
        refs = list(iwe_refs)
    if not refs:
        raise ValueError("iwe_refs must contain at least one IWE")
    return torch.stack([-torch.var(iwe) for iwe in refs]).mean()


def l1_loss(x, beta = 0.5):
    smooth_loss = torch.nn.SmoothL1Loss(beta = beta, reduction='mean')
    return smooth_loss(x, torch.zeros_like(x))


def tv_loss(x, tv_order=1, tv_tau=1e-4, iso=True, spatial_scale=1.0):
    '''The smaller tv_tau, the smoother the image.'''
    arr_size = 1
    for idx in range(len(x.shape)):
        arr_size = arr_size * x.shape[idx]

    if tv_order == 1:
        x_d = x - torch.roll(x, -1, dims=1)
        y_d = x - torch.roll(x, -1, dims=0)

    elif tv_order == 2:
        '''need check'''
        x_d = x - 2 * torch.roll(x, -1, dims=1) + torch.roll(x, 2, dims=1)
        y_d = x - 2 * torch.roll(x, -1, dims=0) + torch.roll(x, 2, dims=0)

    elif tv_order == 3:
        '''need check'''
        x_d = x - 3 * torch.roll(x, -1, dims=1) + 3 * torch.roll(x, 2, dims=1) - torch.roll(x, 3, dims=1)
        y_d = x - 3 * torch.roll(x, -1, dims=0) + 3 * torch.roll(x, 2, dims=0) - torch.roll(x, 3, dims=0)

    else:
        print('order should be smaller than 3')

    x_d = x_d * spatial_scale ** tv_order
    y_d = y_d * spatial_scale ** tv_order
    x_d[:, -tv_order:] = torch.tensor(0.0, device=x.device)
    y_d[-tv_order:, :] = torch.tensor(0.0, device=x.device)
    # x_d[:, :tv_order-1] = torch.zeros(1, device=x.device)
    # y_d[-tv_order:-1, :] = torch.zeros(1, device=x.device)

    if iso == True:
        TV_amp = torch.sqrt(x_d.abs() ** 2 + y_d.abs() ** 2 + tv_tau).sum()

    elif iso == False:
        TV_amp = (x_d.abs() + y_d.abs()).sum()

    elif iso == 'mix':
        '''https://arxiv.org/abs/2005.04401'''
        TV_amp = (x_d.abs() + y_d.abs()).sum() - 0.5 * torch.sqrt(x_d.abs() ** 2 + y_d.abs() ** 2 + tv_tau).sum()

    else:
        print('wrong parameter')

    return TV_amp / arr_size


def flow_regularization_weights(Dxy_pc, eps=1e-8, min_flow_mag=0.0):
    """Direction weights B with B.T B = I - (D.T D)/lambda_max.

    Rank-one motion gives the usual transverse TV projection.  Independent
    motion in the second direction reduces its weight; equal coverage gives
    zero directional penalty.  No motion gives ordinary isotropic TV.
    Segment reversals preserve D.T D, including oblique back-and-forth motion.
    Detach the geometry so the image prior cannot steer the estimated flow.
    Use the event branch's min_flow_mag so excluded segments cannot falsely
    fill its null space and reduce TV.
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


def tv_loss_flow(x, tv_order=1, tv_tau=1e-4, iso=True, Dxy=[1, 1],
                 spatial_scale=1.0, direction_weights=None):
    '''The smaller tv_tau, the smoother the image.'''
    arr_size = 1
    for idx in range(len(x.shape)):
        arr_size = arr_size * x.shape[idx]

    if tv_order == 1:
        x_d = x - torch.roll(x, -1, dims=1)
        y_d = x - torch.roll(x, -1, dims=0)
    elif tv_order == 2:
        '''need check'''
        x_d = x - 2 * torch.roll(x, -1, dims=1) + torch.roll(x, 2, dims=1)
        y_d = x - 2 * torch.roll(x, -1, dims=0) + torch.roll(x, 2, dims=0)

    # Express the discrete derivative in sensor-pixel coordinates so the same
    # kappa_tv has the same meaning for M=1 and M>1.
    x_d = x_d * spatial_scale ** tv_order
    y_d = y_d * spatial_scale ** tv_order

    x_d[:, -tv_order:] = torch.tensor(0.0, device=x.device)
    y_d[-tv_order:, :] = torch.tensor(0.0, device=x.device)
    if direction_weights is None:
        d_ver = -x_d * Dxy[1] + y_d * Dxy[0]
        d_ver_sq = d_ver ** 2
    else:
        # A matrix of directional weights generalizes the single transverse
        # projection; its Euclidean norm also has a finite gradient at zero.
        B = direction_weights.detach().to(device=x.device, dtype=x.dtype)
        if B.ndim == 4:
            # one 2x2 per pixel, from flow_regularization_weights_dense.  Written
            # out rather than contracted: einsum is about twice as slow here.
            projected = torch.stack((B[0, 0] * x_d + B[0, 1] * y_d,
                                     B[1, 0] * x_d + B[1, 1] * y_d))
        else:
            projected = B @ torch.stack((x_d, y_d)).reshape(2, -1)
        # Carry the squared norm, which is what the isotropic branch needs.
        # torch.linalg.vector_norm over dim 0 of a 2 x N tensor costs about
        # 400 times this on CPU for the same arithmetic.
        d_ver_sq = projected[0] ** 2 + projected[1] ** 2
        d_ver = None
    if   iso:
        TV_amp = torch.sqrt(d_ver_sq + tv_tau).sum()
    else:
        TV_amp = (d_ver.abs() if d_ver is not None else d_ver_sq.sqrt()).sum()
    return TV_amp / arr_size


def down_sampling(x, sigma):
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.tensor(x, dtype=torch.float32)

    # Unfold with a step size equal to the block size to prevent overlap
    patches = x.unfold(0, sigma, sigma).unfold(1, sigma, sigma)
    y = patches.mean(dim=-1).mean(dim=-1)

    return y


def reset_optimizer_and_params(optimizer, param_groups):
    """
    彻底清除 optimizer 的内部状态和参数梯度，并删除参数引用。
    支持 param_groups 结构形如 [{'params': param, 'lr': ...}, ...]
    """
    # 1️⃣ 遍历所有参数组
    all_params = []
    for group in param_groups:
        params = group.get('params', None)
        if params is None:
            continue
        # 如果 params 是单个 tensor 而不是列表
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        all_params.extend(params)

    # 2️⃣ 清理梯度
    for p in all_params:
        if hasattr(p, "grad") and p.grad is not None:
            p.grad = None

    # 3️⃣ 清空优化器状态
    optimizer.state.clear()

    # 4️⃣ 删除参数对象（从当前命名空间）
    # 注意：无法直接在函数内部删除外部变量名，
    #       这里只能返回需要删除的对象列表。
    del all_params[:]

    # 5️⃣ 回收资源
    torch.cuda.empty_cache()
    gc.collect()

    print("Optimizer state, gradients, and parameter references have been reset.")


def as_flow_tensor(
    flow: ArrayLike,
    image_shape: tuple[int, int],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if not torch.is_tensor(flow):
        flow = torch.from_numpy(np.asarray(flow))
    if device is not None:
        flow = flow.to(device=torch.device(device))
    flow = flow.float()
    if flow.ndim == 4 and flow.shape[0] == 1:
        flow = flow[0]
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError("flow tensor must have shape [2,H,W] or [1,2,H,W]")
    if tuple(flow.shape[-2:]) != tuple(image_shape):
        flow = F.interpolate(flow[None], size=image_shape, mode="bilinear", align_corners=True)[0]
    return flow


def dense_flow_grid(flow_xy: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    flow_xy = as_flow_tensor(flow_xy, image_shape)
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


def unit_flow_mag_mask(
    flow_xy: torch.Tensor,
    eps: float = 1e-1,
    border: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if flow_xy.ndim != 3 or flow_xy.shape[0] != 2:
        raise ValueError("flow_xy must have shape [2,H,W]")
    mag = torch.linalg.vector_norm(flow_xy, dim=0)
    safe_mag = torch.where(mag > float(eps), mag, torch.ones_like(mag))
    mask = torch.isfinite(mag) & (mag > float(eps))
    if border > 0:
        border_mask = torch.zeros_like(mask)
        border_mask[border:-border, border:-border] = True
        #mask = border_mask
        mask = mask & border_mask
    mask = mask.float()
    unit_flow = flow_xy / safe_mag.unsqueeze(0)
    unit_flow = unit_flow * mask.unsqueeze(0)
    return unit_flow.float(), safe_mag.float(), mask


def warp_iwe_from_log_image(
    log_image: torch.Tensor,
    flow_map: torch.Tensor,
    padding_mode: str = "zeros",
    stencil: str = "forward",
) -> torch.Tensor:
    """L(x) - L(x + u) ("forward", stencil centre at x + u/2) or
    L(x - u/2) - L(x + u/2) ("centered", stencil centre at x).

    Both equal -grad L . u for affine L; on curved L the forward stencil is the
    centred one shifted by u/2, so its IWE is registered half a step downstream
    of the latent it is compared with."""
    if stencil not in ("forward", "centered"):
        raise ValueError(f"stencil must be 'forward' or 'centered', got {stencil!r}")

    def _sample(flow):
        grid = dense_flow_grid(flow, tuple(log_image.shape)).to(device=log_image.device, dtype=log_image.dtype)
        return F.grid_sample(
            log_image[None, None],
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )[0, 0]

    if stencil == "forward":
        return log_image - _sample(flow_map)
    return _sample(-0.5 * flow_map) - _sample(0.5 * flow_map)


def segment_iwe_from_log_image(
    log_image: torch.Tensor,
    segment_displacements: torch.Tensor,
    min_flow_mag: float = 0.25,
    stencil: str = "forward",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-step frame differences on the existing segment-EKLT amplitude scale.

    Each D_i is in sensor pixels; u_i = D_i / |D_i| is a one-pixel stencil
    on the CURRENT image grid (one HR pixel during SR).  The prediction is
    |D_i| * [L(x) - L(x + u_i)] (stencil="forward") or
    |D_i| * [L(x - u_i/2) - L(x + u_i/2)] (stencil="centered"); both agree
    with -grad_grid(L).D_i for affine L, and the centred one is registered on
    x rather than x + u_i/2.  This is a finite-difference IWE model, not an
    exact inter-event constraint.

    Both |D_i| and u_i stay differentiable.  The mask uses the ORIGINAL
    sensor displacement, before normalization, and excludes the image border
    so both samples remain in bounds.  Apply it to the rendered target too.
    Uniform [K,2] or spatial [K,2,H,W] displacements are accepted.
    """
    if log_image.ndim != 2 or segment_displacements.ndim not in (2, 4):
        raise ValueError("Expected L [H,W] and displacements [K,2] or [K,2,H,W]")
    if segment_displacements.shape[0] == 0 or segment_displacements.shape[1] != 2:
        raise ValueError("At least one segment with x/y displacement is required")
    predictions, masks = [], []
    for displacement in segment_displacements:
        flow = displacement.to(device=log_image.device, dtype=log_image.dtype)
        if flow.ndim == 1:
            flow = flow[:, None, None].expand(2, *log_image.shape)
        if flow.shape[1:] != log_image.shape:
            raise ValueError("Dense segment flow must match the image grid")
        unit_flow, magnitude, mask = unit_flow_mag_mask(flow, eps=min_flow_mag, border=1)
        predictions.append(warp_iwe_from_log_image(log_image, unit_flow, stencil=stencil) * magnitude * mask)
        masks.append(mask)
    return torch.stack(predictions), torch.stack(masks)


def track_position(img_seq=[], is_mask=1):
    if is_mask:
        img_seq = img_seq / torch.mean(img_seq, dim=0, keepdim=True)

    img_ref = img_seq[0]
    img_num = img_seq.size(0)
    pos = np.zeros((2, img_num))
    edge_ignore1 = 10
    standardImg = img_ref[edge_ignore1:-edge_ignore1, edge_ignore1:-edge_ignore1]

    print('calculating position...')
    for idx in range(1, img_num):
        copyImg = img_seq[idx, edge_ignore1:-edge_ignore1, edge_ignore1:-edge_ignore1]
        pos[:, idx], error, diffphase = phase_cross_correlation(standardImg.cpu().numpy(), copyImg.cpu().numpy(),
                                                                upsample_factor=100, space='real', normalization=None)
    
    return pos


def refine_position(img_seq=[], pos=[], is_mask=1):
    device = img_seq.device
    if is_mask:
        img_seq = img_seq / torch.mean(img_seq, dim=0, keepdim=True)

    img_size = img_seq.size(-1)
    img_num = img_seq.size(0)

#     fx_shift, fy_shift = shift_frequency(img_size=img_size, device=device)
#     img_refine = torch.zeros(img_size, img_size, dtype=torch.float64, device=device)
# #     padded_image = F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))
#     for idx in range(img_num):
#         Hs = torch.exp(-1j * 2 * np.pi * (fx_shift * pos[0, idx] / img_size + fy_shift * pos[1, idx] / img_size)).to(
#             device)
#         img_refine += (ifft2(fft2(img_seq[:, :, idx]) * Hs)).real.abs()
#     img_refine = img_refine / img_num

    img_refine = img_seq[0, :, :]

    # Refine the subpixel shifts
    pos = np.zeros((2, img_num))
    edge_ignore1 = 10
    # print('Refine position...')
    img_reference = img_refine[edge_ignore1:-edge_ignore1, edge_ignore1:-edge_ignore1]
    for idx in range(1, img_num):
        copyImg = img_seq[idx, edge_ignore1:-edge_ignore1, edge_ignore1:-edge_ignore1]
        pos[:, idx], error, diffphase = phase_cross_correlation(img_reference.cpu().numpy(), copyImg.cpu().numpy(),
                                                                upsample_factor=100, space='real', normalization=None)


    return pos


class Simple_sr:
    def __init__(self, holo_raw=None,
                 img_num=9, img_size= np.asarray([260, 346]) , mag=3, pp_sensor=1.1e-6,device = "cpu"
                 ):
        self.holo_raw = holo_raw
        self.device = device
        self.img_num = img_num
        self.img_size = img_size
        self.mag = mag
        self.img_size_up = img_size * mag
        self.pp_sensor = pp_sensor
        self.pp_up = pp_sensor / mag


    def pos_registration(self, is_mask=0):
        holo_raw_hr = torch.zeros(self.img_num, self.img_size_up[0], self.img_size_up[1], dtype=torch.float64,
                                  device=self.device)
        for idx_img in range(self.img_num):
            tmp = self.holo_raw[idx_img, :, :]
            #             holo_raw_hr[:,:,idx_img] = F.interpolate(tmp[(None,) * 2], size=(self.img_size_up, self.img_size_up),
            #                                     mode='bicubic').squeeze()
            holo_raw_hr[idx_img, :, :] = F.interpolate(tmp[(None,) * 2], size=(self.img_size_up[0], self.img_size_up[1]),
                                                       mode='bilinear', align_corners=True).squeeze()

        pos_track = track_position(img_seq=holo_raw_hr, is_mask=is_mask)
        pos_refine = pos_track

        for iLoc in range(3):
            pos_refine = refine_position(img_seq=holo_raw_hr, pos=pos_refine, is_mask=is_mask)

        self.pos = pos_refine
