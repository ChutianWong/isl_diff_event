"""Joint event reconstruction at fixed transport and contrast, from an EDI anchor.

Two event terms act on the same latent and share one pixel aperture, a native-pixel box average
of the sensor intensity taken before the log response:

    IWE term    the finite-step prediction along each segment against c times the segmented
                image of warped events;
    pair term   the event-wise (EW) vote: for each event, the log-intensity increment since
                the previous event at its pixel, sampled at the two warped positions, against
                +-c; the residuals are splatted with the transport-shaped kernel and normalized
                by the splatted count.

The frame term compares the pixel-aperture average of the latent with the anchor.  The latent is
an intensity, or the intensity |P_z{O}|^2 of a propagated complex field.
"""
import math

import torch
import torch.nn.functional as F

from ._constants import MIN_FLOW_MAG, PIXEL_PITCH_M, SIGMA, SIGMA_SR, WAVELENGTH_M
from .optics import AngularSpectrum, laplacian
from .ops import (
    AdamP, down_sampling, l1_loss, tv_loss,
    Rot_Flow_Gau_events_to_image_torch_sr_segments as render_segments,
    events_to_image_torch,
    gaussian_interpolate_to_image_rot_flow,
    lin_log,
)
from .events import previous_event_indices, sample_image_at_events

EPS = 1e-6
MIXING = (.5, .5)             # IWE term, pair term, after their values are matched at the first step
IWE_WEIGHT = 0.2              # IWE term against the frame term, for an intensity latent
SHORT_SEGMENT_PX = 0.1        # a shorter segment has no usable direction: its kernel is isotropic
LOG_TV_WEIGHT = 0.02          # isotropic TV of log(I + LOG_TV_OFFSET) on the fine intensity
LOG_TV_OFFSET = 0.02
# Step size, StepLR (step_size, gamma) and prior weights of the three fits.
NATIVE = dict(lr=4e-3, lr_background=4e-3, schedule=(200, 0.7), background=0.4, background_beta=1e-5)
FINE = dict(lr=1e-3, schedule=(100, 0.7))
FIELD = dict(lr=1e-2, lr_distance=2e-6, schedule=(500, 1.0), tv_object=4e-2, laplacian=1e-3)


def native_box(image, m):
    """Rolling m x m mean: the native pixel aperture on the fine grid.

    The output centre is at index + (m - 1) / 2 - left, including even m."""
    if m == 1:
        return image
    left = (m - 1) // 2
    right = m - 1 - left
    padded = F.pad(image[None, None], (left, right, left, right), mode="replicate")
    return F.avg_pool2d(padded, m, stride=1)[0, 0]


class _EventTerms:
    """The two event terms at fixed transport and contrast, with their sampling and splat
    geometry computed once."""
    def __init__(self, x, y, t, polarity, flow, tref, c, m, sigma):
        self.m, self.c = m, c
        h, w = flow.sensor
        self.h, self.w = h * m, w * m
        self.k = flow.K
        dev = x.device
        # Map the pixel centre onto the rolling box lattice, including even M.
        left = (m - 1) // 2
        xw, yw = flow.warp(x, y, t, tref)
        self.x, self.y = m * xw + left, m * yw + left
        seg = torch.bucketize(t.contiguous(), flow.segment_boundaries[1:-1].to(t), right=True)
        d = flow.segments()
        self.length = d.norm(dim=1)
        u = d / self.length.clamp_min(1e-6)[:, None]
        # IWE term: c times the segmented IWE, each event moved half a step against the transport
        # so that it sits at the centre of the finite forward step.
        self.target = c * render_segments(
            (self.x - .5 * u[seg, 0]) / m, (self.y - .5 * u[seg, 1]) / m,
            t, polarity, flow.sensor, flow.segment_boundaries, M=m, Dxy_pc=d,
            sigma=sigma, device=dev, polar=True)

        native_count = events_to_image_torch(xw, yw, torch.ones_like(x), flow.sensor,
                                             device=dev, polar=False)
        # Support: native pixels with more than two warped events, without the one-pixel border
        # of the finite difference, in segments that move.
        count = (native_count > 2).repeat_interleave(m, 0).repeat_interleave(m, 1)
        count[0, :] = count[-1, :] = False
        count[:, 0] = count[:, -1] = False
        self.mask = count[None] & (self.length[:, None, None] > MIN_FLOW_MAG)

        rows, cols = torch.meshgrid(torch.arange(self.h, device=dev),
                                    torch.arange(self.w, device=dev), indexing="ij")
        self.iwe_grid = torch.stack((
            (cols[None] + u[:, 0, None, None]) * (2.0 / (self.w - 1)) - 1,
            (rows[None] + u[:, 1, None, None]) * (2.0 / (self.h - 1)) - 1), dim=-1)
        # The prediction is blurred by the kernel that rendered the measured IWE.
        size = int(2 * sigma * m + 1) | 1
        self.pad = size // 2
        middle = torch.full((1,), self.pad, device=dev, dtype=torch.long)
        zero = torch.zeros(1, device=dev)
        along = torch.where(self.length < SHORT_SEGMENT_PX, m * sigma / 2, sigma / 2)
        self.kernel = torch.stack([
            gaussian_interpolate_to_image_rot_flow(
                middle, middle, zero, zero, torch.ones_like(zero), zero.new_zeros(size, size),
                along[k], m * sigma / 2, d[k], kernel_size=(size, size))
            for k in range(self.k)])[:, None]
        torch.testing.assert_close(self.kernel.sum(dim=(1, 2, 3)), torch.ones(self.k, device=dev))

        # Pair term: each event with the previous event at its pixel, both inside the image
        # after warping; the residual is splatted at the later event with the same kernel.
        prev = previous_event_indices(x, y, flow.sensor)
        inside = (self.x >= 0) & (self.x <= self.w - 1) & (self.y >= 0) & (self.y <= self.h - 1)
        eligible = (prev >= 0) & inside & inside[prev.clamp_min(0)]
        self.end = torch.where(eligible)[0]
        self.prev = prev[self.end]
        self.sign = (polarity[self.end] > 0).float() * 2 - 1
        if self.end.numel() == 0:
            raise ValueError("No valid same-pixel event pairs.")
        offsets = torch.arange(-self.pad, self.pad + 1, device=dev)
        dx, dy = torch.meshgrid(offsets, offsets, indexing="ij")
        xe, ye, se = self.x[self.end], self.y[self.end], seg[self.end]
        x0, y0 = xe.round(), ye.round()
        ix, iy = x0.long()[:, None] + dx.flatten(), y0.long()[:, None] + dy.flatten()
        ax, ay = ix - xe[:, None], iy - ye[:, None]
        direction = torch.where((self.length[se] < SHORT_SEGMENT_PX)[:, None],
                                u.new_tensor([1., 0.]), u[se])
        ux, uy = direction[:, :1], direction[:, 1:]
        g = torch.exp(-.5 * ((ax * ux + ay * uy) / along[se, None]).square()
                      -.5 * ((-ax * uy + ay * ux) / (m * sigma / 2)).square())
        g = m * g / (g.sum(dim=1, keepdim=True) + 1e-12)
        keep = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        pixels = ((se[:, None] * self.h + iy) * self.w + ix)[keep]
        self.vote_pixels, self.vote_rows = torch.unique(pixels, sorted=True, return_inverse=True)
        self.vote_cols = torch.where(keep)[0]
        self.vote_weights = g[keep]
        self.vote_counts = self.vote_weights.new_zeros(self.vote_pixels.numel())
        self.vote_counts.index_add_(0, self.vote_rows, self.vote_weights)

    def vote_splat(self, r):
        image = r.new_zeros(self.vote_pixels.numel())
        image.index_add_(0, self.vote_rows, self.vote_weights * r[self.vote_cols])
        return image

    def losses(self, intensity):
        """Pair term, IWE term, the pair residuals and the predicted segmented IWE."""
        L = lin_log(native_box(intensity.float(), self.m) * 255).float()
        samples = sample_image_at_events(L, self.x, self.y)
        pairs = samples[self.end] - samples[self.prev] - self.c * self.sign
        votes = self.vote_splat(pairs)
        vote = (votes.square() / (self.vote_counts + EPS)).sum() * self.m / (self.h * self.w)
        fields = F.conv2d(F.pad(L.expand(1, self.k, -1, -1), (self.pad,) * 4, mode="replicate"),
                          self.kernel, groups=self.k)[0]
        shifted = F.grid_sample(fields[:, None], self.iwe_grid, align_corners=True)[:, 0]
        pred = (fields - shifted) * self.length[:, None, None]
        iwe_loss = ((pred - self.target) * self.mask).square().mean() * self.k
        return vote, iwe_loss, pairs, pred


def _upsample(seed, m):
    """Bilinear start of the fit at M; amplitude and phase of a complex field separately."""
    if not seed.is_complex():
        return F.interpolate(seed[None, None], scale_factor=m, mode="bilinear", align_corners=False)[0, 0]
    amplitude, phase = (F.interpolate(x[None, None], scale_factor=m, mode="bilinear", align_corners=True)[0, 0]
                        for x in (seed.abs(), seed.angle()))
    return torch.polar(amplitude, phase)


def _optical_model(sensor, m, distance, device):
    return AngularSpectrum(WAVELENGTH_M, PIXEL_PITCH_M / m, tuple(n * m for n in sensor),
                           distance=distance, device=device, learn_distance=True)


def _fit(events, edi, seed, background, distance, iterations):
    """Minimise frame + IWE + pair + prior from seed, for one of three cases.

    field    a complex seed: object field and propagation distance, amplitude-domain frame term,
             complex-field TV and hologram Laplacian priors;
    native   an intensity at M = 1: intensity and a sparse sensor-fixed background;
    fine     an intensity at M > 1: intensity only, with the log-TV prior and the background held.

    The two event weights are set at the first step, where the pair term is scaled to the value
    of the weighted IWE term, and are then held.
    """
    m = events.m
    holo = seed.is_complex()
    native = m == 1 and not holo
    case = FIELD if holo else (NATIVE if native else FINE)
    state = torch.nn.Parameter(seed.clone())
    bg = torch.nn.Parameter(background.clone(), requires_grad=native)
    propagation = _optical_model(edi.shape, m, distance, edi.device) if holo else None
    groups = [{"params": state, "lr": case["lr"]}]
    if holo:
        groups.append({"params": propagation.distance, "lr": FIELD["lr_distance"]})
    if native:
        groups.append({"params": bg, "lr": NATIVE["lr_background"]})
    optimizer = (AdamP if holo else torch.optim.Adam)(groups)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, *case["schedule"])
    history = []
    for i in range(iterations + 1):
        final = i == iterations
        with torch.set_grad_enabled(not final):
            image = propagation(state).abs().square() if holo else state
            vote, iwe, pairs, pred = events.losses(image)
            if i == 0:
                iwe_weight = IWE_WEIGHT
                if holo:
                    # the weight of the normalized IWE fidelity, 1e3 per fine pixel, carried
                    # over to the contrast-scaled term at the seed
                    a, b = pred.detach() * events.mask, events.target * events.mask
                    shape_loss = (a / a.norm().clamp_min(1e-8) - b / b.norm().clamp_min(1e-8)).square().sum()
                    iwe_weight = float(1e3 / (events.h * events.w) * shape_loss / iwe.detach().clamp_min(1e-12))
                initial = (float(vote.detach()), float(iwe.detach()), iwe_weight)
                if any(not math.isfinite(v) or v <= 1e-12 for v in initial):
                    raise ValueError(f"Cannot balance joint event terms at initialization: {initial}")
                vote_weight = iwe_weight * float(iwe.detach() / vote.detach())
            pooled = down_sampling(image, m) + bg
            if holo:
                frame = ((pooled + 1e-8).sqrt() - (edi + 1e-8).sqrt()).square().mean()
                prior = FIELD["tv_object"] * tv_loss(state, tv_tau=1e-4, iso=False)
                prior = prior + FIELD["laplacian"] * laplacian(image).abs().mean()
            elif native:
                frame = (pooled - edi).square().mean()
                prior = NATIVE["background"] * l1_loss(bg, beta=NATIVE["background_beta"])
            else:
                frame = (pooled - edi).square().mean()
                prior = LOG_TV_WEIGHT * tv_loss((image + LOG_TV_OFFSET).log(), tv_tau=1e-4, iso=True,
                                                spatial_scale=m)
            # Floating-point addition is not associative: each case keeps its own order of
            # summation, on which the stored references depend.
            if holo or native:
                loss = MIXING[0] * iwe_weight * iwe + MIXING[1] * vote_weight * vote + frame + prior
            else:
                loss = frame + MIXING[0] * iwe_weight * iwe + MIXING[1] * vote_weight * vote + prior
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite joint loss at M={m}, step {i}")
        if i % 100 == 0 or final:
            history.append(dict(iteration=i, objective=float(loss.detach()), frame=float(frame.detach()),
                                prior=float(prior.detach()), vote=float(vote.detach()), iwe=float(iwe.detach())))
        if not final:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            if not holo:
                with torch.no_grad():
                    state.clamp_(0, 1.1)
    report = dict(iterations=iterations, pairs=events.end.numel(), history=history,
                  iwe_weight=iwe_weight, ew_vote_weight=vote_weight,
                  pair_rms=float(pairs.square().mean().sqrt()))
    if holo:
        report["z_m"] = float(propagation.distance.detach())
    return state.detach(), image.detach(), bg.detach(), pred.detach(), report


def reconstruct_joint(edi, events, flow, *, contrast, reference_us, scale, object_seed=None,
                      distance_m=None, native_steps=1000, sr_steps=None):
    """Fit the joint objective with fixed linear transport and effective contrast.

    edi is a native-grid float32 intensity anchor. events is (x, y, time, polarity),
    with times in microseconds from APS exposure start. reference_us is the
    exposure midpoint in that clock. A complex native object_seed and a positive
    distance_m select holography. In intensity mode, native_steps=0 uses EDI as
    the SR seed and leaves the native additive background at zero.

    Returns the result arrays and a report of the fits.
    """
    holo = object_seed is not None
    if (flow.basis != "linear" or tuple(edi.shape) != flow.sensor
            or edi.dtype != torch.float32 or edi.device != flow.device):
        raise ValueError("Expected a float32 native EDI anchor and matching linear transport")
    if (scale < 2 or int(scale) != scale or not 0 < contrast or not math.isfinite(contrast)
            or not 0 <= reference_us <= flow.t_win):
        raise ValueError("Expected integer scale >= 2, positive finite contrast and a reference inside the window")
    scale = int(scale)
    if holo and (not object_seed.is_complex() or object_seed.shape != edi.shape
                 or object_seed.device != edi.device or distance_m is None
                 or not math.isfinite(distance_m) or distance_m <= 0):
        raise ValueError("Holography requires a native complex field and positive finite distance")
    if not holo and distance_m is not None:
        raise ValueError("A propagation distance requires a complex object seed")
    if sr_steps is None:
        sr_steps = 2001 if holo else 1000
    if native_steps < 0 or sr_steps < 1:
        raise ValueError("Expected nonnegative native steps and positive SR steps")

    events = tuple(a.to(device=edi.device, dtype=torch.float32) for a in events)
    edi = edi.detach()
    seed = object_seed.detach() if holo else edi
    bg = torch.zeros_like(edi)
    report = dict(mixing=list(MIXING), log_tv_weight=None if holo else LOG_TV_WEIGHT, fits={})
    if not holo and native_steps:
        with torch.no_grad():
            native = _EventTerms(*events, flow, reference_us, contrast, 1, SIGMA)
        seed, _, bg, _, report["fits"]["native"] = _fit(native, edi, seed, bg, None, native_steps)
        del native

    with torch.no_grad():
        plan = _EventTerms(*events, flow, reference_us, contrast, scale, SIGMA if holo else SIGMA_SR)
    sr_seed = _upsample(seed, scale)
    state, image, _, predicted, report["fits"]["sr"] = _fit(plan, edi, sr_seed, bg, distance_m, sr_steps)
    arrays = dict(edi=edi, background=bg, sr_seed=sr_seed, joint=image, iwe_measured=plan.target,
                  iwe_predicted=predicted, iwe_support=plan.mask,
                  iwe_residual=(predicted - plan.target) * plan.mask)
    if holo:
        arrays.update(object_m1=seed, object_sr=state)
    else:
        arrays["intensity_m1"] = seed
    return arrays, report
