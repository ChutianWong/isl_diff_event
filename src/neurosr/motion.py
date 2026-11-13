"""Transport: a continuous path of spatially uniform displacement over one recording window.

The window is divided into K equal segments.  The path r(t) is a clamped B-spline of the K
per-segment increments; the default linear basis is the continuous piecewise-linear trajectory
with piecewise-constant velocity.  An event at (x, t) is warped to the reference time t0 by
x' = x - r(t) + r(t0).
"""
import numpy as np
import torch
import torch.nn as nn

from ._constants import MIN_FLOW_MAG, SENSOR
from .ops import flow_regularization_weights, track_position, window_event_to_img


def bspline_increment_basis(num_pieces, degree, u):
    """[len(u), K] map from K per-piece increments to displacement at parameter u in [0, spans].

    Clamped uniform B-spline of the given degree with the K + 1 cumulative control points
    P_i = sum_{j < i} F_j.  Degree 1 is the straight chord, degree 3 the cubic B-spline,
    degree K the Bezier curve; the ends are interpolated for every degree, so D(0) = 0 and
    D(t_win) = sum_k F_k.
    """
    K = int(num_pieces)
    p = int(min(degree, K))
    n = K + 1
    spans = n - p
    knots = np.concatenate([np.zeros(p + 1), np.arange(1, spans), np.full(p + 1, spans)]).astype(np.float64)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    N = np.zeros((u.size, knots.size - 1))
    for i in range(knots.size - 1):
        N[:, i] = (u >= knots[i]) & (u < knots[i + 1])
    at_end = u >= knots[-1]
    N[at_end, :] = 0.0
    N[at_end, np.flatnonzero(np.diff(knots) > 0)[-1]] = 1.0
    for q in range(1, p + 1):
        Nq = np.zeros((u.size, knots.size - 1 - q))
        for i in range(knots.size - 1 - q):
            left = knots[i + q] - knots[i]
            right = knots[i + q + 1] - knots[i + 1]
            a = (u - knots[i]) / left if left > 0 else 0.0
            b = (knots[i + q + 1] - u) / right if right > 0 else 0.0
            Nq[:, i] = a * N[:, i] + b * N[:, i + 1]
        N = Nq
    C = np.tril(np.ones((n, K)), -1)
    return N @ C


class SplineFlow(nn.Module):
    """r(t) = sum_k W[t, k] F_k, in sensor pixels, with t in microseconds from the window origin.

    knots [K, 2, 1, 1] are the per-piece increments F_k (x, y).  The trailing singleton axes are
    the spatial lattice of the general formulation, which holds a single node here: the
    transport is spatially uniform.  A constant-velocity straight path has constant increments
    in the linear and Bezier bases; in the B-spline basis only for K <= 3, where it is the
    Bezier curve.
    """
    DEGREE = {"linear": 1, "bspline": 3}

    def __init__(self, t_win, num_pieces, sensor=SENSOR, basis="linear", device="cuda"):
        super().__init__()
        self.t_win = int(t_win)
        self.K = int(num_pieces)
        self.sensor = tuple(int(v) for v in sensor)
        if self.K < 1 or self.t_win < 1 or min(self.sensor) < 2:
            raise ValueError("Positive duration, pieces and sensor dimensions >= 2 are required")
        if basis not in ("linear", "bspline", "bezier"):
            raise ValueError("basis must be linear, bspline or bezier")
        self.basis = basis
        self.degree = min(self.K if basis == "bezier" else self.DEGREE[basis], self.K)
        spans = self.K + 1 - self.degree
        W = bspline_increment_basis(self.K, self.degree, np.arange(self.t_win + 1) * (spans / self.t_win))
        A = bspline_increment_basis(self.K, self.degree, np.arange(self.K + 1) * (spans / self.K))
        self.register_buffer("W", torch.as_tensor(W, dtype=torch.float32, device=device))
        self.register_buffer("A", torch.as_tensor(A[1:] - A[:-1], dtype=torch.float32, device=device))
        self.register_buffer("segment_boundaries", torch.linspace(0, self.t_win, self.K + 1, device=device,
                                                                  dtype=torch.float32))
        self.knots = nn.Parameter(torch.zeros(self.K, 2, 1, 1, device=device, dtype=torch.float32))

    @property
    def device(self):
        return self.knots.device

    def set_increments(self, increments):
        """Load [K, 2] control increments."""
        with torch.no_grad():
            value = torch.as_tensor(increments, dtype=self.knots.dtype, device=self.device)
            if value.ndim == 2:
                value = value[:, :, None, None]
            self.knots.copy_(value.expand_as(self.knots))
        return self

    def set_segments(self, segments):
        """Load the path from its [K, 2] displacements over the segments: the control increments
        solve A F = S, which is the identity for the linear basis."""
        S = torch.as_tensor(segments, dtype=torch.float32, device=self.device).reshape(self.K, 2)
        return self.set_increments(torch.linalg.solve(self.A.double(), S.double()).float())

    def rows(self, t):
        """W at real times, [N, K]: linear between the microsecond samples."""
        t = torch.as_tensor(t, dtype=self.W.dtype, device=self.device).reshape(-1).clamp(0, self.t_win)
        i0 = t.floor().long()
        i1 = (i0 + 1).clamp(max=self.t_win)
        w = (t - i0.to(t.dtype))[:, None]
        return self.W[i0] + w * (self.W[i1] - self.W[i0])

    def sample(self, x, y):
        """F_k at sensor coordinates (x, y), [N, K, 2]."""
        n = x.numel()
        return self.knots[:, :, 0, 0].expand(n, self.K, 2)

    def warp(self, xt, yt, tt, t0):
        """x' = x - r(t) + r(t0): the events moved to the reference time t0."""
        Fe = self.sample(xt, yt)
        D = torch.einsum("nk,nkc->nc", self.rows(tt), Fe)
        D0 = torch.einsum("k,nkc->nc", self.rows(t0)[0], Fe)
        return xt - D[:, 0] + D0[:, 0], yt - D[:, 1] + D0[:, 1]

    def segments(self):
        """Displacement over each time segment, [K, 2] in sensor pixels."""
        return self.A @ self.knots[:, :, 0, 0]

    def prior(self, lam, beta=1e-4):
        """lam * sum_k |S_{k+1} - S_k|^2 + beta * sum_k |S_k|^2 on the segment displacements
        S = A F of the path (S is the knots for the linear basis).  The prior acts on the
        physical path, so at a fixed piece count it means the same for every basis."""
        S = torch.einsum("kj,jchw->kchw", self.A, self.knots)
        loss = lam * ((S[1:] - S[:-1]) ** 2).sum((0, 1)).mean() + beta * (S ** 2).sum((0, 1)).mean()
        return loss

    def tv_weights(self, min_flow_mag=MIN_FLOW_MAG):
        """The 2 x 2 direction weights of the transverse total variation, tv_loss_flow."""
        return flow_regularization_weights(self.segments().detach(), min_flow_mag=min_flow_mag)

    def param_groups(self, lr):
        return [{"params": [self.knots], "lr": lr}]

    def state(self):
        seg = self.segments().detach()
        return {"basis": self.basis, "degree": self.degree, "num_pieces": self.K,
                "segments_px": seg.cpu().tolist(), "total_px": seg.sum(0).cpu().tolist()}


def register_segments(xs, ys, ts, ps, flow):
    """Initial transport from the events alone.

    K + 1 signed event images, each one segment long and centred on a segment boundary, are
    registered to the first by phase correlation.  Returns the successive displacement
    differences [K, 2] (x, y) in sensor pixels.
    """
    boundaries = flow.segment_boundaries.detach().cpu().numpy()
    width = np.float32(flow.t_win / flow.K)
    iwes = [window_event_to_img(xs, ys, ts, ps, flow.sensor, s=boundaries[i] - width // 2, win=width)
            for i in range(flow.K + 1)]
    iwes_t = torch.from_numpy(np.asarray(iwes)).to(flow.device)
    pos = torch.from_numpy(track_position(iwes_t.double())).float().to(flow.device)
    xy = (-pos[[1, 0], :]).T
    return xy[1:, :] - xy[:-1, :]
