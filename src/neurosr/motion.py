"""Linear-piece transport and image registration for one recording window."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._constants import *
from .ops import window_event_to_img, flow_regularization_weights

def bspline_increment_basis(num_pieces, degree, u):
    """[len(u), K] map from K per-piece increments to displacement at parameter u in [0, spans].

    Clamped uniform B-spline of the given degree with the K + 1 cumulative control points
    P_i = sum_{j < i} F_j.  Degree 1 is the straight chord (ContinuousPiecewiseLinear_Dxy_pw),
    degree 3 the cubic B-spline, degree K the Bezier curve; the ends are interpolated for
    every degree, so D(0) = 0 and D(t_win) = sum_k F_k.
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
    """D(x, t) = sum_k W[t, k] F_k(x).

    knots [K, 2, th, tw] are per-piece increments on a tile lattice whose corners sit on the
    sensor corners; the dense field is their bilinear interpolation.  tiles = (1, 1) gives
    spatially uniform motion. Times are microseconds relative to
    the window origin. Bezier has degree K; fitting noisy samples with a high-degree curve
    can require large control increments. A constant-velocity straight path has constant
    increments in the linear and Bezier bases; in the B-spline basis only for K <= 3, where it
    is the Bezier curve.
    """
    DEGREE = {"linear": 1, "bspline": 3}

    def __init__(self, t_win, num_pieces, sensor=SENSOR, basis="linear", tiles=(1, 1), device="cuda"):
        super().__init__()
        self.t_win = int(t_win)
        self.K = int(num_pieces)
        self.sensor = tuple(int(v) for v in sensor)
        if self.K < 1 or self.t_win < 1 or min(self.sensor) < 2 or min(tiles) < 1:
            raise ValueError("Positive duration, pieces, tiles and sensor dimensions >= 2 are required")
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
        self.knots = nn.Parameter(torch.zeros(self.K, 2, *tiles, device=device, dtype=torch.float32))

    @property
    def tiles(self):
        return tuple(int(v) for v in self.knots.shape[2:])

    @property
    def uniform(self):
        return self.tiles == (1, 1)

    @property
    def device(self):
        return self.knots.device

    def set_increments(self, increments):
        """Load [K, 2] or [K, 2, th, tw] control increments."""
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
        if self.uniform:
            return self.knots[:, :, 0, 0].expand(n, self.K, 2)
        H, Wd = self.sensor
        gx = 2 * x.reshape(-1).to(self.knots) / (Wd - 1) - 1
        gy = 2 * y.reshape(-1).to(self.knots) / (H - 1) - 1
        grid = torch.stack((gx, gy), -1).reshape(1, 1, n, 2)
        out = F.grid_sample(self.knots.reshape(1, 2 * self.K, *self.tiles), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.reshape(self.K, 2, n).permute(2, 0, 1)




    def warp(self, xt, yt, tt, t0):
        """x_w = x - D(x, t) + D(x, t0): the flow read at the event's own pixel."""
        Fe = self.sample(xt, yt)
        D = torch.einsum("nk,nkc->nc", self.rows(tt), Fe)
        D0 = torch.einsum("k,nkc->nc", self.rows(t0)[0], Fe)
        return xt - D[:, 0] + D0[:, 0], yt - D[:, 1] + D0[:, 1]

    def segments(self, magnification=1):
        """Displacement over each time segment: [K, 2] when uniform, else [K, 2, M H, M W]."""
        if self.uniform:
            return self.A @ self.knots[:, :, 0, 0]
        return torch.einsum("kj,jchw->kchw", self.A, self.field(magnification))

    def prior(self, lam, beta=1e-4, mu=0.0):
        """lam * sum_k |S_{k+1} - S_k|^2 + beta * sum_k |S_k|^2 on the segment displacements
        S = A F of the path (S is the knots for the linear basis),
        averaged over the tile nodes so that the weight does not grow with the tile count (refinement
        can still change the value), plus mu times the total variation of the tile field over the
        sensor.  The prior acts on the physical path, so at a fixed piece count it means the same
        for every basis."""
        S = torch.einsum("kj,jchw->kchw", self.A, self.knots)
        loss = lam * ((S[1:] - S[:-1]) ** 2).sum((0, 1)).mean() + beta * (S ** 2).sum((0, 1)).mean()
        return loss

    def tv_weights(self, magnification=1, min_flow_mag=MIN_FLOW_MAG):
        """Direction weights for tv_loss_flow: one 2x2 when uniform, one per pixel otherwise."""
        if self.uniform:
            return flow_regularization_weights(self.segments().detach(), min_flow_mag=min_flow_mag)
        return flow_regularization_weights_dense(self.segments(magnification).detach(), min_flow_mag=min_flow_mag)

    def param_groups(self, lr):
        return [{"params": [self.knots], "lr": lr}]

    def project(self):
        """The knots are unbounded."""



    def state(self):
        seg = self.segments().detach()
        if seg.ndim == 4:
            seg = seg.mean((2, 3))
        return {"basis": self.basis, "degree": self.degree, "num_pieces": self.K, "tiles": list(self.tiles),
                "segments_px": seg.cpu().tolist(), "total_px": seg.sum(0).cpu().tolist()}


def register_segments(xs, ys, ts, ps, flow):
    """Register K + 1 boundary-centred event images with Simple_sr; return successive displacement
    differences [K, 2] in sensor pixels."""
    from .ops import Simple_sr
    boundaries = flow.segment_boundaries.detach().cpu().numpy()
    width = np.float32(flow.t_win / flow.K)
    iwes = [window_event_to_img(xs, ys, ts, ps, flow.sensor, s=boundaries[i] - width // 2, win=width)
            for i in range(flow.K + 1)]
    iwes_t = torch.from_numpy(np.asarray(iwes)).to(flow.device)
    sr = Simple_sr(holo_raw=iwes_t, img_size=np.asarray(flow.sensor), mag=1,
                   img_num=flow.K + 1, device=flow.device)
    sr.pos_registration()
    pos = torch.from_numpy(sr.pos).float().to(flow.device)
    xy = (-pos[[1, 0], :]).T
    return xy[1:, :] - xy[:-1, :]


