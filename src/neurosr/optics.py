"""Angular-spectrum propagation; lengths are metres and fields are complex.

IdentityOptics is retained as the original solver type-check sentinel.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._constants import *

class IdentityOptics(nn.Module):
    """Identity on sensor-plane intensity, already formed by the imaging system.

    This representation does not assert a particular relay, objective or magnification.
    It is separate from the displacement model and the event generative model.
    """

    def forward(self, intensity):
        if intensity.is_complex():
            raise ValueError("IdentityOptics expects sensor-plane intensity, not a complex field")
        return intensity

    def at_scale(self, magnification):
        return self


class AngularSpectrum(nn.Module):
    """Free-space propagation by the angular spectrum, reflect padded by a quarter on each side.
    I = |P_z u|^2 is then the latent the sensor sees.  The distance is
    a float32 parameter, and a distance passed to forward is taken in float64."""

    def __init__(self, wavelength, pitch, sensor=SENSOR, distance=0.0, pad=True, device="cuda",
                 learn_distance=False):
        super().__init__()
        ny, nx = (int(v) for v in sensor)
        self.dim = (ny, nx)
        self.pad = bool(pad)
        self.wavelength, self.pitch = float(wavelength), float(pitch)
        self.learn_distance = bool(learn_distance)
        self.distance = nn.Parameter(torch.as_tensor(distance, device=device, dtype=torch.float32).detach().clone(),
                                     requires_grad=self.learn_distance)
        px, py = (nx + 2 * (nx // 4), ny + 2 * (ny // 4)) if pad else (nx, ny)
        fx = (torch.arange(px, dtype=torch.float64) - px // 2) / (px * pitch)
        fy = (torch.arange(py, dtype=torch.float64) - py // 2) / (py * pitch)
        mfx, mfy = torch.meshgrid(fx, fy, indexing="xy")
        kz = torch.sqrt(torch.clamp(1 - wavelength * wavelength * (mfx * mfx + mfy * mfy), min=0.0))
        self.register_buffer("kz", (1j * (2 * np.pi / wavelength) * kz).to(device))

    def forward(self, u, distance=None):
        z = self.distance if distance is None else torch.as_tensor(distance, dtype=torch.float64, device=u.device)
        ny, nx = self.dim
        if tuple(u.shape[-2:]) != self.dim:
            raise ValueError(f"AngularSpectrum expects {self.dim}, got {tuple(u.shape[-2:])}")
        lead = u.shape[:-2]
        u = u.reshape(-1, ny, nx)
        if self.pad:
            u = F.pad(u.unsqueeze(1), (nx // 4, nx // 4, ny // 4, ny // 4), "reflect").squeeze(1)
        dims = (-2, -1)
        spec = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(u, dim=dims)), dim=dims) * torch.exp(z * self.kz)
        out = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(spec, dim=dims)), dim=dims)
        if self.pad:
            out = out[:, ny // 4:ny + ny // 4, nx // 4:nx + nx // 4]
        return out.reshape(*lead, ny, nx)

    def at_scale(self, magnification):
        """Same physical field of view and distance, with pitch divided by M."""
        M = int(magnification)
        new = AngularSpectrum(self.wavelength, self.pitch / M,
                              sensor=tuple(M * n for n in self.dim), distance=self.distance,
                              pad=self.pad, device=self.kz.device, learn_distance=self.learn_distance)
        new.distance = self.distance
        return new

    def autofocus(self, frame, distances):
        """Sets the distance to the one whose back-propagated intensity has the largest spread of
        its absolute Laplacian, and
        returns it with the scores."""
        I = frame.float() / (frame.float().max() + 1e-8)
        u = torch.sqrt(I)
        scores = []
        with torch.no_grad():
            for z in distances:
                edge = laplacian(self(u, -float(z)).abs() ** 2).abs()
                scores.append(torch.sqrt(torch.var(edge)).item())
            best = float(distances[int(np.argmax(scores))])
            self.distance.copy_(torch.as_tensor(best, dtype=torch.float32))
        return best, np.asarray(scores)


def laplacian(I):
    """kornia.filters.laplacian(I, 3): the 3 x 3 Laplacian divided by 16, reflect padded."""
    kernel = torch.ones(3, 3, dtype=I.dtype, device=I.device)
    kernel[1, 1] = -8
    return F.conv2d(F.pad(I[None, None], (1, 1, 1, 1), mode="reflect"), (kernel / 16)[None, None])[0, 0]


