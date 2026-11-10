"""One experiment object: acquisition, motion, optics and the three original solver stages.

The extracted numerical methods retain their original defaults and comments.
Only the from_directory/run path, with the settings in data/_template, is supported.
"""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._constants import *
from torch.optim import Adam, lr_scheduler
from .ops import (AdamP, np_to_torch, tv_loss, tv_loss_flow, down_sampling, l2_loss, reset_optimizer_and_params)
from .motion import SplineFlow, register_segments
from .events import EventModel
from .frame import FrameModel
from .optics import AngularSpectrum, IdentityOptics, laplacian
from .config import load_config
from .data import load_window_npz

class NeuroSR:
    """The reconstruction stages on one motion model.

    cmax   events only: contrast maximisation at t_ref (default 0 us), knot prior lam = 4e-1
    joint  M = 1: event residual + focus + knot prior + frame residual + background prior
    sr     M > 1: motion, c and background frozen; the latent against the SR target

    Times are on the caller's event clock (load_window zeroes it at the window's first event);
    with a frame the latent sits at the nominal exposure midpoint t0 = win_frame // 2.
    With optics, optimize u with latent |P_z u|^2 and object priors at both stages; optimize
    distance when it is learnable and its stage rate is nonzero.  Project |u| to
    [0, 1.1]; without optics, project intensity to [0, 1.1].  optimizer serves joint and sr.
    kappa pnp / pnp_sr add the DRUNet prior with the pnp schedule, and its denoised latent replaces
    the stage's result.
    """

    def __init__(self, flow, event, frame, optics=None, t0=None, kappa=KAPPA, lr=LR, schedule=SCHEDULE,
                 medi=None, medi_weight=1.0, optimizer=Adam, pnp=PNP):
        self.flow, self.event, self.frame, self.optics = flow, event, frame, optics
        self.field_optics = optics is not None and not isinstance(optics, IdentityOptics)
        # the latent sits at the nominal exposure midpoint; t0 is free only without a frame
        self.t0 = (0 if t0 is None else t0) if frame is None else frame.win_frame // 2
        if frame is not None and t0 is not None and float(t0) != self.t0:
            raise ValueError("The EDI latent reference must be the exposure centre")
        if medi is not None and medi.raw_c is not event.raw_c:
            raise ValueError("mEDI and EventModel must share the same raw_c Parameter")
        self.medi, self.medi_weight = medi, float(medi_weight)
        self.c_init = float(event.contrast.detach())
        self.optimizer = optimizer
        self.kappa = dict(KAPPA, **kappa)
        self.lr = dict(LR, **lr)
        self.schedule = dict(SCHEDULE, **schedule)
        self.pnp, self.pnp_history = dict(PNP, **pnp), {}
        if self.field_optics and (self.kappa["pnp"] or self.kappa["pnp_sr"]):
            raise ValueError("The PnP prior acts on the sensor latent; it has no complex-field form")
        self.history, self.timings = {}, {}
        self.cmax_focus, self.curriculum, self._cmax_iteration = "variance", [], 0
        self.latent_sr = self.object_m1 = self.object_sr = None
        self.target_m1 = self.pred_m1 = self.target_sr = self.pred_sr = None
        self.sr_magnification = self.sr_sigma = None
        self.latent_poisson = self.support_sr = None
        self.sr_options = {}

    def _log(self, stage, iteration, values):
        self.history.setdefault(stage, []).append(dict(iteration=iteration, **values))
        print(stage + " iter = {}: ".format(iteration) + ", ".join("%s = %s" % (k, v) for k, v in values.items()),
              flush=True)

    def _latent(self, variable, P):
        """The latent intensity: the variable itself, or |P u|^2 on the sensor."""
        if P is None or isinstance(P, IdentityOptics):
            return variable if P is None else P(variable)
        U = P(variable)
        return (U * U.conj()).abs()

    def _variable(self, I0, P):
        """The latent itself, or the complex64 object field: given, or sqrt(I0) back-propagated
        over the current distance."""
        if P is None or isinstance(P, IdentityOptics):
            if I0.is_complex():
                raise ValueError("Intensity optics require a real sensor-plane initial latent")
            return nn.Parameter(I0.clone().detach(), requires_grad=True)
        if not I0.is_complex():
            with torch.no_grad():
                I0 = P(I0.detach().clamp_min(0).sqrt(), -P.distance.detach())
        return nn.Parameter(I0.detach().clone().to(torch.complex64), requires_grad=True)

    def _optical_groups(self, lr):
        """The distance is optimised when it is learnable and its rate is nonzero, else held."""
        if self.optics is None:
            return []
        parameters = list(self.optics.parameters())
        learn = bool(getattr(self.optics, "learn_distance", False)) and lr != 0
        for parameter in parameters:
            parameter.requires_grad_(learn)
        return [{"params": parameters, "lr": lr}] if learn and parameters else []

    def _object_prior(self, I, field):
        """Mean |Laplacian| of the latent hologram and anisotropic TV of the complex field."""
        prior = self.kappa["laplacian"] * (laplacian(I).abs().sum(dim=(-1, -2)) / I.shape[0] / I.shape[1])
        return prior + self.kappa["tv_object"] * tv_loss(field, tv_order=1, tv_tau=1e-4, iso=False)

    def _image_priors(self, L_tot, I, magnification, stage):
        """L_tot plus the nonzero image priors of the stage (the _sr weights at SR): iso_tv the
        isotropic TV and l0 the smooth L0 at spatial scale M, flow_l0 the
        smooth L0 across the window's net displacement at unit scale."""
        k, s, M = self.kappa, "_sr" if stage == "sr" else "", int(magnification)
        if k["iso_tv" + s]:
            L_tot = L_tot + k["iso_tv" + s] * tv_loss(I, tv_order=1, tv_tau=1e-4, iso=True, spatial_scale=M)
        if k["l0" + s]:
            L_tot = L_tot + k["l0" + s] * smooth_l0(I, alpha=30.0, spatial_scale=M)
        if k["flow_l0" + s]:
            if not self.flow.uniform:
                raise ValueError("flow_l0 takes one direction, the net displacement of a uniform path")
            d = self.flow.segments().sum(0).detach()
            L_tot = L_tot + k["flow_l0" + s] * smooth_l0_flow(I, alpha=30.0,
                                                              d_hat=d / (torch.linalg.vector_norm(d) + 1e-9))
        return L_tot

    def _pnp(self, stage, magnification, iterations):
        """The DRUNet prior of the stage (function/pnp_prior.py with the pnp schedule), or None."""
        weight = self.kappa["pnp_sr" if stage == "sr" else "pnp"]
        if not weight:
            return None

    def _pnp_finalize(self, prior, variable):
        """The denoised latent after the last outer step replaces the variable."""
        if prior is not None:
            with torch.no_grad():
                variable.data = prior.finalize_intensity(self.event.log(variable) / np.log(255)).to(variable.dtype)

    def _project(self, variable):
        with torch.no_grad():
            if not self.field_optics:
                variable.clamp_(0, 1.1)
            else:
                variable.copy_(variable.abs().clamp(0, 1.1) * torch.exp(1j * variable.angle()))

    def cmax(self, iterations=2100, lam=4e-1, beta=1e-4, t_ref=0.0, log_every=100,
             focus="variance", mu=0.0):
        start = time.time()
        if self.event.tt.numel() == 0:
            raise ValueError("CMax needs nonempty events")
        for parameter in self.flow.parameters():
            parameter.requires_grad_(True)
        self.cmax_focus = focus
        offset = self._cmax_iteration
        groups = self.flow.param_groups(self.lr["cmax"])
        optimizer = Adam(groups)
        scheduler = lr_scheduler.StepLR(optimizer, *self.schedule["cmax"])
        for iteration in range(iterations):
            optimizer.zero_grad()
            loss = self.cmax_objective(lam, beta, t_ref, focus, mu)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CMax objective")
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.flow.project()
            if iteration % log_every == 0:
                grad = torch.cat([p.grad.reshape(-1) for p in self.flow.parameters()]).mean().item()
                self._log("cmax", offset + iteration, {"loss": loss.item(), "grad": grad})
        reset_optimizer_and_params(optimizer, groups)
        self._cmax_iteration = offset + iterations
        self.timings["cmax_s"] = time.time() - start
        return self.flow

    def cmax_objective(self, lam=4e-1, beta=1e-4, t_ref=0.0, focus="variance", mu=0.0):
        """-var(count image warped to t_ref) + knot prior."""
        if focus == "variance":
            xw, yw = self.flow.warp(self.event.xt, self.event.yt, self.event.tt, t_ref)
            data = self.event.focus(xw, yw)
        elif focus == "secrets":
            data = self.event.multi_reference_focus(self.flow)
        else:
            raise ValueError("focus must be variance or secrets")
        return data + self.flow.prior(lam, beta, mu)


    def joint_objective(self, I, lam=2e-1, beta=1e-4, mu=0.0, target=None, field=None):
        """The M = 1 objective and its parts: event residual, focus, knot prior, frame residual,
        transverse TV (weight 0 by default), background sparsity, and with a given object field
        the object priors.  A supplied target assumes fixed motion and omits its focus and prior."""
        kappa, M = self.kappa, 1
        moving = target is None
        if moving:
            xw, yw = self.flow.warp(self.event.xt, self.event.yt, self.event.tt, self.t0)
            target = self.event.target(xw, yw, self.flow, M)
        L_D, pred, _ = self.event.residual(I, self.flow, M, target)
        L_F = self.frame.residual(I, M, kernel=self.frame.kernel(self.flow, M, self.t0) if self.frame.reblur else None)
        L_tot = kappa["event"] * L_D
        if moving:
            L_tot = L_tot + kappa["sog"] * self.event.focus(xw, yw)
            L_tot = L_tot + kappa["sog"] * self.flow.prior(lam, beta, mu)
        L_tot = L_tot + kappa["frame"] * L_F
        if kappa["tv"]:
            L_tot = L_tot + kappa["tv"] * tv_loss_flow(I, tv_order=1, tv_tau=1e-4, iso=True, spatial_scale=M,
                                                       direction_weights=self.flow.tv_weights(M))
        L_tot = self._image_priors(L_tot, I, M, "joint")
        L_tot = L_tot + kappa["bg"] * self.frame.prior()
        if self.field_optics and field is not None:
            L_tot = L_tot + self._object_prior(I, field)
        parts = {"event": L_D, "frame": L_F}
        if self.medi is not None and self.medi_weight:
            parts["medi"] = self.medi.loss()
            L_tot = L_tot + self.medi_weight * parts["medi"]
        return L_tot, parts, target, pred

    def joint(self, I0, iterations=1900, lam=2e-1, beta=1e-4, log_every=100, mu=0.0):
        """lr["motion"] = 0 holds the motion, and the target is then rendered once."""
        start = time.time()
        moving = self.lr["motion"] != 0
        for parameter in self.flow.parameters():
            parameter.requires_grad_(moving)
        for parameter in (self.event.raw_c, self.frame.background):
            parameter.requires_grad_(True)
        fixed_target = None
        if not moving:
            with torch.no_grad():
                xw, yw = self.flow.warp(self.event.xt, self.event.yt, self.event.tt, self.t0)
                fixed_target = self.event.target(xw, yw, self.flow, 1)
        variable = self._variable(I0, self.optics)
        groups = ([{"params": variable, "lr": self.lr["image"]},
                   {"params": self.frame.background, "lr": self.lr["background"]}]
                  + self.flow.param_groups(self.lr["motion"])
                  + [{"params": self.event.raw_c, "lr": self.lr["contrast"]}]
                  + self._optical_groups(self.lr["distance"]))
        optimizer = self.optimizer(groups)
        scheduler = lr_scheduler.StepLR(optimizer, *self.schedule["joint"])
        pnp = self._pnp("joint", 1, iterations)
        for iteration in range(iterations):
            optimizer.zero_grad()
            I = self._latent(variable, self.optics)
            L_tot, parts, target, pred = self.joint_objective(I, lam, beta, mu, fixed_target, variable)
            if pnp is not None:
                L_tot = L_tot + pnp.loss(self.event.log(I) / np.log(255), iteration)
            if not torch.isfinite(L_tot):
                raise FloatingPointError("Nonfinite joint objective")
            L_tot.backward()
            optimizer.step()
            scheduler.step()
            self._project(variable)
            self.flow.project()
            if iteration == 1 or (iteration % log_every == 0 and iteration > 0):
                self._log("joint", iteration, dict(loss=L_tot.item(),
                          **{key: value.item() for key, value in parts.items()}, c=self.event.contrast.item()))
        self._pnp_finalize(pnp, variable)
        reset_optimizer_and_params(optimizer, groups)
        self.latent_m1 = self._latent(variable, self.optics).detach()
        if self.field_optics:
            self.object_m1 = variable.detach().clone()
        with torch.no_grad():
            _, _, target, pred = self.joint_objective(self.latent_m1, lam, beta, mu, fixed_target, variable)
        self.target_m1, self.pred_m1 = target.detach(), pred.detach()
        self.timings["joint_s"] = time.time() - start
        return self.latent_m1

    def sr_objective(self, I, magnification, target, contrast, weights, latent, field=None, kernel=None,
                     anchor=None):
        """The SR objective at fixed motion, c and background: event residual, frame residual against
        the fixed MEDI latent, transverse TV along the fixed flow, the other image priors, and with
        optics the object priors.  A given anchor image replaces the frame residual by
        |pool(I) - anchor|^2.  sr() optimizes the latent, or the object
        field and a learnable distance."""
        kappa, M = self.kappa, int(magnification)
        kappa_event = kappa["event"] if kappa["event_sr"] is None else kappa["event_sr"]
        L_D, pred, _ = self.event.residual(I, self.flow, M, target, contrast=contrast)
        if anchor is None:
            L_F = self.frame.residual(I, M, latent=latent, kernel=kernel)
        else:
            L_F = l2_loss(down_sampling(I, M) - anchor)
        L_tot = kappa_event * L_D + kappa["frame"] * L_F
        L_tot = L_tot + kappa["tv_sr"] * tv_loss_flow(I, tv_order=1, tv_tau=1e-4, iso=True, spatial_scale=M,
                                                      direction_weights=weights)
        L_tot = self._image_priors(L_tot, I, M, "sr")
        if self.field_optics and field is not None:
            L_tot = L_tot + self._object_prior(I, field)
        return L_tot, {"event": L_D, "frame": L_F}, pred

    def sr(self, I_m1, magnification=3, iterations=1000, sigma=SIGMA_SR, log_every=100, init="bilinear",
           anchor="frame", support=None):
        """init "nearest" upsamples the M = 1 latent by repetition.  anchor "m1"
        compares the pooled latent with the M = 1 latent in place of the frame.  support "events" updates
        only the fine pixels the warped events touch and adds latent_poisson, the latent rebuilt from
        its gradients between such pixels (EPBA semi-dense support and densification)."""
        start = time.time()
        M = int(magnification)
        if M < 1:
            raise ValueError("magnification must be a positive integer")
        if init not in ("bilinear", "nearest") or anchor not in ("frame", "m1") or support not in (None, "events"):
            raise ValueError("init bilinear/nearest, anchor frame/m1, support None/events")
        self.sr_options = dict(init=init, anchor=anchor, support=support)
        self.sr_magnification = M
        self.sr_sigma = self.event.sigma if M == 1 else float(sigma)
        if M == 1:
            self.latent_sr = I_m1.detach().clone()
            self.target_sr, self.pred_sr = self.target_m1, self.pred_m1
            self.timings["sr_s"] = 0.0
            return self.latent_sr
        H, Wd = self.event.sensor
        for parameter in (*self.flow.parameters(), self.event.raw_c, self.frame.background):
            parameter.requires_grad_(False)
        with torch.no_grad():
            xw, yw = self.flow.warp(self.event.xt, self.event.yt, self.event.tt, self.t0)
            target = self.event.target(xw, yw, self.flow, M, sigma=sigma)
            contrast = self.event.contrast.detach()
            weights = self.flow.tv_weights(M)
            latent = self.frame.latent().detach()
            kernel = self.frame.kernel(self.flow, M, self.t0) if self.frame.reblur and anchor == "frame" else None
            if init == "bilinear":
                I0 = F.interpolate(I_m1[None, None], size=(H * M, Wd * M), mode="bilinear",
                                   align_corners=False).squeeze()
            else:
                I0 = F.interpolate(I_m1[None, None], size=(H * M, Wd * M), mode="nearest").squeeze()
            mask = None
            if support == "events":
                mask = (events_to_image_torch(*self.event.fine(xw, yw, M), self.event.pt, (H * M, Wd * M),
                                              device=self.event.device, polar=False) > 0).float()
        anchor_image = I_m1.detach() if anchor == "m1" else None
        # the same field of view on an M-times finer pitch; the distance Parameter is shared
        P = self.optics.at_scale(M) if isinstance(self.optics, AngularSpectrum) else self.optics
        if self.object_m1 is not None:
            # amplitude and phase upsampled separately
            u, size = self.object_m1, (H * M, Wd * M)
            a = F.interpolate(u.abs()[None, None], size=size, mode="bilinear", align_corners=True)[0, 0]
            phi = F.interpolate(u.angle()[None, None], size=size, mode="bilinear", align_corners=True)[0, 0]
            I0 = a.to(torch.complex64) * torch.exp(1j * phi.to(torch.complex64))
        variable = self._variable(I0, P)
        groups = [{"params": variable, "lr": self.lr["sr"]}] + self._optical_groups(self.lr["distance_sr"])
        optimizer = self.optimizer(groups)
        scheduler = lr_scheduler.StepLR(optimizer, *self.schedule["sr"])
        pnp = self._pnp("sr", M, iterations)
        for iteration in range(iterations):
            optimizer.zero_grad()
            I = self._latent(variable, P)
            L_tot, parts, pred = self.sr_objective(I, M, target, contrast, weights, latent, variable, kernel,
                                                   anchor_image)
            if pnp is not None:
                # on the event pixel's response
                L_tot = L_tot + pnp.loss(self.event.log(self.event.response(I, M)) / np.log(255), iteration)
            if not torch.isfinite(L_tot):
                raise FloatingPointError("Nonfinite SR objective")
            L_tot.backward()
            if mask is not None:
                variable.grad *= mask
            optimizer.step()
            scheduler.step()
            self._project(variable)
            if iteration == 1 or (iteration % log_every == 0 and iteration > 0):
                self._log("sr", iteration, {"loss": L_tot.item(), "event": parts["event"].item(),
                                            "frame": parts["frame"].item()})
        self._pnp_finalize(pnp, variable)
        reset_optimizer_and_params(optimizer, groups)
        self.latent_sr = self._latent(variable, P).detach()
        if self.field_optics:
            self.object_sr = variable.detach().clone()
        with torch.no_grad():
            _, _, pred = self.sr_objective(self.latent_sr, M, target, contrast, weights, latent, variable, kernel,
                                           anchor_image)
        self.target_sr, self.pred_sr = target.detach(), pred.detach()
        if mask is not None:
            self.support_sr = mask
            self.latent_poisson = torch.from_numpy(poisson_densify(self.latent_sr.cpu().numpy(), mask.cpu().numpy()))
        self.timings["sr_s"] = time.time() - start
        return self.latent_sr

    @classmethod
    def from_directory(cls, dataset_dir, *, device="cpu", overrides=None):
        """Read measurement facts and optional solver settings for the dh acquisition.

        CPU is the default device. Use device="cuda:0" for CUDA.
        """
        config = load_config(dataset_dir, overrides=overrides)
        settings, facts = config["solver"], config["measurement"]
        torch.manual_seed(settings["seed"])
        device = torch.device(device)
        rec = load_window_npz(config)
        sensor = tuple(facts["sensor"]["shape_yx"])
        xt, yt, tt, pt = np_to_torch(rec.xs, rec.ys, rec.ts, rec.ps, device)
        I_ref = torch.as_tensor(rec.frame, device=device).float()
        I_lr = torch.as_tensor(rec.frame, device=device).double()
        I_lr = I_lr / I_lr.max()
        flow = SplineFlow(rec.win, settings["pieces"], sensor, device=device)
        flow.set_segments(register_segments(rec.xs, rec.ys, rec.ts, rec.ps, flow))
        optics = AngularSpectrum(facts["optics"]["wavelength_m"], facts["optics"]["pixel_pitch_m"],
                                 sensor, device=device, learn_distance=True)
        scan = settings["autofocus"]
        distances = np.linspace(scan["min_m"], scan["max_m"], scan["samples"])
        z_af, scores = optics.autofocus(I_ref, distances)
        event = EventModel(xt, yt, tt, pt, sensor, fidelity="normalized")
        frame = FrameModel(I_lr, rec.xs, rec.ys, rec.ts, rec.ps, rec.win_frame, event.raw_c, sensor,
                           edi=False, domain="amplitude")
        kappa = dict(settings["kappa"])
        if kappa["event_sr"] is None:
            M = settings["M"]
            kappa["event_sr"] = 1e3 / (M * M * sensor[0] * sensor[1])
        model = cls(flow, event, frame, optics, optimizer=AdamP, kappa=kappa,
                    lr=settings["lr"], schedule=settings["schedule"])
        model.config, model.recording = config, rec
        model.I_ref, model.I_lr = I_ref, I_lr
        model.z_autofocus, model.autofocus_scores = z_af, scores
        return model

    def run(self):
        """Run CMax, frame-only M=1, then SR."""
        settings = self.config["solver"]
        self.cmax(iterations=settings["iterations"]["cmax"])
        u_bp = self.optics(torch.sqrt(self.I_ref / self.I_ref.max()), -self.z_autofocus).to(torch.complex64)
        self.backpropagated = u_bp
        latent_m1 = self.joint(u_bp, iterations=settings["iterations"]["joint"])
        self.z_m1 = self.optics.distance.item()
        self.sr(latent_m1, settings["M"], iterations=settings["iterations"]["sr"], sigma=settings["sigma"])
        return self

    def arrays(self):
        """Named arrays use the original solver save convention for trajectory [K,2]."""
        arrays = {name: getattr(self, name).detach().cpu().numpy() for name in
                  ("object_m1", "object_sr", "latent_m1", "latent_sr", "target_m1", "pred_m1", "target_sr", "pred_sr")}
        arrays.update(trajectory=self.flow.segments().detach().cpu().numpy(),
                      knots=self.flow.knots.detach().cpu().numpy(),
                      background=self.frame.background.detach().cpu().numpy(),
                      z=np.asarray(self.optics.distance.detach().item()),
                      z_m1=np.asarray(self.z_m1), z_autofocus=np.asarray(self.z_autofocus),
                      c=np.asarray(float(self.event.contrast.detach())))
        return arrays

    def save(self, out):
        """Write arrays, source/settings provenance, and both dh summary figures outside data/."""
        out = Path(out).expanduser().resolve()
        data_roots = (Path(__file__).resolve().parents[2] / "data", self.config["dataset_dir"])
        if any(out == root or root in out.parents for root in data_roots):
            raise ValueError("Reconstruction output must be outside data/ and the acquisition directory")
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "reconstruction.npz", **self.arrays())
        source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
        inputs = [self.config["dataset_dir"] / self.config["measurement"]["data"]["file"],
                  self.config["dataset_dir"] / "config.yaml"]
        if (self.config["dataset_dir"] / "config_neurosr.yaml").is_file():
            inputs.append(self.config["dataset_dir"] / "config_neurosr.yaml")
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        dataset_dir = self.config["dataset_dir"]
        metadata = dict(dataset=dataset_dir.name, measurement=self.config["measurement"],
                        solver=self.config["solver"], effective_kappa=self.kappa, effective_lr=self.lr,
                        effective_schedule=self.schedule, t0_us=float(self.t0),
                        files={p.relative_to(dataset_dir).as_posix(): digest(p) for p in inputs},
                        code={p.name: digest(p) for p in source_files},
                        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                         device=str(self.event.device), cpu_threads=torch.get_num_threads()),
                        z_autofocus_m=self.z_autofocus, z_m1_m=self.z_m1,
                        z_sr_m=self.optics.distance.item(), c=float(self.event.contrast.detach()),
                        flow=self.flow.state(), timings=self.timings, history=self.history,
                        autofocus_scores=self.autofocus_scores.tolist())
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        from .visualize import save_summary
        save_summary(self, out)
        return out
