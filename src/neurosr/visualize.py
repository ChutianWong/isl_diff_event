"""Result figures: each panel over the full sensor, with a dashed box, above its zoom."""
import numpy as np

from ._constants import SENSOR


def mark_transport(ax, displacement, sensor, M):
    """Arrow along the net displacement of the scene on the sensor over the window, with its
    length in sensor pixels, in the lower left corner of a full-field panel.  The arrow has a
    fixed size; the label gives the displacement."""
    from matplotlib import patheffects
    step = np.asarray(displacement, dtype=float)
    length = float(np.hypot(*step))
    if length == 0:
        return
    height, width = sensor
    centre = np.asarray([0.14 * width, 0.88 * height]) * M
    half = 0.08 * width * M * step / length
    outline = [patheffects.withStroke(linewidth=3.5, foreground="white")]
    ax.annotate("", xy=centre + half, xytext=centre - half, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=2, mutation_scale=14,
                                path_effects=outline))
    ax.text(0.03 * width * M, centre[1] - 0.08 * height * M, f"transport {length:.0f} px", color="tab:orange",
            fontsize=9, ha="left", va="center", path_effects=outline)


def show_panels(panels, roi_zoom, sensor=SENSOR, signed_groups=None, colorbars=False, transport=None):
    """One column per panel, the full sensor with the zoom box above its zoom.

    panels are (image, M, title, kind) with kind False for an intensity or amplitude (grey),
    True for a signed quantity (red positive, blue negative) and "phase" for a phase in
    radians (viridis).  The grey panels of a row share one range, the 0.5-99.5 percentiles over
    what that row shows, and so do its phase panels at the 1-99 percentiles; a signed panel is
    centred on zero at the 1 and 99 percentiles of its own values in view.  Optional
    signed_groups shares a fixed symmetric range across named panels and both rows; colorbars
    shows those ranges.  transport, the net (x, y) displacement of the scene in sensor pixels, is
    marked on the first full-field panel."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from .ops import plot_ax_roi_norm, add_roi
    H, Wd = sensor
    rows = (np.asarray([[0, Wd], [H, 0]], dtype=float), np.asarray(roi_zoom, dtype=float))
    fig, axes = plt.subplots(2, len(panels), figsize=(3.6 * len(panels), 6.4), squeeze=False,
                             constrained_layout=colorbars)
    shared_norms, shared_images = {}, {}
    if signed_groups is not None:
        if len(signed_groups) != len(panels):
            raise ValueError("signed_groups must have one entry per panel")
        for group in dict.fromkeys(g for g in signed_groups if g is not None):
            values = np.concatenate([np.asarray(p[0]).ravel() for p, g in zip(panels, signed_groups) if g == group])
            limit = max(float(np.percentile(np.abs(values), 99)), float(np.finfo(np.float32).eps))
            shared_norms[group] = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    for r, roi in enumerate(rows):
        views = []
        for img, M, _, _ in panels:
            x0, x1 = sorted(np.clip(np.round(roi[0] * M).astype(int), 0, img.shape[1]))
            y0, y1 = sorted(np.clip(np.round(roi[1] * M).astype(int), 0, img.shape[0]))
            views.append(img[y0:y1, x0:x1])
        grey = [v.ravel() for v, p in zip(views, panels) if p[3] is False]
        lo, hi = np.percentile(np.concatenate(grey), [0.5, 99.5]) if grey else (0.0, 1.0)
        phase = [v.ravel() for v, p in zip(views, panels) if p[3] == "phase"]
        phase_lo, phase_hi = np.percentile(np.concatenate(phase), [1, 99]) if phase else (-1.0, 1.0)
        for c, ((img, M, title, kind), view) in enumerate(zip(panels, views)):
            signed = kind is True
            if signed:
                group = None if signed_groups is None else signed_groups[c]
                if group in shared_norms:
                    norm = shared_norms[group]
                else:
                    pos, neg = view[view > 0], view[view < 0]
                    norm = TwoSlopeNorm(vmin=np.nextafter(np.percentile(neg, 1) if neg.size else -1.0, -np.inf),
                                        vcenter=0.0, vmax=np.nextafter(np.percentile(pos, 99) if pos.size else 1.0, np.inf))
            elif kind == "phase":
                norm = Normalize(phase_lo, phase_hi)
            else:
                norm = Normalize(lo, hi)
            cmap = "seismic" if signed else ("viridis" if kind == "phase" else "gray")
            artist = plot_ax_roi_norm(axes[r, c], img, cmap, "off", roi=roi, M=M, _norm=norm)
            if signed and signed_groups is not None and group in shared_norms:
                shared_images[group] = artist
            if r == 0:
                add_roi(axes[r, c], roi_zoom, M, color="black" if kind is True else "red")
                if c == 0 and transport is not None:
                    mark_transport(axes[r, c], transport, sensor, M)
                axes[r, c].set_title(title)
    if colorbars:
        for group, artist in shared_images.items():
            columns = [i for i, g in enumerate(signed_groups) if g == group]
            fig.colorbar(artist, ax=axes[:, columns].ravel().tolist(), shrink=0.75, label=group)
    else:
        fig.tight_layout()
    return fig


def show_iwe_panels(solver, roi_zoom):
    """Summed event targets and log-intensity predictions, at the native and SR grids.

    Event splats are kept separate from predictions: the contrast fidelity compares c times
    the target, while the normalized fidelity discards the stack scale.  Each quantity shares
    one colour range across M and the zoom rows.
    """
    M = solver.sr_magnification
    panels = [(solver.latent_m1.cpu().numpy(), 1, "Sensor intensity M=1", False),
              (solver.latent_sr.cpu().numpy(), M, f"Sensor intensity SR M={M}", False)]
    for scale, target, pred in ((1, solver.target_m1, solver.pred_m1), (M, solver.target_sr, solver.pred_sr)):
        shape = " x ".join(str(n) for n in target.shape[-2:])
        stage = "native" if target is solver.target_m1 else "SR"
        panels.extend([(target.sum(0).cpu().numpy(), scale, f"{stage} IWE target M={scale}\n{shape}", True),
                       (pred.sum(0).cpu().numpy(), scale, f"{stage} predicted delta L M={scale}\n{shape}", True)])
    fig = show_panels(panels, roi_zoom, solver.event.sensor,
                      signed_groups=[None, None, "signed event splat", "log-intensity change",
                                     "signed event splat", "log-intensity change"], colorbars=True,
                      transport=solver.flow.segments().sum(0).cpu().numpy())
    fig.suptitle(f"Event targets and sensor-latent predictions | c={solver.event.contrast.item():.6g} | "
                 f"fidelity={solver.event.fidelity} | sum over {solver.flow.K} segments")
    return fig


def auto_roi(img_lr, hw, hh, band=(0.25, 0.5), margin=6):
    """Window carrying the most energy in the top octave below the sensor Nyquist, as
    [[x0, x1], [y1, y0]] in native pixels, and its centre.  The zoom is placed by this rule,
    not by hand."""
    X = np.fft.fft2(img_lr - img_lr.mean())
    fr = np.maximum(np.abs(np.fft.fftfreq(img_lr.shape[0]))[:, None], np.abs(np.fft.fftfreq(img_lr.shape[1]))[None, :])
    e = np.abs(np.fft.ifft2(X * ((fr > band[0]) & (fr <= band[1])))) ** 2
    C = np.zeros((e.shape[0] + 1, e.shape[1] + 1))
    C[1:, 1:] = e.cumsum(0).cumsum(1)
    ws, hs = int(round(2 * hw)), int(round(2 * hh))
    best = (-np.inf, margin, margin)
    for y0 in range(margin, e.shape[0] - hs - margin + 1):
        for x0 in range(margin, e.shape[1] - ws - margin + 1):
            v = C[y0 + hs, x0 + ws] - C[y0, x0 + ws] - C[y0 + hs, x0] + C[y0, x0]
            if v > best[0]:
                best = (v, x0, y0)
    cx, cy = best[1] + ws / 2.0, best[2] + hs / 2.0
    return np.asarray([[cx - hw, cx + hw], [cy + hh, cy - hh]]), (cx, cy)


def _relative_phase(field):
    """Phase against the mean field, in radians."""
    return np.angle(field * np.conj(field.mean()))


def save_summary(solver, out):
    """Holography route: the reconstruction panels and the native/SR IWE diagnostic."""
    import matplotlib.pyplot as plt
    M = solver.sr_magnification
    sensor = solver.event.sensor
    roi_zoom, _ = auto_roi(np.abs(solver.target_m1.sum(0).cpu().numpy()), 30, 30 * sensor[0] / sensor[1], margin=1)
    fig = show_panels([(np.sqrt(solver.I_lr.cpu().numpy()), 1, "frame amplitude", False),
                       (solver.backpropagated.abs().cpu().numpy(), 1, "back-propagated", False),
                       (solver.object_m1.abs().cpu().numpy(), 1, "M = 1", False),
                       (solver.object_sr.abs().cpu().numpy(), M, f"M = {M}", False),
                       (_relative_phase(solver.object_sr.cpu().numpy()), M, f"M = {M} phase", "phase"),
                       (solver.target_sr.sum(0).cpu().numpy(), M, f"SR IWE target M={M}", True)], roi_zoom, sensor,
                      transport=solver.flow.segments().sum(0).cpu().numpy())
    fig.savefig(out / "summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    iwe_fig = show_iwe_panels(solver, roi_zoom)
    iwe_fig.savefig(out / "iwe_native_and_sr.png", dpi=160, bbox_inches="tight")
    plt.close(iwe_fig)


def save_joint_figures(out, arrays, scale):
    """Joint route: reconstruction.png (frame, anchor or frame-only field, joint result) and
    iwe.png (measured c IWE, prediction and their difference, summed over the segments)."""
    import matplotlib.pyplot as plt
    sensor = arrays["frame"].shape
    iwe = [(arrays[key] * arrays["iwe_support"]).sum(0) for key in ("iwe_measured", "iwe_predicted", "iwe_residual")]
    native_iwe = np.abs(iwe[0]).reshape(sensor[0], scale, sensor[1], scale).mean((1, 3))
    roi_zoom, _ = auto_roi(native_iwe, 30, 30 * sensor[0] / sensor[1], margin=1)
    if "object_sr" in arrays:
        native, fine = arrays["object_m1"], arrays["object_sr"]
        panels = [(np.sqrt(arrays["frame"]), 1, "frame amplitude", False),
                  (np.abs(native), 1, "frame only, M = 1", False),
                  (np.abs(fine), scale, f"joint, M = {scale}", False),
                  (_relative_phase(native), 1, "frame only, M = 1 phase", "phase"),
                  (_relative_phase(fine), scale, f"joint, M = {scale} phase", "phase")]
    else:
        panels = [(arrays["frame"], 1, "APS frame", False),
                  (arrays["edi"], 1, "EDI", False),
                  (arrays["joint"], scale, f"joint, M = {scale}", False)]
    transport = arrays["segments_px"].sum(0)
    fig = show_panels(panels, roi_zoom, sensor, transport=transport)
    fig.savefig(out / "reconstruction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    titles = ("measured c x IWE", "predicted", "predicted - measured")
    fig = show_panels([(image, scale, title, True) for image, title in zip(iwe, titles)], roi_zoom, sensor,
                      signed_groups=["log-intensity change"] * 3, colorbars=True, transport=transport)
    fig.savefig(out / "iwe.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
