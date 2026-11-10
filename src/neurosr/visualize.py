"""The original dh full-field and automatic-ROI comparison panels."""
import numpy as np
from ._constants import SENSOR

def show_panels(panels, roi_zoom, sensor=SENSOR, signed_groups=None, colorbars=False):
    """One column per panel, the full sensor with the zoom box above its zoom.

    panels are (image, M, title, signed).  The unsigned panels of a row share one grey range,
    the 0.5-99.5 percentiles over what that row shows; a signed panel is centred on zero at the
    1 and 99 percentiles of its own values in view. Optional signed_groups shares a fixed
    symmetric range across named panels and both rows; colorbars shows those ranges."""
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
        grey = [v.ravel() for v, p in zip(views, panels) if not p[3]]
        lo, hi = np.percentile(np.concatenate(grey), [0.5, 99.5]) if grey else (0.0, 1.0)
        for c, ((img, M, title, signed), view) in enumerate(zip(panels, views)):
            if signed:
                group = None if signed_groups is None else signed_groups[c]
                if group in shared_norms:
                    norm = shared_norms[group]
                else:
                    pos, neg = view[view > 0], view[view < 0]
                    norm = TwoSlopeNorm(vmin=np.nextafter(np.percentile(neg, 1) if neg.size else -1.0, -np.inf),
                                        vcenter=0.0, vmax=np.nextafter(np.percentile(pos, 99) if pos.size else 1.0, np.inf))
            else:
                norm = Normalize(lo, hi)
            artist = plot_ax_roi_norm(axes[r, c], img, "seismic" if signed else "gray", "off", roi=roi, M=M, _norm=norm)
            if signed and signed_groups is not None and group in shared_norms:
                shared_images[group] = artist
            if r == 0:
                add_roi(axes[r, c], roi_zoom, M, color="k" if signed else "w")
                axes[r, c].set_title(title)
    if colorbars:
        for group, artist in shared_images.items():
            columns = [i for i, g in enumerate(signed_groups) if g == group]
            fig.colorbar(artist, ax=axes[:, columns].ravel().tolist(), shrink=0.75, label=group)
    else:
        fig.tight_layout()
    return fig


def show_iwe_panels(solver, roi_zoom):
    """Exact summed event targets and log-intensity predictions, at native and SR grids.

    Raw event splats are kept separate from predictions: the contrast loss compares c times
    the target, while normalized fidelity discards the stack scale. Red is positive,
    blue negative; each quantity shares one colour range across M and the zoom rows.
    """
    if solver.event.fidelity == "pairs":
        raise ValueError("the pairs fidelity has no event image; show the latents with show_panels")
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
                                     "signed event splat", "log-intensity change"], colorbars=True)
    fig.suptitle(f"Raw event targets and sensor-latent predictions | c={solver.event.contrast.item():.6g} | "
                 f"fidelity={solver.event.fidelity} | sum over {solver.flow.K} segments")
    return fig


def auto_roi(img_lr, hw, hh, band=(0.25, 0.5), margin=6):
    """Window carrying the most energy in the top octave below the sensor Nyquist, in the
    plot_ax_roi convention [[x0, x1], [y1, y0]] (LR px), and its centre."""
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



def save_summary(solver, out):
    """Save the six dh panels and the native/SR IWE diagnostic, without displaying a window."""
    import matplotlib.pyplot as plt
    M = solver.sr_magnification
    roi_zoom, _ = auto_roi(np.abs(solver.target_m1.sum(0).cpu().numpy()), 30, 30 * 260 / 346, margin=1)
    phase = (solver.object_sr * solver.object_sr.mean().conj()).angle()
    fig = show_panels([(np.sqrt(solver.I_lr.cpu().numpy()), 1, "frame amplitude", False),
                       (solver.backpropagated.abs().cpu().numpy(), 1, "back-propagated", False),
                       (solver.object_m1.abs().cpu().numpy(), 1, "M = 1", False),
                       (solver.object_sr.abs().cpu().numpy(), M, f"M = {M}", False),
                       (phase.cpu().numpy(), M, f"M = {M} phase", True),
                       (solver.target_sr.sum(0).cpu().numpy(), M, f"SR IWE target M={M}", True)], roi_zoom)
    fig.savefig(out / "summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    iwe_fig = show_iwe_panels(solver, roi_zoom)
    iwe_fig.savefig(out / "iwe_native_and_sr.png", dpi=160, bbox_inches="tight")
    plt.close(iwe_fig)
