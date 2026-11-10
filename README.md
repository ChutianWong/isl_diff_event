# NeuroSR

Event and frame reconstruction for computational imaging. The joint solver
combines segmented images of warped events with aggregated event-pair
residuals. Both terms use the same native-pixel aperture.

## Installation

With Python 3.10 or newer, install the package from this directory:

```bash
python -m pip install -e .
```

## Joint reconstruction

Prepare a 100 ms window from an AEDAT4 recording, then run the joint solver.
The offline converter uses `dv` and also uses OpenCV for colour recordings.

```bash
python tools/make_joint_window.py earthworm microscopy.aedat4 results/microscopy-window.npz
python -m neurosr joint results/microscopy-window.npz --output results/microscopy --device cuda:0

python tools/make_joint_window.py testis hologram.aedat4 results/hologram-window.npz
python -m neurosr joint results/hologram-window.npz --output results/hologram --device cuda:0
```

The microscopy preset uses frame 7 and an M=3 output grid. The holography
preset uses frame 69 and M=2. Frame indices are zero-based. The solver fits
an effective contrast from two consecutive APS exposures and holds it and
the CMax trajectory fixed during joint reconstruction. Intensity
super-resolution uses isotropic TV of `log(I + 0.02)` with weight 0.02.
Holography starts from a frame-only complex field and propagation distance,
then jointly refines both with complex-field TV and a hologram Laplacian prior.

Use a fresh output directory. Each run saves the reconstruction, measured and
predicted segmented IWE, their support and residual, and the fit settings.
Use `--device cpu` for a CPU run. M is the ratio of output sampling to native
sensor sampling.

## Bundled holography example

A 100 ms USAF resolution-target window is included. Run the reference
holography solver on the CPU with five updates per stage:

```bash
python -m neurosr data/holography/usaf_veryclose --tiny --output results/usaf
```

See [data layout](data/README.md) for the input format. The bundled example
uses the reference normalized IWE solver.

## Data

The experimental recordings are available on Figshare (https://doi.org/10.6084/m9.figshare.33545614).

## Source Data

[`source_data/Source Data.xlsx`](source_data/) contains the numerical values underlying the graphs,
with one worksheet per figure. The Inventory worksheet lists every panel and its status, and
the Rerun notes worksheet describes the six series that were regenerated from the recordings.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

See [licence information](NOTICE.md) and
[third-party notices](THIRD_PARTY_NOTICES.md).
