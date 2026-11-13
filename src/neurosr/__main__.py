"""NeuroSR command line.

    python -m neurosr joint WINDOW.npz --output DIR    joint event-frame reconstruction from an EDI anchor
    python -m neurosr ACQUISITION_DIR --output DIR     holography from a sharp exposure

Bundled inputs: data/joint/earthworm.npz and data/joint/testis.npz for the first route,
data/holography/usaf_veryclose for the second.  Run "python -m neurosr joint --help" for the
options of the first route.
"""
import argparse
from pathlib import Path
import sys

import torch

from . import NeuroSR
from .data import output_directory


def main():
    if sys.argv[1:2] == ["joint"]:
        from .joint_demo import main as run_joint

        return run_joint(sys.argv[2:])
    parser = argparse.ArgumentParser(
        prog="python -m neurosr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="python -m neurosr joint WINDOW.npz --output DIR [--device DEVICE] [--threads N] [--tiny]\n"
              "       python -m neurosr ACQUISITION_DIR [--output DIR] [--device DEVICE] [--threads N] [--tiny]")
    parser.add_argument("dataset_dir", type=Path, help="acquisition directory with config.yaml, e.g. "
                                                       "data/holography/usaf_veryclose")
    parser.add_argument("--output", type=Path, default=None,
                        help="new directory outside data/; default: results/<group>/<acquisition>")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4, help="CPU worker threads")
    parser.add_argument("--tiny", action="store_true", help="5 iterations per stage: a smoke check, not a result")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not (args.dataset_dir / "config.yaml").is_file():
        parser.error(f"{args.dataset_dir} is not an acquisition directory (no config.yaml)")
    resolved = args.dataset_dir.resolve()
    try:
        out = output_directory(args.output or Path("results") / resolved.parent.name / resolved.name, resolved)
        device = torch.device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if out.exists():
        parser.error(f"{out} exists; choose a new output directory")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu for a slower run.")
    torch.set_num_threads(args.threads)
    overrides = {"iterations": {"cmax": 5, "joint": 5, "sr": 5}} if args.tiny else None
    model = NeuroSR.from_directory(args.dataset_dir, device=args.device, overrides=overrides)
    result = model.run()
    print("Saved", result.save(out))


if __name__ == "__main__":
    main()
