"""Run joint NeuroSR or a configured reference acquisition."""
import argparse
from pathlib import Path
import sys
import torch
from . import NeuroSR

def main():
    if sys.argv[1:2] == ["joint"]:
        from .joint_demo import main as run_joint

        return run_joint(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Joint reconstruction: python -m neurosr joint --help")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="CPU worker threads")
    parser.add_argument("--tiny", action="store_true", help="5 iterations per stage, for a CPU smoke check")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(args.threads)
    overrides = {"iterations": {"cmax": 5, "joint": 5, "sr": 5}} if args.tiny else None
    model = NeuroSR.from_directory(args.dataset_dir, device=args.device, overrides=overrides)
    result = model.run()
    out = args.output or Path("results") / args.dataset_dir.parent.name / args.dataset_dir.name
    print("Saved", result.save(out))

if __name__ == "__main__":
    main()
