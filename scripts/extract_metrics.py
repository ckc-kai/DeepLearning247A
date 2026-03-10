"""Extract val/test metrics from TensorBoard events."""
import os
import sys

def main():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("pip install tensorboard")
        sys.exit(1)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    steps_per_epoch = 480

    dirs = [
        ("10epoch (intermediate_ctc_l3_w03)", os.path.join(base, "logs/conformer_independent_study/ten_epoch/intermediate_ctc_l3_w03/lightning_logs/version_0")),
        ("20epoch cont (intermediate_ctc_l3_w03_20epoch)", os.path.join(base, "logs/conformer_independent_study/ten_epoch/intermediate_ctc_l3_w03_20epoch/lightning_logs/version_0")),
    ]

    tags = ["val/CER", "val/p_blank", "val/loss", "val/IER", "val/DER", "val/SER"]

    for name, path in dirs:
        if not os.path.isdir(path):
            print(f"Skip {name}: {path} not found")
            continue
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        scalars = ea.Tags().get("scalars", [])
        print(f"\n=== {name} ===")
        for tag in tags:
            if tag not in scalars:
                continue
            vals = ea.Scalars(tag)
            vals = sorted(vals, key=lambda x: x.step)
            print(f"\n{tag}:")
            for x in vals:
                ep = x.step // steps_per_epoch
                print(f"  epoch={ep:2d} step={x.step:5d}  {x.value:.4f}")

if __name__ == "__main__":
    main()
