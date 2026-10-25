"""Train the injection detection classifier (Layer 2)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/defense.yaml")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false", default=True,
                        help="Disable Weights & Biases logging (also respects WANDB_DISABLED).")
    parser.add_argument("--resume", metavar="CHECKPOINT_DIR", default=None,
                        help="Resume an interrupted run from a checkpoint-N directory. "
                             "CAUTION (transformers 5.8): the resume path skips mBERT's legacy "
                             "LayerNorm gamma/beta keys, so those weights are NOT restored; "
                             "prefer a fresh run.")
    args = parser.parse_args()
    train(args.config, use_wandb=args.use_wandb, resume_from_checkpoint=args.resume)
