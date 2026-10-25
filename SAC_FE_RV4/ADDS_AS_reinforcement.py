"""Training entry point (kept at RV3's name so Start_training.py and old
habits keep working). The trainer itself is learn/trainer.py.

    python3 ADDS_AS_reinforcement.py [--preset FILE] [--set KEY=VALUE ...]
"""
import sys

if __name__ == "__main__":
    from learn.trainer import main
    sys.exit(main())
