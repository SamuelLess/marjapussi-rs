"""
train.py is deprecated and has been removed to reduce technical debt.
Please use one of the modern unified pipelines:
- `python ml/train_online.py` for PPO Self-Play Reinforcement Learning.
- `python ml/train_from_dataset.py` for Supervised Learning.
"""

import sys

def main():
    print("❌ train.py is deprecated!")
    print("👉 Use `python ml/train_online.py` for PPO Reinforcement Learning.")
    print("👉 Use `python ml/train_from_dataset.py` for Supervised Learning.")
    sys.exit(1)

if __name__ == "__main__":
    main()
