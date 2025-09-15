
# learning_curve_pct.py
import sys
import random, time, os
from typing import List
import matplotlib.pyplot as plt

import dtree
from data_reader import construct_dtree, test_dtree  # use your existing functions

KEEPS = list(range(5, 101, 5))  # percent of training data to KEEP


def drop_to_keep_percent(tree: dtree.DTree, keep_pct: int) -> None:
    """Randomly drop elements in-place so that dataset size ~= keep_pct% of original."""
    n = tree.dataset_size
    target = int(round(n * (keep_pct / 100.0)))
    drop_amount = n - target
    # drop by random index pop for efficiency without extra memory
    for _ in range(drop_amount):

        # time-based seed
        random.seed(time.time_ns())
        idx = random.randrange(len(tree.dataset))
        tree.dataset.pop(idx)
    tree.dataset_size = len(tree.dataset)

def main():
    if len(sys.argv) != 3:
        print("Usage: python learning_curve_pct.py <train_file> <test_file>")
        sys.exit(1)

    train_file, test_file = sys.argv[1], sys.argv[2]

    for i in range(50):
        keeps = []
        accs = []
        for keep in KEEPS:
            # Load full training set each time (as you specified)
            tree = construct_dtree(train_file)
            drop_to_keep_percent(tree, keep)
            tree.build_tree()

            total, acc = test_dtree(tree, test_file)  # acc in [0,1]
            pct = 100.0 * acc

            keeps.append(keep)
            accs.append(pct)
        # Plot: keep% vs test accuracy
        plt.plot(keeps, accs, marker="o")
        plt.title("Learning Curve vs. Kept Training Data (%)")
        plt.xlabel("Kept Training Data (%)")
        plt.ylabel("Test Accuracy (%)")

        ymin = min(accs) - 5
        ymax = max(accs) + 5
        ymin = max(ymin, 0)
        ymax = min(ymax, 100)
        plt.ylim(ymin, ymax)

        plt.grid(True)
        plt.tight_layout()

    plt.savefig(f"learning_curve.png")
    print("Plot saved to learning_curve.png")


if __name__ == "__main__":
    main()

