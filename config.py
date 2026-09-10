"""
Paths and global settings.

This is notebook cell 8. The only change is how the base directory is resolved: the
notebook uses `os.getcwd()`, which requires the working directory to be the project
root. Here the paths are anchored to this file instead, so `main.py` runs correctly from
any directory. This is the same pattern already used in the original `ML Project - FINAL.py`.
"""

import os

# Define the path of the data and images
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")
IMG_DIR = os.path.join(current_dir, "img")
F_CSV_TRAIN = os.path.join(DATA_DIR, "train.csv")
F_CSV_TEST = os.path.join(DATA_DIR, "test.csv")
F_CSV_SUB = os.path.join(DATA_DIR, "submission.csv")

# Fixing the seed for replicability
SEED = 1
