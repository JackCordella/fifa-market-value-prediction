"""
Paths and global settings.

Paths are anchored to this file rather than the working directory, so the pipeline runs
correctly no matter where it is invoked from.
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
