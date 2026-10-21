"""
config.py - All configurable settings for Person 3's modules.

Change values HERE instead of editing the other files.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# PROJECT PATHS
# ----------------------------------------------------------------------------

# Project root: zero-trust-anomaly-detection/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directory: zero-trust-anomaly-detection/data/
DATA_DIR = PROJECT_ROOT / "data"

# Dataset files
TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

# Output directory: zero-trust-anomaly-detection/results/
OUTPUT_DIR = PROJECT_ROOT / "results"

# Official UNSW-NB15 dataset information
DATASET_PAGE_URL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"


# ----------------------------------------------------------------------------
# DATASET COLUMN SETTINGS
# ----------------------------------------------------------------------------

# Target column: 0 = normal traffic, 1 = attack
LABEL_COLUMN = "label"

# Attack category column
ATTACK_CAT_COLUMN = "attack_cat"

# ID column
ID_COLUMN = "id"

# Columns that must NEVER be used as model/rule input
# because they can cause label leakage.
LEAKAGE_COLUMNS = [
    LABEL_COLUMN,
    ATTACK_CAT_COLUMN,
    ID_COLUMN,
]

# Categorical columns in the UNSW-NB15 dataset
CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "state",
]


# ----------------------------------------------------------------------------
# RULE-BASED BASELINE SETTINGS
# ----------------------------------------------------------------------------

# Feature -> "high" or "low"
# These rules identify unusual network traffic patterns.
RULE_FEATURES = {
    "ct_srv_src": "high",
    "ct_dst_ltm": "high",
    "ct_src_dport_ltm": "high",
    "ct_dst_sport_ltm": "high",
    "ct_dst_src_ltm": "high",
    "ct_src_ltm": "high",
    "ct_srv_dst": "high",
    "rate": "high",
    "sload": "high",
    "sbytes": "high",
    "dbytes": "low",
    "dpkts": "low",
}

# Thresholds are learned from normal training rows only.
RULE_PERCENTILE = 90.0

# Predict an attack when at least this many rules fire.
MIN_RULES_FOR_ATTACK = 2

# Risk score reaches 1.0 at this number of fired rules.
RULES_FOR_FULL_RISK = 5


# ----------------------------------------------------------------------------
# ZERO-TRUST POLICY SETTINGS
# ----------------------------------------------------------------------------

# Risk score below this value -> ALLOW
ALLOW_BELOW = 0.30

# Risk score at or above this value -> DENY
# Scores between ALLOW_BELOW and DENY_AT_OR_ABOVE -> FLAG
DENY_AT_OR_ABOVE = 0.60

# Whether FLAG decisions count as attacks during evaluation.
FLAG_COUNTS_AS_ATTACK = True


# ----------------------------------------------------------------------------
# THRESHOLD-SWEEP SETTINGS
# ----------------------------------------------------------------------------

# Candidate ALLOW thresholds
SWEEP_ALLOW_VALUES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

# Candidate DENY thresholds
SWEEP_DENY_VALUES = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]


# ----------------------------------------------------------------------------
# SYNTHETIC DATA SETTINGS
# ----------------------------------------------------------------------------

RANDOM_SEED = 42

SYNTHETIC_ROWS_TRAIN = 4000
SYNTHETIC_ROWS_TEST = 2000