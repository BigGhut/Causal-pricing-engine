"""Pre-registered choices for the portfolio run. See PLAN.md, 2026-10-02.

These values were fixed before that run. Do not retune them from its metrics.
"""

N = 30000
GENERATION_SEEDS = (42, 7, 11, 99, 123)
PRIMARY_GENERATION_SEED = 42
SPLIT_SEED = 20261002
MODEL_RANDOM_STATE = 42
CALIBRATION_BOOT_SEED = 1
TEST_BOOT_SEED = 2
RANDOM_SCORE_SEED = 3
N_BOOT = 1000
N_RANDOM_DRAWS = 1000
SCORE_THRESHOLD = 0.05
FALSE_OVERRIDE_RATE_MAX = 0.10
