"""The runner's fold construction, available without running any models."""

import numpy as np
from sklearn.model_selection import KFold


def split_indices(
    n_rows: int, *, n_folds: int, seed: int, sample_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split the full dataset and cap each fold using the benchmark protocol."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_cap = int(sample_size * (n_folds - 1) / n_folds) if sample_size else 0
    test_cap = sample_size // n_folds if sample_size else 0
    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_rows))):
        if train_cap and len(train_idx) > train_cap:
            rng = np.random.default_rng(seed * 10007 + fold_idx)
            train_idx = rng.choice(train_idx, size=train_cap, replace=False)
        if test_cap and len(test_idx) > test_cap:
            rng = np.random.default_rng(seed * 10007 + fold_idx + 1)
            test_idx = rng.choice(test_idx, size=test_cap, replace=False)
        splits.append((train_idx, test_idx))
    return splits
