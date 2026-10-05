import logging

import pandas as pd
from sklearn.utils import resample

logger = logging.getLogger(__name__)


def split_features_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into features and labels; the last column holds the labels."""
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y


def balance_classes(
    data: pd.DataFrame,
    class_size: int,
    label_col: int = 187,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Resample every class to exactly class_size rows.

    Larger classes are downsampled without replacement and smaller ones upsampled with
    replacement. Class i (in sorted label order) uses seed random_seed + i, so the
    result is reproducible. label_col is the label's column index (187 for EKG data).
    """
    logger.info("Balancing classes to %s samples each", class_size)

    class_labels = sorted(data[label_col].unique())
    balanced_dfs = []

    for i, label in enumerate(class_labels):
        class_data = data[data[label_col] == label]
        current_size = len(class_data)

        if current_size == class_size:
            balanced_dfs.append(class_data)
        elif current_size > class_size:
            # downsample majority class
            downsampled = class_data.sample(
                n=class_size,
                random_state=random_seed + i,
            )
            balanced_dfs.append(downsampled)
        else:
            # upsample minority class
            upsampled = resample(
                class_data,
                replace=True,
                n_samples=class_size,
                random_state=random_seed + i,
            )
            balanced_dfs.append(upsampled)

        logger.debug("Class %s: %s -> %s", label, current_size, class_size)

    result: pd.DataFrame = pd.concat(balanced_dfs, ignore_index=True)
    logger.info("Balanced dataset: %s total samples", len(result))
    return result
