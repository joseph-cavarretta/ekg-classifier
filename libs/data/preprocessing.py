import logging

import pandas as pd
from sklearn.utils import resample

logger = logging.getLogger(__name__)


def split_features_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features (X) and labels (y).

    Assumes the last column contains labels.
    """
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y


def balance_classes(
    data: pd.DataFrame,
    class_size: int,
    label_col: int = 187,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Balance classes by upsampling minority classes and downsampling majority.

    Args:
        data: DataFrame with features and label column
        class_size: Target number of samples per class
        label_col: Index of the label column (default 187 for EKG data)
        random_seed: Base seed for reproducibility

    Returns:
        Balanced DataFrame with equal samples per class
    """
    logger.info(f"Balancing classes to {class_size} samples each")

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

        logger.debug(f"Class {label}: {current_size} -> {class_size}")

    result = pd.concat(balanced_dfs, ignore_index=True)
    logger.info(f"Balanced dataset: {len(result)} total samples")
    return result
