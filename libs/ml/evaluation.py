import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from models import HeartbeatClass, TrainingResult

logger = logging.getLogger(__name__)


def compute_metrics(y_true: Any, y_pred: Any) -> TrainingResult:
    """Compute classification metrics from predictions."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    return TrainingResult(
        accuracy=acc,
        f1_score=f1,
        precision=prec,
        recall=rec,
        confusion_matrix=cm.tolist(),
        classification_report=report,
    )


def format_classification_report(result: TrainingResult) -> str:
    """Format a TrainingResult as a readable report."""
    lines = [
        "=" * 60,
        "Classification Results",
        "=" * 60,
        "",
        f"Accuracy:  {result.accuracy:.4f}",
        f"F1 Score:  {result.f1_score:.4f}",
        f"Precision: {result.precision:.4f}",
        f"Recall:    {result.recall:.4f}",
        "",
        "Confusion Matrix:",
    ]

    cm = np.array(result.confusion_matrix)
    class_labels = [hc.short_name for hc in HeartbeatClass]

    # header
    header = "     " + "  ".join(f"{label:>5}" for label in class_labels)
    lines.append(header)

    # rows
    for i, row in enumerate(cm):
        row_str = f"{class_labels[i]:>3}  " + "  ".join(f"{val:>5}" for val in row)
        lines.append(row_str)

    lines.extend(["", "Detailed Report:", result.classification_report])

    return "\n".join(lines)
