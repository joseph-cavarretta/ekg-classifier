import pandas as pd
import pytest

from app.services.training_service import TrainingService
from config import Settings
from errors import UnknownBackendError
from libs.ml.evaluation import compute_metrics, format_classification_report


def test_compute_metrics_on_perfect_predictions() -> None:
    labels = [0, 1, 2, 3, 4, 0]
    result = compute_metrics(labels, labels)
    assert result.accuracy == 1.0
    assert result.f1_score == 1.0
    assert result.confusion_matrix[0] == [2, 0, 0, 0, 0]


def test_report_lists_metrics_and_labelled_matrix() -> None:
    result = compute_metrics([0, 1, 2, 3, 4, 4], [0, 1, 2, 3, 4, 0])

    report = format_classification_report(result).splitlines()

    assert "Accuracy:  0.8333" in report
    assert report[report.index("Confusion Matrix:") + 1].split() == [
        "N",
        "S",
        "V",
        "F",
        "Q",
    ]
    assert report[report.index("Confusion Matrix:") + 6].split() == [
        "Q",
        "1",
        "0",
        "0",
        "0",
        "1",
    ]
    assert "Detailed Report:" in report


def test_unknown_backend_is_rejected(
    settings: Settings,
    sample_train_data: pd.DataFrame,
    sample_test_data: pd.DataFrame,
) -> None:
    train_path = settings.data_dir / settings.train_file
    test_path = settings.data_dir / settings.test_file
    for frame, path in ((sample_train_data, train_path), (sample_test_data, test_path)):
        frame.to_csv(path, header=False, index=False)

    with pytest.raises(UnknownBackendError, match="tensorflow"):
        TrainingService(settings).train(backend="tensorflow")
