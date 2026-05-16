import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier

from config import ModelConfig
from models import TrainingResult

logger = logging.getLogger(__name__)


class SklearnMLPTrainer:
    """Multi-layer perceptron trainer using scikit-learn."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model: MLPClassifier | None = None

    def train(self, x_train: Any, y_train: Any) -> None:
        """Train the MLP classifier."""
        logger.info(
            f"Training MLP with layers {self.config.hidden_layers}, "
            f"max_iter={self.config.max_iter}"
        )

        self.model = MLPClassifier(
            hidden_layer_sizes=self.config.hidden_layers,
            max_iter=self.config.max_iter,
            random_state=self.config.random_seed,
            activation="relu",
            solver="adam",
        )

        self.model.fit(x_train, y_train)
        logger.info(f"Model trained on {len(x_train)} samples")

    def predict(self, x: Any) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(x)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Generate prediction probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(x)

    def evaluate(self, x_test: Any, y_test: Any) -> TrainingResult:
        """Evaluate model and return metrics."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        logger.info(f"Evaluating model on {len(x_test)} samples")
        y_pred = self.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

        return TrainingResult(
            accuracy=acc,
            f1_score=f1,
            precision=prec,
            recall=rec,
            confusion_matrix=cm.tolist(),
            classification_report=report,
        )

    def save(self, path: Path) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load trained model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
