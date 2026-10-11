import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from config import ModelConfig
from errors import MissingFileError, ModelNotTrainedError
from libs.ml.evaluation import compute_metrics
from models import TrainingResult

type Features = pd.DataFrame | np.ndarray
type Labels = pd.Series | np.ndarray

logger = logging.getLogger(__name__)


class SklearnMLPTrainer:
    """Multi-layer perceptron trainer using scikit-learn."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model: MLPClassifier | None = None

    def train(self, x_train: Features, y_train: Labels) -> None:
        """Train the MLP classifier."""
        logger.info(
            "Training MLP with layers %s, max_iter=%s",
            self.config.hidden_layers,
            self.config.max_iter,
        )

        self.model = MLPClassifier(
            hidden_layer_sizes=self.config.hidden_layers,
            max_iter=self.config.max_iter,
            random_state=self.config.random_seed,
            activation="relu",
            solver="adam",
        )

        self.model.fit(x_train, y_train)
        logger.info("Model trained on %s samples", len(x_train))

    def predict(self, x: Features) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ModelNotTrainedError("Model not trained. Call train() first.")
        predictions: np.ndarray = self.model.predict(x)
        return predictions

    def predict_proba(self, x: Features) -> np.ndarray:
        """Generate prediction probabilities."""
        if self.model is None:
            raise ModelNotTrainedError("Model not trained. Call train() first.")
        probabilities: np.ndarray = self.model.predict_proba(x)
        return probabilities

    def evaluate(self, x_test: Features, y_test: Labels) -> TrainingResult:
        """Evaluate model and return metrics."""
        if self.model is None:
            raise ModelNotTrainedError("Model not trained. Call train() first.")

        logger.info("Evaluating model on %s samples", len(x_test))
        result = compute_metrics(y_test, self.predict(x_test))
        logger.info("Accuracy: %.4f, F1: %.4f", result.accuracy, result.f1_score)
        return result

    def save(self, path: Path) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ModelNotTrainedError("Model not trained. Call train() first.")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load trained model from disk."""
        if not path.exists():
            raise MissingFileError(f"Model file not found: {path}")

        self.model = joblib.load(path)
        logger.info("Model loaded from %s", path)
