import logging
from pathlib import Path
from typing import Any

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import ModelConfig
from models import TrainingResult

logger = logging.getLogger(__name__)

NUM_FEATURES = 187
NUM_CLASSES = 5


class SparkMLPTrainer:
    """Multi-layer perceptron trainer using PySpark ML."""

    def __init__(
        self,
        config: ModelConfig,
        spark: SparkSession | None = None,
        label_col: str = "_c187",
    ) -> None:
        self.config = config
        self.spark = spark or self._create_spark_session()
        self.label_col = label_col
        self.model: PipelineModel | None = None
        self._pipeline: Pipeline | None = None

    def _create_spark_session(self) -> SparkSession:
        """Create a local Spark session."""
        return (
            SparkSession.builder.appName("EKGClassifier")
            .config("spark.driver.memory", "2g")
            .getOrCreate()
        )

    def _build_pipeline(self, feature_cols: list[str]) -> Pipeline:
        """Build the ML pipeline with indexer, assembler, and classifier."""
        indexer = StringIndexer(inputCol=self.label_col, outputCol="label")

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        layers = [NUM_FEATURES, *self.config.hidden_layers, NUM_CLASSES]
        mlp = MultilayerPerceptronClassifier(
            maxIter=self.config.max_iter,
            layers=layers,
            blockSize=self.config.block_size,
            seed=self.config.random_seed,
        )

        return Pipeline(stages=[indexer, assembler, mlp])

    def train(self, x_train: Any, y_train: Any = None) -> None:  # noqa: ARG002
        """Train the MLP classifier.

        Args:
            x_train: Spark DataFrame containing features and label column.
                     y_train is ignored for Spark (label is in the DataFrame).
        """
        if not isinstance(x_train, DataFrame):
            raise TypeError("x_train must be a Spark DataFrame")

        feature_cols = [c for c in x_train.columns if c != self.label_col]
        logger.info(
            f"Training Spark MLP with layers "
            f"{[NUM_FEATURES, *self.config.hidden_layers, NUM_CLASSES]}"
        )

        self._pipeline = self._build_pipeline(feature_cols)
        self.model = self._pipeline.fit(x_train)

        row_count = x_train.count()
        logger.info(f"Model trained on {row_count} samples")

    def predict(self, x: Any) -> DataFrame:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.transform(x)

    def evaluate(self, x_test: Any, y_test: Any = None) -> TrainingResult:  # noqa: ARG002
        """Evaluate model and return metrics."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if not isinstance(x_test, DataFrame):
            raise TypeError("x_test must be a Spark DataFrame")

        predictions = self.predict(x_test)

        # spark evaluator for f1
        evaluator = MulticlassClassificationEvaluator(metricName="f1")
        spark_f1 = evaluator.evaluate(predictions.select("prediction", "label"))
        logger.info(f"Spark F1 score: {spark_f1:.4f}")

        # collect for sklearn metrics
        y_true = [row.label for row in predictions.select("label").collect()]
        y_pred = [row.prediction for row in predictions.select("prediction").collect()]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        prec = precision_score(y_true, y_pred, average="weighted")
        rec = recall_score(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)

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

        self.model.write().overwrite().save(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load trained model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        self.model = PipelineModel.load(str(path))
        logger.info(f"Model loaded from {path}")
