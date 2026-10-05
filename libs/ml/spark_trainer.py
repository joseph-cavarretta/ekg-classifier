import logging
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession

from config import ModelConfig
from errors import MissingFileError, ModelNotTrainedError
from libs.ml.evaluation import compute_metrics
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

    def train(self, x_train: DataFrame) -> None:
        """Train on a DataFrame holding the feature columns and the label column."""
        feature_cols = [c for c in x_train.columns if c != self.label_col]
        logger.info(
            "Training Spark MLP with layers %s",
            [NUM_FEATURES, *self.config.hidden_layers, NUM_CLASSES],
        )

        self._pipeline = self._build_pipeline(feature_cols)
        self.model = self._pipeline.fit(x_train)

        row_count = x_train.count()
        logger.info("Model trained on %s samples", row_count)

    def predict(self, x: DataFrame) -> DataFrame:
        """Generate predictions."""
        if self.model is None:
            raise ModelNotTrainedError("Model not trained. Call train() first.")
        return self.model.transform(x)

    def evaluate(self, x_test: DataFrame) -> TrainingResult:
        """Evaluate on a DataFrame holding features and labels; return the metrics."""
        if self.model is None:
            raise ModelNotTrainedError("Model not trained. Call train() first.")

        predictions = self.predict(x_test)

        # spark evaluator for f1
        evaluator = MulticlassClassificationEvaluator(metricName="f1")
        spark_f1 = evaluator.evaluate(predictions.select("prediction", "label"))
        logger.info("Spark F1 score: %.4f", spark_f1)

        # collect for sklearn metrics
        y_true = [row.label for row in predictions.select("label").collect()]
        y_pred = [row.prediction for row in predictions.select("prediction").collect()]

        result = compute_metrics(y_true, y_pred)
        logger.info("Accuracy: %.4f, F1: %.4f", result.accuracy, result.f1_score)
        return result

    def save(self, path: Path) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ModelNotTrainedError("Model not trained. Call train() first.")

        self.model.write().overwrite().save(str(path))
        logger.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load trained model from disk."""
        if not path.exists():
            raise MissingFileError(f"Model directory not found: {path}")

        self.model = PipelineModel.load(str(path))
        logger.info("Model loaded from %s", path)
