import logging
from pathlib import Path

import pandas as pd

from config import Settings
from errors import UnknownBackendError
from libs.data.loader import LocalDataLoader
from libs.data.preprocessing import balance_classes, split_features_labels
from libs.ml.sklearn_trainer import SklearnMLPTrainer
from models import TrainingResult

logger = logging.getLogger(__name__)


class TrainingService:
    """Orchestrates data loading, preprocessing, training, and evaluation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loader = LocalDataLoader(settings)

    def train(
        self,
        backend: str = "sklearn",
        output_path: Path | None = None,
    ) -> TrainingResult:
        """Load, optionally balance, train on backend "sklearn" or "spark", evaluate.

        When output_path is given the model is saved there and the result records it.
        Raises ValueError for an unknown backend.
        """
        logger.info("Starting training pipeline with backend: %s", backend)

        # load data
        train_data = self.loader.load_train()
        test_data = self.loader.load_test()

        train_stats = self.loader.get_stats(train_data)
        test_stats = self.loader.get_stats(test_data)

        logger.info(
            "Loaded %s training samples, %s test samples",
            train_stats.num_samples,
            test_stats.num_samples,
        )

        # preprocess
        if self.settings.model.balance_classes:
            train_data = balance_classes(
                train_data,
                class_size=self.settings.model.balanced_class_size,
                random_seed=self.settings.model.random_seed,
            )
            test_balance_size = self.settings.model.balanced_class_size // 4
            test_data = balance_classes(
                test_data,
                class_size=test_balance_size,
                random_seed=self.settings.model.random_seed,
            )

        if backend == "sklearn":
            result = self._train_sklearn(train_data, test_data, output_path)
        elif backend == "spark":
            result = self._train_spark(train_data, test_data, output_path)
        else:
            raise UnknownBackendError(f"Unknown backend: {backend}")

        return result

    def _train_sklearn(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        output_path: Path | None,
    ) -> TrainingResult:
        """Train using scikit-learn."""
        x_train, y_train = split_features_labels(train_data)
        x_test, y_test = split_features_labels(test_data)

        trainer = SklearnMLPTrainer(self.settings.model)
        trainer.train(x_train, y_train)
        result = trainer.evaluate(x_test, y_test)

        if output_path:
            trainer.save(output_path)
            result = result.model_copy(update={"model_path": output_path})

        return result

    def _train_spark(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        output_path: Path | None,
    ) -> TrainingResult:
        """Train using PySpark."""
        # pyspark starts a JVM on import, so it loads only for --backend spark.
        from pyspark.sql import SparkSession  # noqa: PLC0415

        from libs.ml.spark_trainer import SparkMLPTrainer  # noqa: PLC0415

        spark = (
            SparkSession.builder.appName("EKGClassifier")
            .config("spark.driver.memory", "2g")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("ERROR")

        # convert pandas to spark dataframes
        train_spark = spark.createDataFrame(train_data)
        test_spark = spark.createDataFrame(test_data)

        trainer = SparkMLPTrainer(self.settings.model, spark=spark)
        trainer.train(train_spark)
        result = trainer.evaluate(test_spark)

        if output_path:
            trainer.save(output_path)
            result = result.model_copy(update={"model_path": output_path})

        spark.stop()
        return result
