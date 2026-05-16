"""Tests for sklearn trainer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import ModelConfig
from libs.data.preprocessing import split_features_labels
from libs.ml.sklearn_trainer import SklearnMLPTrainer


class TestSklearnMLPTrainer:
    def test_train(self, sample_train_data: pd.DataFrame, model_config: ModelConfig):
        x_train, y_train = split_features_labels(sample_train_data)
        trainer = SklearnMLPTrainer(model_config)

        trainer.train(x_train, y_train)

        assert trainer.model is not None

    def test_predict(self, sample_train_data: pd.DataFrame, model_config: ModelConfig):
        x_train, y_train = split_features_labels(sample_train_data)
        trainer = SklearnMLPTrainer(model_config)
        trainer.train(x_train, y_train)

        predictions = trainer.predict(x_train[:10])

        assert len(predictions) == 10
        assert all(p in [0, 1, 2, 3, 4] for p in predictions)

    def test_predict_without_training(self, model_config: ModelConfig):
        trainer = SklearnMLPTrainer(model_config)

        with pytest.raises(RuntimeError, match="not trained"):
            trainer.predict(np.random.rand(10, 187))

    def test_predict_proba(
        self, sample_train_data: pd.DataFrame, model_config: ModelConfig
    ):
        x_train, y_train = split_features_labels(sample_train_data)
        trainer = SklearnMLPTrainer(model_config)
        trainer.train(x_train, y_train)

        proba = trainer.predict_proba(x_train[:10])

        assert proba.shape == (10, 5)
        # probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_evaluate(
        self,
        sample_train_data: pd.DataFrame,
        sample_test_data: pd.DataFrame,
        model_config: ModelConfig,
    ):
        x_train, y_train = split_features_labels(sample_train_data)
        x_test, y_test = split_features_labels(sample_test_data)

        trainer = SklearnMLPTrainer(model_config)
        trainer.train(x_train, y_train)
        result = trainer.evaluate(x_test, y_test)

        assert 0 <= result.accuracy <= 1
        assert 0 <= result.f1_score <= 1
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert len(result.confusion_matrix) == 5
        assert result.classification_report != ""

    def test_save_and_load(
        self,
        sample_train_data: pd.DataFrame,
        model_config: ModelConfig,
        tmp_path: Path,
    ):
        x_train, y_train = split_features_labels(sample_train_data)
        trainer = SklearnMLPTrainer(model_config)
        trainer.train(x_train, y_train)

        model_path = tmp_path / "model.pkl"
        trainer.save(model_path)

        assert model_path.exists()

        # load and verify predictions match
        original_predictions = trainer.predict(x_train[:5])

        new_trainer = SklearnMLPTrainer(model_config)
        new_trainer.load(model_path)
        loaded_predictions = new_trainer.predict(x_train[:5])

        assert np.array_equal(original_predictions, loaded_predictions)

    def test_save_without_training(self, model_config: ModelConfig, tmp_path: Path):
        trainer = SklearnMLPTrainer(model_config)

        with pytest.raises(RuntimeError, match="not trained"):
            trainer.save(tmp_path / "model.pkl")

    def test_load_nonexistent_file(self, model_config: ModelConfig, tmp_path: Path):
        trainer = SklearnMLPTrainer(model_config)

        with pytest.raises(FileNotFoundError):
            trainer.load(tmp_path / "nonexistent.pkl")

    def test_custom_config(self, sample_train_data: pd.DataFrame):
        config = ModelConfig(
            hidden_layers=(50,),
            max_iter=10,
            random_seed=123,
        )
        x_train, y_train = split_features_labels(sample_train_data)
        trainer = SklearnMLPTrainer(config)
        trainer.train(x_train, y_train)

        assert trainer.model.hidden_layer_sizes == (50,)
        assert trainer.model.max_iter == 10
        assert trainer.model.random_state == 123
