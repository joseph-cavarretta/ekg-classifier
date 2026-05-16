"""Training CLI for EKG classifier."""

import argparse
import logging
import sys
from pathlib import Path

from app.services.training_service import TrainingService
from config import get_settings
from libs.ml.evaluation import format_classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EKG classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--backend",
        choices=["sklearn", "spark"],
        default="sklearn",
        help="ML backend to use for training",
    )

    parser.add_argument(
        "--balance-classes",
        action="store_true",
        default=True,
        help="Balance classes by resampling",
    )

    parser.add_argument(
        "--no-balance-classes",
        action="store_false",
        dest="balance_classes",
        help="Do not balance classes",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for saved model (default: output/model.pkl or output/spark_model)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting training with backend: {args.backend}")

    settings = get_settings()
    settings.model.balance_classes = args.balance_classes

    if args.output:
        output_path = args.output
    elif args.backend == "sklearn":
        output_path = settings.output_dir / "model.pkl"
    else:
        output_path = settings.output_dir / "spark_model"

    service = TrainingService(settings)

    try:
        result = service.train(
            backend=args.backend,
            output_path=output_path,
        )

        print(format_classification_report(result))

        if result.model_path:
            logger.info(f"Model saved to: {result.model_path}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
