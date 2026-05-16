"""GCP infrastructure setup CLI."""

import argparse
import logging
import sys
from pathlib import Path

from app.services.gcp_service import GCPService
from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up GCP infrastructure for EKG classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # setup command
    setup_parser = subparsers.add_parser("setup", help="Create GCP resources")
    setup_parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading data files to GCS",
    )

    # upload command
    upload_parser = subparsers.add_parser("upload", help="Upload data to GCS")
    upload_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing data files",
    )

    # load-bigquery command
    subparsers.add_parser("load-bigquery", help="Load data from GCS to BigQuery")

    # validate command
    subparsers.add_parser("validate", help="Validate GCP configuration")

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

    if not args.command:
        logger.error("No command specified. Use --help for usage.")
        return 1

    settings = get_settings()

    if not settings.gcp.project_id or not settings.gcp.bucket_name:
        logger.error(
            "GCP configuration missing. Set GCP_PROJECT_ID and GCP_BUCKET_NAME "
            "environment variables or create a .env file."
        )
        return 1

    service = GCPService(settings)

    try:
        if args.command == "setup":
            service.setup(skip_upload=args.skip_upload)
            logger.info("GCP setup completed successfully")

        elif args.command == "upload":
            data_dir = args.data_dir or settings.data_dir
            service.upload_data(data_dir)
            logger.info("Data upload completed successfully")

        elif args.command == "load-bigquery":
            train_rows, test_rows = service.load_to_bigquery()
            logger.info(f"Loaded {train_rows} training rows, {test_rows} test rows")

        elif args.command == "validate":
            if service.validate_config():
                logger.info("Configuration is valid")
            else:
                logger.error("Configuration validation failed")
                return 1

        return 0

    except Exception as e:
        logger.exception(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
