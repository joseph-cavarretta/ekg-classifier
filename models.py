from enum import IntEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HeartbeatClass(IntEnum):
    """Classification labels for heartbeat types."""

    NORMAL = 0  # N: Normal beat
    SVEB = 1  # S: Supraventricular premature beat
    VEB = 2  # V: Premature ventricular contraction
    FUSION = 3  # F: Fusion of ventricular and normal beat
    UNKNOWN = 4  # Q: Unclassifiable beat

    @property
    def description(self) -> str:
        descriptions = {
            self.NORMAL: "Normal beat",
            self.SVEB: "Supraventricular premature beat",
            self.VEB: "Premature ventricular contraction",
            self.FUSION: "Fusion of ventricular and normal beat",
            self.UNKNOWN: "Unclassifiable beat",
        }
        return descriptions[self]

    @property
    def short_name(self) -> str:
        names = {
            self.NORMAL: "N",
            self.SVEB: "S",
            self.VEB: "V",
            self.FUSION: "F",
            self.UNKNOWN: "Q",
        }
        return names[self]


class DatasetStats(BaseModel):
    """Statistics for a dataset."""

    model_config = ConfigDict(frozen=True)

    num_samples: int = Field(ge=0, description="Total number of samples")
    num_features: int = Field(ge=0, description="Number of features per sample")
    class_distribution: dict[int, int] = Field(
        description="Count of samples per class"
    )

    @property
    def total_classes(self) -> int:
        return len(self.class_distribution)

    @property
    def is_balanced(self) -> bool:
        if not self.class_distribution:
            return True
        counts = list(self.class_distribution.values())
        min_count, max_count = min(counts), max(counts)
        return max_count <= min_count * 1.5


class TrainingResult(BaseModel):
    """Result from model training."""

    model_config = ConfigDict(frozen=True)

    accuracy: Annotated[float, Field(ge=0.0, le=1.0)]
    f1_score: Annotated[float, Field(ge=0.0, le=1.0)]
    precision: Annotated[float, Field(ge=0.0, le=1.0)]
    recall: Annotated[float, Field(ge=0.0, le=1.0)]
    confusion_matrix: list[list[int]]
    classification_report: str
    model_path: Path | None = None

    @property
    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.4f}, "
            f"F1: {self.f1_score:.4f}, "
            f"Precision: {self.precision:.4f}, "
            f"Recall: {self.recall:.4f}"
        )


class GCSUri(BaseModel):
    """Validated Google Cloud Storage URI."""

    model_config = ConfigDict(frozen=True)

    bucket: str = Field(min_length=3, max_length=63)
    path: str = Field(default="")

    @field_validator("bucket")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        if not v[0].isalnum():
            raise ValueError("Bucket name must start with a letter or number")
        if not all(c.isalnum() or c in "-_." for c in v):
            raise ValueError(
                "Bucket name can only contain letters, numbers, hyphens, underscores, and periods"
            )
        return v

    @property
    def uri(self) -> str:
        if self.path:
            return f"gs://{self.bucket}/{self.path}"
        return f"gs://{self.bucket}"

    def with_path(self, path: str) -> "GCSUri":
        return GCSUri(bucket=self.bucket, path=path)


class BigQueryTable(BaseModel):
    """Validated BigQuery table reference."""

    model_config = ConfigDict(frozen=True)

    project_id: str = Field(min_length=6, max_length=30)
    dataset_id: str = Field(min_length=1, max_length=1024)
    table_id: str = Field(min_length=1, max_length=1024)

    @property
    def full_table_id(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"

    @property
    def dataset_ref(self) -> str:
        return f"{self.project_id}.{self.dataset_id}"
