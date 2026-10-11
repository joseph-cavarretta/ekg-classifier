from pathlib import Path

import pytest

from app.cli import gcp_setup, train
from config import GCPConfig, Settings
from models import TrainingResult

RESULT = TrainingResult(
    accuracy=0.9,
    f1_score=0.8,
    precision=0.85,
    recall=0.9,
    confusion_matrix=[[1, 0, 0, 0, 0]] * 5,
    classification_report="report-body",
)


class StubTrainingService:
    """Records each train() call; fails the way `error` says to."""

    def __init__(self, settings: Settings, error: Exception | None = None) -> None:
        self.settings = settings
        self.error = error
        self.calls: list[tuple[str, Path | None]] = []

    def train(self, backend: str, output_path: Path | None) -> TrainingResult:
        self.calls.append((backend, output_path))
        if self.error is not None:
            raise self.error
        return RESULT.model_copy(update={"model_path": output_path})


def _install_training(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    argv: list[str],
    error: Exception | None = None,
) -> list[StubTrainingService]:
    created: list[StubTrainingService] = []

    def make(cli_settings: Settings) -> StubTrainingService:
        created.append(StubTrainingService(cli_settings, error))
        return created[-1]

    monkeypatch.setattr(train, "get_settings", lambda: settings)
    monkeypatch.setattr(train, "TrainingService", make)
    monkeypatch.setattr("sys.argv", ["ekg-train", *argv])
    return created


def test_train_prints_report_and_uses_default_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
) -> None:
    created = _install_training(monkeypatch, settings, ["--no-balance-classes", "-v"])

    assert train.main() == 0

    assert created[0].calls == [("sklearn", settings.output_dir / "model.pkl")]
    assert created[0].settings.model.balance_classes is False
    assert "report-body" in capsys.readouterr().out


def test_train_honours_output_flag(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    created = _install_training(
        monkeypatch, settings, ["--output", str(tmp_path / "m")]
    )
    assert train.main() == 0
    assert created[0].calls[0][1] == tmp_path / "m"


def test_train_reports_missing_data(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    created = _install_training(
        monkeypatch, settings, ["--backend", "spark"], FileNotFoundError("no data")
    )
    assert train.main() == 1
    assert created[0].calls == [("spark", settings.output_dir / "spark_model")]


def test_train_reports_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _install_training(monkeypatch, settings, [], RuntimeError("boom"))
    assert train.main() == 1


class StubGCPService:
    """Records which command ran; validate_config answers `valid`."""

    def __init__(
        self, calls: list[str], *, valid: bool = True, fail: bool = False
    ) -> None:
        self.calls = calls
        self.valid = valid
        self.fail = fail

    def setup(self, skip_upload: bool) -> None:
        self.calls.append(f"setup skip={skip_upload}")

    def upload_data(self, data_dir: Path) -> tuple[str, str]:
        self.calls.append(f"upload {data_dir.name}")
        return "gs://a", "gs://b"

    def load_to_bigquery(self) -> tuple[int, int]:
        self.calls.append("load")
        if self.fail:
            raise RuntimeError("boom")
        return 1, 2

    def validate_config(self) -> bool:
        self.calls.append("validate")
        return self.valid


def _install_gcp(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    argv: list[str],
    *,
    valid: bool = True,
    fail: bool = False,
) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(gcp_setup, "get_settings", lambda: settings)
    monkeypatch.setattr(
        gcp_setup, "GCPService", lambda _: StubGCPService(calls, valid=valid, fail=fail)
    )
    monkeypatch.setattr("sys.argv", ["ekg-gcp-setup", *argv])
    return calls


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["setup"], "setup skip=False"),
        (["setup", "--skip-upload"], "setup skip=True"),
        (["upload"], "upload data"),
        (["upload", "--data-dir", "/srv/other"], "upload other"),
        (["load-bigquery"], "load"),
        (["validate"], "validate"),
    ],
)
def test_gcp_commands(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    argv: list[str],
    expected: str,
) -> None:
    calls = _install_gcp(monkeypatch, settings, ["-v", *argv])
    assert gcp_setup.main() == 0
    assert calls == [expected]


def test_gcp_needs_a_command(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls = _install_gcp(monkeypatch, settings, [])
    assert gcp_setup.main() == 1
    assert calls == []


def test_gcp_needs_project_and_bucket(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    empty = settings.model_copy(
        update={"gcp": GCPConfig(project_id="", bucket_name="")}
    )
    calls = _install_gcp(monkeypatch, empty, ["validate"])
    assert gcp_setup.main() == 1
    assert calls == []


def test_gcp_invalid_config_fails(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _install_gcp(monkeypatch, settings, ["validate"], valid=False)
    assert gcp_setup.main() == 1


def test_gcp_command_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _install_gcp(monkeypatch, settings, ["load-bigquery"], fail=True)
    assert gcp_setup.main() == 1
