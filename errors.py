# Every error is permanent: the same input fails the same way, so none is retried.
# Each also subclasses the builtin it replaced, so existing handlers still match.


class EkgClassifierError(Exception):
    """Base for every error raised by this package."""


class ModelNotTrainedError(EkgClassifierError, RuntimeError):
    """A trainer was asked to predict, evaluate or save before train() or load()."""


class MissingFileError(EkgClassifierError, FileNotFoundError):
    """A data file or saved model the caller named does not exist."""


class InvalidDatasetError(EkgClassifierError, ValueError):
    """A dataset cannot be used as loaded, e.g. it contains nulls."""


class UnknownBackendError(EkgClassifierError, ValueError):
    """The training backend is neither "sklearn" nor "spark"."""
