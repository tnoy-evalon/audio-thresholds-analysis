import logging
import time

from s3pathlib import S3Path as _S3Path

logger = logging.getLogger(__name__)

S3_READ_MAX_RETRIES = 3
S3_READ_RETRY_DELAY = 0.5  # seconds


class S3Path(_S3Path):
    def __str__(self) -> str:
        """Return a string representation of the S3 path."""
        return self.uri


def read_s3_bytes_with_retry(s3_path: S3Path, max_retries: int = S3_READ_MAX_RETRIES) -> bytes:
    """Read bytes from S3 with retry mechanism.

    Converts S3-specific exceptions to standard exceptions to avoid pickling issues
    in multiprocessing contexts.
    """
    last_exception: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return s3_path.read_bytes()
        except Exception as e:
            last_exception = e
            exception_type = type(e).__name__
            logger.warning(
                f"S3 read attempt {attempt}/{max_retries} failed for {s3_path}: "
                f"{exception_type}: {e}"
            )
            if attempt < max_retries:
                time.sleep(S3_READ_RETRY_DELAY * attempt)  # exponential backoff

    # Convert to a standard exception that can be pickled in multiprocessing
    error_msg = f"Failed to read from S3 after {max_retries} attempts: {s3_path}"
    if last_exception is not None:
        error_msg += f" (last error: {type(last_exception).__name__}: {last_exception})"
    raise OSError(error_msg)

