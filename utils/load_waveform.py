import logging
import tempfile
from os import PathLike

import numpy as np
import librosa

from .s3_utils import S3Path, read_s3_bytes_with_retry

logger = logging.getLogger(__name__)



def _load_wav_wrapper(path, sr=None):
    # Try torchaudio first
    try:
        waveform, sample_rate = librosa.load(path, sr=sr)
    except Exception as e:
        logger.exception(f"librosa failed to load {path}: {e}")
        
        raise e
    
    return waveform, sample_rate


def load_waveform(
    wav_path: PathLike, sample_rate: int | None = 16000
) -> tuple[np.ndarray, int]:
    if str(wav_path).startswith("s3://"):
        s3_path = S3Path(wav_path)

        # download the file from s3 to a temporary file with retry mechanism
        file_bytes = read_s3_bytes_with_retry(s3_path)
        with tempfile.NamedTemporaryFile(delete=True, delete_on_close=False, mode="wb") as temp_file:
            temp_file.write(file_bytes)
            temp_file.close()  # Close so audio decoder can open it
            waveform, sr = _load_wav_wrapper(temp_file.name, sample_rate)
    else:
        waveform, sr = _load_wav_wrapper(str(wav_path), sample_rate)

    if sample_rate is not None:
        assert sr == sample_rate
    else:
        sample_rate = sr

    # Ensure mono and channel-first shape (1, num_samples)
    if waveform.ndim == 2:
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # else already (1, N)
    # elif waveform.ndim == 1:
    #     waveform = waveform[np.newaxis, :]

    return waveform, sample_rate


