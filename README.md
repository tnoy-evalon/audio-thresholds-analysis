# Threshold Analysis Dashboard

A Streamlit-based web dashboard for analyzing audio quality thresholds and speaker verification metrics.

## Features

### Quality Thresholds
- Interactive histograms for audio quality metrics (VAD ratio, SNR, SQUIM scores)
- Adjustable thresholds with real-time yield calculations
- Visual threshold markers on histograms
- Combined yield showing samples passing all quality thresholds

### Speaker Verification (ASV)
- Computes speaker similarity scores using multiple embedding models
- Configurable reference sample count and aggregation method (mean/min/max/median)
- Overlaid histograms showing true (same speaker) vs false (different speaker) distributions
- Accuracy, precision, recall, and F1 metrics per model
- Combined model scoring when multiple models are available

### Audio Player
- Waveform visualization with VAD mask overlay
- Playback of original and VAD-filtered audio
- Click-to-select rows from data preview

### Data Preview
- Filterable data table with quality threshold highlighting
- Rejected samples shown in red
- Row selection for audio playback

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

### Data Sources
- Supports local files: `.parquet`, `.csv`, `.pkl`
- Supports S3 paths: `s3://bucket/path/to/file.pkl`

### Expected DataFrame Columns

**Required:**
- `session_id` - Unique session identifier
- `user_id` - User identifier for speaker verification
- `file_path` - Path to audio file (local or S3)

**Quality Metrics (optional):**
- `vad_ratio` - Voice activity detection ratio
- `snr` - Signal-to-noise ratio
- `squim_STOI`, `squim_PESQ`, `squim_SI-SDR` - SQUIM quality scores

**Speaker Verification (optional):**
- `emb_*` - Embedding columns (e.g., `emb_WavLM`, `emb_Titanet`)
- `vad_mask` - Boolean mask for VAD-filtered playback

## Project Structure

```
threshold-analysis/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── asv.py            # Speaker verification logic
│   ├── load_waveform.py  # Audio loading utilities
│   ├── reference_samples.py
│   ├── s3_utils.py       # S3 file access
│   └── similarity.py     # Embedding similarity
└── tmp/                   # Local data files
```

## Configuration

Default thresholds can be modified in `app.py`:

```python
DEFAULT_THRESHOLDS = {
    'vad_ratio': 0.2,
    'snr': 0.0,
    'squim_STOI': 0.6,
    'squim_PESQ': 1.5,
    'squim_SI-SDR': 0.0,
}

DEFAULT_ASV_THRESHOLDS = {
    'WESpeakerONNX': 0.35,
    'WavLM': 0.90,
    'Titanet': 0.4,
    'combined_models': 0.5,
}
```

