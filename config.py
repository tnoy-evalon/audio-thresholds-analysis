"""
Configuration file for default threshold values.
"""

# Default threshold values for numeric columns
DEFAULT_THRESHOLDS = {
    'vad_ratio': 0.2,
    'snr': 0.0,
    'squim_STOI': 0.6,
    'squim_PESQ': 1.5,
    'squim_SI-SDR': 0.0,
    'adfd_score': 0.5,
}

# Default threshold values for ASV models
DEFAULT_ASV_THRESHOLDS = {
    'WESpeakerONNX': 0.72,
    'WavLM': 0.90,
    'Titanet': 0.6,
    'combined_models': 0.65,
}
