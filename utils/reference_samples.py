import pandas as pd


def get_reference_samples_per_user(
    df: pd.DataFrame,
    # snr_threshold: float = 0.0,
    # squim_stoi_threshold: float = 0.6,
    # squim_pesq_threshold: float = 1.5,
    # squim_si_sdr_threshold: float = 0.0,
    # vad_ratio_threshold: float = 0.2,
    max_samples: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Get reference samples from the dataframe.
    """

    ret = {}
    # filter df by thresholds:
    # df = df[(df["snr"] > snr_threshold) | (df["snr"].isna())]
    # df = df[(df["squim_STOI"] > squim_stoi_threshold) | (df["squim_STOI"].isna())]
    # df = df[(df["squim_PESQ"] > squim_pesq_threshold) | (df["squim_PESQ"].isna())]
    # df = df[(df["squim_SI-SDR"] > squim_si_sdr_threshold) | (df["squim_SI-SDR"].isna())]
    # df = df[(df["vad_ratio"] > vad_ratio_threshold) | (df["vad_ratio"].isna())]

    # group by user_id
    g = df.groupby("user_id")

    # for each user, sort by snr, squim_stoi, squim_pesq, squim_si_sdr
    for user_id, user_df in g:
        ret[user_id] = user_df.sort_values(by=["snr", "squim_STOI", "squim_PESQ", "squim_SI-SDR"], ascending=False).head(max_samples)
    
    return ret
