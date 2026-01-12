"""
Threshold Analysis Dashboard
A web-based GUI for analyzing dataframe data.
"""

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from config import DEFAULT_ASV_THRESHOLDS, DEFAULT_THRESHOLDS
from utils.asv import asv_metrics, compute_asv
from utils.load_waveform import load_waveform
from utils.s3_utils import S3Path, read_s3_bytes_with_retry

# os.environ.pop("AWS_ACCESS_KEY_ID", None)
# os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
# os.environ.pop("AWS_DEFAULT_REGION", None)

# Load AWS credentials from Streamlit secrets (for cloud deployment)
if hasattr(st, "secrets") and "AWS_ACCESS_KEY_ID" in st.secrets:
    
    os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
    os.environ["AWS_DEFAULT_REGION"] = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Threshold Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Columns safe to display (no numpy arrays or complex objects)
DISPLAY_COLUMNS = [
    'created_at', 'session_id', 'user_id', 'text',
    'vad_ratio', 'snr',
    'squim_STOI', 'squim_PESQ', 'squim_SI-SDR',
    'adfd_score',
    'status_reason'
]


# =============================================================================
# Audio Utilities
# =============================================================================

def waveform_to_audio_bytes(waveform, sample_rate: int) -> bytes:
    """Convert numpy waveform to WAV bytes for st.audio()."""
    # Ensure 1D array
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


# =============================================================================
# Data Loading
# =============================================================================

def load_data_from_path(file_path: str | Path) -> pd.DataFrame:
    """Load dataframe from parquet, csv, or pickle file. Supports S3 paths."""
    file_path_str = str(file_path)
    
    # Handle S3 paths
    if file_path_str.startswith("s3://"):
        s3_path = S3Path(file_path_str)
        file_bytes = read_s3_bytes_with_retry(s3_path)
        buffer = io.BytesIO(file_bytes)
        
        if file_path_str.endswith(".parquet"):
            return pd.read_parquet(buffer)
        elif file_path_str.endswith(".csv"):
            return pd.read_csv(buffer)
        elif file_path_str.endswith(".pkl"):
            return pd.read_pickle(buffer)
        else:
            raise ValueError(f"Unsupported file format: {file_path_str}")
    
    # Local file handling
    file_path = Path(file_path)
    loaders = {
        ".parquet": pd.read_parquet,
        ".csv": pd.read_csv,
        ".pkl": pd.read_pickle,
    }
    loader = loaders.get(file_path.suffix)
    if loader is None:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    return loader(file_path)


def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    """Load dataframe from uploaded file object."""
    if uploaded_file.name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    elif uploaded_file.name.endswith(".pkl"):
        return pd.read_pickle(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)


def get_dataframe() -> pd.DataFrame | None:
    """Get dataframe from upload, local file, or session state."""
    df = None
    
    # Check for uploaded file
    uploaded_file = st.session_state.get("_uploaded_file")
    if uploaded_file is not None:
        # Only reload if it's a different file
        if st.session_state.get("_loaded_source") != uploaded_file.name:
            try:
                df = load_data_from_upload(uploaded_file)
                st.session_state["df"] = df
                st.session_state["_loaded_source"] = uploaded_file.name
                st.cache_data.clear()  # Clear cached computations for new data
                st.sidebar.success(f"Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return st.session_state.get("df")
        return st.session_state.get("df")
    
    # Check for local file load request
    file_path = st.session_state.get("_file_path")
    if st.session_state.get("_load_local") and file_path:
        try:
            df = load_data_from_path(file_path)
            st.session_state["df"] = df
            st.session_state["_loaded_source"] = file_path
            st.cache_data.clear()  # Clear cached computations for new data
            st.sidebar.success(f"Loaded: {file_path}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
        return df if df is not None else st.session_state.get("df")
    
    # Fall back to session state
    return st.session_state.get("df")


def load_adfd_predictions() -> pd.DataFrame | None:
    """Load ADFD predictions from upload or local file."""
    adfd_df = None
    
    # Check for uploaded ADFD file
    uploaded_adfd = st.session_state.get("_adfd_uploaded_file")
    if uploaded_adfd is not None:
        if st.session_state.get("_adfd_loaded_source") != uploaded_adfd.name:
            try:
                adfd_df = load_data_from_upload(uploaded_adfd)
                st.session_state["adfd_df"] = adfd_df
                st.session_state["_adfd_loaded_source"] = uploaded_adfd.name
                st.sidebar.success(f"Loaded ADFD: {uploaded_adfd.name}")
            except Exception as e:
                st.error(f"Error loading ADFD file: {e}")
                return st.session_state.get("adfd_df")
        return st.session_state.get("adfd_df")
    
    # Check for local ADFD file load request
    adfd_path = st.session_state.get("_adfd_file_path")
    if st.session_state.get("_load_adfd") and adfd_path:
        try:
            adfd_df = load_data_from_path(adfd_path)
            st.session_state["adfd_df"] = adfd_df
            st.session_state["_adfd_loaded_source"] = adfd_path
            st.sidebar.success(f"Loaded ADFD: {adfd_path}")
        except Exception as e:
            st.error(f"Error loading ADFD file: {e}")
        return adfd_df if adfd_df is not None else st.session_state.get("adfd_df")
    
    return st.session_state.get("adfd_df")


def merge_adfd_predictions(df: pd.DataFrame, adfd_df: pd.DataFrame) -> pd.DataFrame:
    """Merge ADFD predictions into main dataframe based on file_path.
    
    Args:
        df: Main dataframe with 'file_path' column
        adfd_df: ADFD predictions dataframe with 'file_path' and 'prediction' columns
    
    Returns:
        Main dataframe with 'adfd_score' column added
    """
    if 'file_path' not in df.columns:
        st.warning("Main dataframe missing 'file_path' column for ADFD merge")
        return df
    
    if 'file_path' not in adfd_df.columns or 'prediction' not in adfd_df.columns:
        st.warning("ADFD file must have 'file_path' and 'prediction' columns")
        return df
    
    # Select only needed columns and rename
    adfd_subset = adfd_df[['file_path', 'prediction']].copy()
    adfd_subset = adfd_subset.rename(columns={'prediction': 'adfd_score'})
    
    # Merge on file_path
    merged_df = df.merge(adfd_subset, on='file_path', how='left')
    
    return merged_df


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with data source options."""
    with st.sidebar:
        st.header("Data Source")
        
        # File upload (drag-drop or browse)
        st.file_uploader(
            "Upload a file",
            type=["parquet", "csv", "pkl"],
            help="Drag and drop or browse for a .parquet, .csv, or .pkl file",
            key="_uploaded_file",
        )
        
        st.divider()
        st.caption("Or load from path:")
        
        with st.form("load_file_form"):
            st.text_input(
                "File path",
                value="s3://asv-data/analysis/predictions.pkl",
                help="Path to a local or S3 .parquet, .csv, or .pkl file",
                key="_file_path",
            )
            st.form_submit_button("Load File", key="_load_local")
        
        # ADFD Predictions section
        st.divider()
        st.header("ADFD Predictions")
        
        st.file_uploader(
            "Upload ADFD file",
            type=["parquet", "csv", "pkl"],
            help="Upload ADFD predictions file with 'file_path' and 'prediction' columns",
            key="_adfd_uploaded_file",
        )
        
        st.caption("Or load from path:")
        
        with st.form("load_adfd_form"):
            st.text_input(
                "ADFD file path",
                value="",
                help="Path to ADFD predictions file (.parquet, .csv, or .pkl)",
                key="_adfd_file_path",
            )
            st.form_submit_button("Load ADFD File", key="_load_adfd")


def render_metrics(df: pd.DataFrame, yield_placeholder):
    """Render the top metrics row."""
    col1, col2, col3, _ = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.2f} MB")
    
    # Yield placeholder is in col4, filled later
    return yield_placeholder


def render_column_histogram(col_name: str, col_data: pd.Series) -> float:
    """Render histogram, stats, and threshold input for a single column.
    
    Returns the threshold value.
    """
    clean_data = col_data.dropna()
    total = len(col_data)
    missing = col_data.isna().sum()
    
    # Get current threshold (from session state or default)
    default_val = DEFAULT_THRESHOLDS.get(col_name, 0.0)
    current_threshold = st.session_state.get(f"threshold_{col_name}", default_val)
    
    # Header with stats
    st.markdown(f"**# {col_name}**")
    st.caption(
        f"Missing: {missing} ({missing / total * 100:.0f}%) · "
        f"Distinct: {col_data.nunique()} ({col_data.nunique() / total * 100:.0f}%)"
    )
    
    # Compact histogram with threshold line
    bin_size = (clean_data.max() - clean_data.min()) / 10
    fig = px.histogram(clean_data)
    fig.update_traces(xbins=dict(start=clean_data.min(), end=clean_data.max(), size=bin_size))
    fig.add_vline(
        x=current_threshold,
        line_color="red",
        line_width=2,
        line_dash="solid",
    )
    fig.update_layout(
        showlegend=False,
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        bargap=0.1,
    )
    fig.update_traces(marker_color='#6C8EBF')
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    # Min/Max/Avg stats
    st.caption(
        f"Min {clean_data.min():.2f} · "
        f"Max {clean_data.max():.2f} · "
        f"Avg {clean_data.mean():.2f}"
    )
    
    # Threshold input
    threshold = st.number_input(
        "Threshold",
        value=default_val,
        step=0.1,
        key=f"threshold_{col_name}",
    )
    
    # Per-column yield (NaN counts as passing)
    above = (col_data.isna() | (col_data >= threshold)).sum()
    yield_pct = (above / total * 100) if total > 0 else 0
    st.markdown(f"**Yield:** {above}/{total} ({yield_pct:.1f}%)")
    
    return threshold


def calculate_passing_mask(df: pd.DataFrame, numeric_cols: list[str], thresholds: dict[str, float]) -> pd.Series:
    """Calculate which rows pass all thresholds (AND logic). NaN counts as passing."""
    mask = pd.Series(True, index=df.index)
    
    for col_name in numeric_cols:
        threshold = thresholds[col_name]
        mask &= df[col_name].isna() | (df[col_name] >= threshold)
    
    return mask


def calculate_combined_yield(df: pd.DataFrame, numeric_cols: list[str], thresholds: dict[str, float]) -> tuple[int, float]:
    """Calculate combined yield across all thresholds (AND logic)."""
    mask = calculate_passing_mask(df, numeric_cols, thresholds)
    count = mask.sum()
    percentage = (count / len(df) * 100) if len(df) > 0 else 0
    return count, percentage


def render_threshold_analysis(df: pd.DataFrame, numeric_cols: list[str], yield_placeholder) -> dict[str, float]:
    """Render the threshold analysis section with histograms. Returns thresholds dict."""
    st.subheader("Quality Thresholds")
    cols = st.columns(len(numeric_cols))
    thresholds = {}
    
    for i, col_name in enumerate(numeric_cols):
        with cols[i]:
            thresholds[col_name] = render_column_histogram(col_name, df[col_name])
    
    # Calculate and display combined yield
    combined_count, combined_pct = calculate_combined_yield(df, numeric_cols, thresholds)
    yield_placeholder.metric(
        "Quality Threshold Yield",
        f"{combined_count}/{len(df)} ({combined_pct:.1f}%)"
    )
    
    return thresholds


# =============================================================================
# Speaker Verification
# =============================================================================

@st.cache_data(show_spinner="Computing ASV similarities...", hash_funcs={pd.DataFrame: lambda _: None})
def get_asv_results(df_hash: str, df: pd.DataFrame, method: str, max_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached wrapper for compute_asv. Returns (true_asv, false_asv).
    
    Note: df_hash is used for cache invalidation; the DataFrame itself is not hashed
    (hash_funcs returns None) to avoid issues with numpy array columns.
    """
    use_advanced = method == "advanced"
    return compute_asv(df, method=method, max_samples=max_samples, use_advanced=use_advanced)


def render_confidence_vs_score_scatter(
    model_name: str,
    true_asv: pd.DataFrame,
    false_asv: pd.DataFrame,
    threshold: float
):
    """Render a scatter plot of confidence (x) vs score (y) for true and false matches."""
    confidence_col = f"{model_name}_confidence"
    
    # Check if confidence column exists
    if confidence_col not in true_asv.columns:
        return
    
    # Get data, filtering out None/NaN confidence values
    true_scores = true_asv[model_name].dropna()
    true_conf = true_asv[confidence_col].dropna()
    false_scores = false_asv[model_name].dropna()
    false_conf = false_asv[confidence_col].dropna()
    
    # Align indices (only keep rows where both score and confidence exist)
    true_valid = true_scores.index.intersection(true_conf.index)
    false_valid = false_scores.index.intersection(false_conf.index)
    
    # Get aligned data arrays
    true_x = true_conf.loc[true_valid].values
    true_y = true_scores.loc[true_valid].values
    false_x = false_conf.loc[false_valid].values
    false_y = false_scores.loc[false_valid].values
    
    # Compute axis limits with margin
    all_x = np.concatenate([true_x, false_x]) if len(true_x) > 0 or len(false_x) > 0 else np.array([0, 1])
    all_y = np.concatenate([true_y, false_y]) if len(true_y) > 0 or len(false_y) > 0 else np.array([0, 1])
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    # Include threshold in y range
    y_min = min(y_min, threshold)
    y_max = max(y_max, threshold)
    
    # Add 5% margin
    x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.05
    y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.05
    
    x_range = [max(0, x_min - x_margin), min(1, x_max + x_margin)]
    y_range = [max(0, y_min - y_margin), min(1, y_max + y_margin)]
    
    fig = go.Figure()
    
    # True matches (green)
    fig.add_trace(go.Scatter(
        x=true_x,
        y=true_y,
        mode='markers',
        name='True (same speaker)',
        marker=dict(
            color='rgba(100, 200, 100, 0.6)',
            size=6,
        ),
    ))
    
    # False matches (red)
    fig.add_trace(go.Scatter(
        x=false_x,
        y=false_y,
        mode='markers',
        name='False (different speaker)',
        marker=dict(
            color='rgba(200, 100, 100, 0.6)',
            size=6,
        ),
    ))
    
    # Trendline for true matches (green)
    slope_true = None
    if len(true_x) >= 2:
        z_true = np.polyfit(true_x, true_y, 1)
        slope_true = z_true[0]
        p_true = np.poly1d(z_true)
        x_line = np.array(x_range)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=p_true(x_line),
            mode='lines',
            name='True trendline',
            line=dict(color='rgba(100, 200, 100, 1)', width=2, dash='dash'),
            showlegend=False,
        ))
    
    # Trendline for false matches (red)
    slope_false = None
    if len(false_x) >= 2:
        z_false = np.polyfit(false_x, false_y, 1)
        slope_false = z_false[0]
        p_false = np.poly1d(z_false)
        x_line = np.array(x_range)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=p_false(x_line),
            mode='lines',
            name='False trendline',
            line=dict(color='rgba(200, 100, 100, 1)', width=2, dash='dash'),
            showlegend=False,
        ))
    
    # Threshold horizontal line
    fig.add_hline(
        y=threshold,
        line_color="red",
        line_width=2,
        line_dash="solid",
        annotation_text=f"threshold={threshold:.2f}",
        annotation_position="top right",
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(title="Confidence", range=x_range),
        yaxis=dict(title="Score", range=y_range),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    # Calculate correlation between confidence and accuracy
    # True matches: correct if score >= threshold
    # False matches: correct if score < threshold
    true_correct = (true_y >= threshold).astype(int)
    false_correct = (false_y < threshold).astype(int)
    
    # Combine all confidence and correctness values
    all_confidence = np.concatenate([true_x, false_x])
    all_correct = np.concatenate([true_correct, false_correct])
    
    # Calculate Pearson correlation
    metrics_parts = []
    if len(all_confidence) >= 2 and np.std(all_confidence) > 0 and np.std(all_correct) > 0:
        correlation = np.corrcoef(all_confidence, all_correct)[0, 1]
        metrics_parts.append(f"Corr: **{correlation:.3f}**")
    
    # Calculate slope difference (true slope - false slope)
    # Positive difference means true scores increase faster with confidence than false scores
    if slope_true is not None and slope_false is not None:
        slope_diff = slope_true - slope_false
        metrics_parts.append(f"Slope Δ: **{slope_diff:.3f}**")
    
    if metrics_parts:
        st.caption(" · ".join(metrics_parts))


def render_asv_model_histogram(model_name: str, true_similarities: pd.Series, false_similarities: pd.Series) -> float:
    """Render histogram and threshold input for a single ASV model.
    
    Returns the threshold value.
    """
    # Get current threshold (from session state or default)
    default_val = DEFAULT_ASV_THRESHOLDS.get(model_name, 0.5)
    current_threshold = st.session_state.get(f"asv_threshold_{model_name}", default_val)
    
    # Header
    st.markdown(f"**{model_name}**")
    
    # Calculate bin boundaries with fixed [0, 1] range
    true_clean = true_similarities.dropna()
    false_clean = false_similarities.dropna()
    num_bins = 100
    bin_size = 1.0 / num_bins
    
    # Overlaid histograms: true (green) and false (red)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=true_clean,
        xbins=dict(start=0.0, end=1.0, size=bin_size),
        name="True (same speaker)",
        marker_color="rgba(100, 200, 100, 0.6)",
    ))
    fig.add_trace(go.Histogram(
        x=false_clean,
        xbins=dict(start=0.0, end=1.0, size=bin_size),
        name="False (different speaker)",
        marker_color="rgba(200, 100, 100, 0.6)",
    ))
    fig.add_vline(
        x=current_threshold,
        line_color="red",
        line_width=2,
        line_dash="solid",
    )
    fig.update_layout(
        barmode='overlay',
        showlegend=False,
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=True, title=None, range=[0, 1]),
        yaxis=dict(visible=False),
        bargap=0.1,
    )
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    st.caption("🟩 True (same speaker) · 🟥 False (different speaker)")
    
    # Threshold input
    threshold = st.number_input(
        "Threshold",
        value=default_val,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key=f"asv_threshold_{model_name}",
    )
    
    # Compute metrics using asv_metrics
    metrics = asv_metrics(true_similarities, false_similarities, threshold)
    
    # Display key metrics
    st.markdown(f"**Accuracy:** {metrics['accuracy']:.1%} · **False Negatives:** {metrics['fnr']:.1%}")
    st.caption(f"Precision: {metrics['precision']:.1%} · Recall: {metrics['recall']:.1%} · F1: {metrics['f1_score']:.1%}")
    
    return threshold


def render_speaker_verification(df: pd.DataFrame, asv_placeholder=None):
    """Render the speaker verification section.
    
    Returns:
        tuple: (true_asv, false_asv) DataFrames, or (None, None) if no results
    """
    st.subheader("Speaker Verification")
    
    # Check if we have embedding columns
    emb_columns = [col for col in df.columns if col.startswith("emb_")]
    if not emb_columns:
        st.warning("No embedding columns (emb_*) found in data")
        return None, None
    
    # Configuration inputs
    col1, col2 = st.columns(2)
    with col1:
        max_samples = st.number_input(
            "Number of reference samples",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="asv_max_samples"
        )
    with col2:
        method = st.selectbox(
            "Method",
            options=["advanced", "mean", "min", "max", "median"],
            index=0,
            key="asv_method"
        )
    
    # Compute ASV results (cached)
    # Use a hash of the dataframe to invalidate cache when data changes
    df_hash = str(hash(tuple(df['session_id'].tolist())))
    true_asv, false_asv = get_asv_results(df_hash, df, method, max_samples)
    
    if true_asv.empty:
        st.warning("No ASV results computed")
        return None, None
    
    # Display histograms for each model (exclude metadata and auxiliary columns, combined_models first)
    metadata_cols = ['reference_session_ids', 'reference_similarities']
    # Exclude confidence and cohesion columns from histogram display (they're auxiliary metrics)
    model_names = [c for c in true_asv.columns 
                   if c not in metadata_cols 
                   and not c.endswith('_confidence') 
                   and not c.endswith('_cohesion')]
    if 'combined_models' in model_names:
        model_names.remove('combined_models')
        model_names.insert(0, 'combined_models')
    
    # Scatter plot for advanced method (confidence vs score)
    if method == "advanced":
        st.markdown("**Confidence vs Score**")
        scatter_cols = st.columns(len(model_names))
        for i, model_name in enumerate(model_names):
            with scatter_cols[i]:
                st.caption(model_name)
                threshold = st.session_state.get(f"asv_threshold_{model_name}", DEFAULT_ASV_THRESHOLDS.get(model_name, 0.5))
                render_confidence_vs_score_scatter(model_name, true_asv, false_asv, threshold)
    
    # Histograms
    cols = st.columns(len(model_names))
    for i, model_name in enumerate(model_names):
        with cols[i]:
            render_asv_model_histogram(model_name, true_asv[model_name], false_asv[model_name])
    
    # Update ASV accuracy in top metrics
    if asv_placeholder is not None:
        # Use combined_models if available, otherwise use the first (only) model
        primary_model = 'combined_models' if 'combined_models' in true_asv.columns else model_names[0]
        threshold = st.session_state.get(f"asv_threshold_{primary_model}", DEFAULT_ASV_THRESHOLDS.get(primary_model, 0.5))
        metrics = asv_metrics(true_asv[primary_model], false_asv[primary_model], threshold)
        asv_placeholder.metric("ASV Accuracy", f"{metrics['accuracy']:.1%}")
    
    return true_asv, false_asv


def render_worst_results(true_asv: pd.DataFrame, false_asv: pd.DataFrame, df: pd.DataFrame):
    """Render worst results section showing lowest true similarities and highest false similarities."""
    st.subheader("Worst Results Inspection")
    st.caption("Inspect the worst performing cases: lowest true similarities (false negatives) and highest false similarities (false positives)")
    
    if true_asv is None or true_asv.empty:
        return
    
    # Get model names (exclude metadata and auxiliary columns)
    metadata_cols = ['reference_session_ids', 'reference_similarities']
    model_names = [c for c in true_asv.columns 
                   if c not in metadata_cols 
                   and not c.endswith('_confidence') 
                   and not c.endswith('_cohesion')]
    if 'combined_models' in model_names:
        model_names.remove('combined_models')
        model_names.insert(0, 'combined_models')
    
    # Select which model to display (default to WESpeakerONNX if available)
    if len(model_names) > 1:
        default_idx = model_names.index('WESpeakerONNX') if 'WESpeakerONNX' in model_names else 0
        model_name = st.selectbox(
            "Select model for inspection",
            options=model_names,
            index=default_idx,
            key="worst_results_model"
        )
    else:
        model_name = model_names[0]
    
    # Get threshold for selected model
    threshold = st.session_state.get(f"asv_threshold_{model_name}", DEFAULT_ASV_THRESHOLDS.get(model_name, 0.5))
    
    # Sort indices for both tables
    worst_true_indices = true_asv[model_name].sort_values(ascending=True).index.tolist()
    worst_false_indices = false_asv[model_name].sort_values(ascending=False).index.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Worst True Matches (False Negatives)**")
        st.caption("Lowest similarity scores - same speaker pairs with low confidence. 🔴 = below threshold")
        
        # Build display dataframe with only key columns
        rows = []
        for idx in worst_true_indices:
            sim = true_asv.loc[idx, model_name]
            row_data = {
                '': '🔴' if sim < threshold else '',
                'similarity': sim,
                'ref_similarities': true_asv.loc[idx, 'reference_similarities'].get(model_name, []) if model_name != 'combined_models' else [],
                'id': df.loc[idx, 'session_id'] if 'session_id' in df.columns else idx,
                'reference_ids': true_asv.loc[idx, 'reference_session_ids'],
            }
            rows.append(row_data)
        
        worst_true_display = pd.DataFrame(rows, index=worst_true_indices)
        
        event_true = st.dataframe(
            worst_true_display,
            width='stretch',
            height=200,
            selection_mode="single-row",
            on_select="rerun",
            key="worst_true_table"
        )
    
    with col2:
        st.markdown("**Worst False Matches (False Positives)**")
        st.caption("Highest similarity scores - different speaker pairs with high confidence. 🔴 = above threshold")
        
        # Build display dataframe with only key columns
        rows = []
        for idx in worst_false_indices:
            sim = false_asv.loc[idx, model_name]
            row_data = {
                '': '🔴' if sim > threshold else '',
                'similarity': sim,
                'ref_similarities': false_asv.loc[idx, 'reference_similarities'].get(model_name, []) if model_name != 'combined_models' else [],
                'id': df.loc[idx, 'session_id'] if 'session_id' in df.columns else idx,
                'reference_ids': false_asv.loc[idx, 'reference_session_ids'],
            }
            rows.append(row_data)
        
        worst_false_display = pd.DataFrame(rows, index=worst_false_indices)
        
        event_false = st.dataframe(
            worst_false_display,
            width='stretch',
            height=200,
            selection_mode="single-row",
            on_select="rerun",
            key="worst_false_table"
        )
    
    # Audio playback section
    if 'file_path' not in df.columns:
        return
    
    # Get selected row indices - only proceed if rows are actually selected
    selected_true_rows = event_true.selection.rows if event_true.selection.rows else []
    selected_false_rows = event_false.selection.rows if event_false.selection.rows else []
    
    # Only render audio if at least one row is selected
    if not selected_true_rows and not selected_false_rows:
        return
    
    selected_true_idx = selected_true_rows[0] if selected_true_rows else None
    selected_false_idx = selected_false_rows[0] if selected_false_rows else None
    
    st.markdown("---")
    audio_col1, audio_col2 = st.columns(2)
    
    with audio_col1:
        if selected_true_idx is not None:
            _render_worst_result_audio(
                df, 
                worst_true_indices[selected_true_idx], 
                true_asv.loc[worst_true_indices[selected_true_idx], 'reference_session_ids'],
                true_asv.loc[worst_true_indices[selected_true_idx], 'reference_similarities'].get(model_name, []) if model_name != 'combined_models' else [],
                "true"
            )
    
    with audio_col2:
        if selected_false_idx is not None:
            _render_worst_result_audio(
                df, 
                worst_false_indices[selected_false_idx], 
                false_asv.loc[worst_false_indices[selected_false_idx], 'reference_session_ids'],
                false_asv.loc[worst_false_indices[selected_false_idx], 'reference_similarities'].get(model_name, []) if model_name != 'combined_models' else [],
                "false"
            )


def _render_worst_result_audio(df: pd.DataFrame, test_idx, reference_ids: list, ref_similarities: list, prefix: str):
    """Render compact audio players for test sample and its references."""
    # Test sample
    test_row = df.loc[test_idx]
    test_file_path = test_row['file_path']
    test_session_id = test_row['session_id'] if 'session_id' in df.columns else test_idx
    
    st.markdown(f"**Test Sample:** `{test_session_id}`")
    try:
        test_audio = prepare_audio_data(test_file_path)
        st.audio(test_audio["original_bytes"], format="audio/wav")
    except Exception as e:
        st.error(f"Error loading audio: {e}")
    
    # Reference samples
    if reference_ids:
        st.markdown("**References:**")
        for i, ref_id in enumerate(reference_ids):
            # Find the reference row by session_id
            ref_rows = df[df['session_id'] == ref_id]
            if ref_rows.empty:
                st.caption(f"Reference {ref_id} not found in data")
                continue
            
            ref_row = ref_rows.iloc[0]
            ref_file_path = ref_row['file_path']
            
            if i < len(ref_similarities):
                sim_val = ref_similarities[i]
                # Handle numpy arrays by extracting the scalar value
                if hasattr(sim_val, 'item'):
                    sim_val = sim_val.item()
                sim_str = f" (sim={float(sim_val):.3f})"
            else:
                sim_str = ""
            st.caption(f"`{ref_id}`{sim_str}")
            try:
                ref_audio = prepare_audio_data(ref_file_path)
                st.audio(ref_audio["original_bytes"], format="audio/wav")
            except Exception as e:
                st.error(f"Error loading reference audio: {e}")


def render_data_preview(df: pd.DataFrame, display_cols: list[str], numeric_cols: list[str], thresholds: dict[str, float]):
    """Render the data preview table with rejected rows highlighted in red."""
    st.subheader("Data Preview")
    st.caption("Click on a row to select it for audio playback")
    
    preview_df = df[display_cols]
    
    if thresholds:
        # Calculate which rows pass thresholds
        passing_mask = calculate_passing_mask(df, numeric_cols, thresholds)
        
        # Style function to highlight rejected rows
        def highlight_rejected(row):
            idx = row.name
            if idx in passing_mask.index and not passing_mask.loc[idx]:
                return ['background-color: #8b0000'] * len(row)
            return [''] * len(row)
        
        styled_df = preview_df.style.apply(highlight_rejected, axis=1)
        st.dataframe(
            styled_df,
            width='stretch',
            selection_mode="single-row",
            on_select="rerun",
            key="data_preview_table"
        )
    else:
        st.dataframe(
            preview_df,
            width='stretch',
            selection_mode="single-row",
            on_select="rerun",
            key="data_preview_table"
        )


@st.cache_data(show_spinner="Loading audio...")
def prepare_audio_data(file_path: str, vad_mask_tuple=None) -> dict:
    """Load waveform and prepare audio data for playback and visualization.
    
    Returns dict with waveform, sample_rate, audio bytes, etc.
    """
    waveform, sr = load_waveform(file_path)
    
    # Ensure 1D
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    
    original_bytes = waveform_to_audio_bytes(waveform, sr)
    
    vad_bytes = None
    vad_mask = None
    if vad_mask_tuple is not None:
        vad_mask = np.array(vad_mask_tuple)
        
        # Ensure vad_mask is 1D boolean
        if vad_mask.ndim > 1:
            vad_mask = vad_mask.squeeze()
        vad_mask = vad_mask.astype(bool)
        
        # Align lengths
        min_len = min(len(vad_mask), len(waveform))
        aligned_mask = vad_mask[:min_len]
        aligned_waveform = waveform[:min_len]
        
        # Apply VAD mask
        filtered_waveform = aligned_waveform[aligned_mask]
        if len(filtered_waveform) > 0:
            vad_bytes = waveform_to_audio_bytes(filtered_waveform, sr)
    
    return {
        "waveform": waveform,
        "sample_rate": sr,
        "original_bytes": original_bytes,
        "vad_bytes": vad_bytes,
        "vad_mask": vad_mask,
    }


def create_waveform_plot(waveform: np.ndarray, sample_rate: int, vad_mask: np.ndarray | None = None) -> go.Figure:
    """Create a waveform plot with optional VAD mask highlighting."""
    # Create time axis
    duration = len(waveform) / sample_rate
    time = np.linspace(0, duration, len(waveform))
    
    fig = go.Figure()
    
    # Add VAD mask regions as background highlighting
    if vad_mask is not None and len(vad_mask) > 0:
        # Align mask length with waveform
        min_len = min(len(vad_mask), len(waveform))
        aligned_mask = vad_mask[:min_len].astype(bool)
        
        # Find contiguous regions where mask is True
        mask_diff = np.diff(np.concatenate([[False], aligned_mask, [False]]).astype(int))
        starts = np.where(mask_diff == 1)[0]
        ends = np.where(mask_diff == -1)[0]
        
        for start, end in zip(starts, ends):
            fig.add_vrect(
                x0=time[start] if start < len(time) else duration,
                x1=time[min(end - 1, len(time) - 1)],
                fillcolor="rgba(100, 200, 100, 0.3)",
                layer="below",
                line_width=0,
            )
    
    # Add waveform trace
    # Downsample for performance if waveform is very long
    max_points = 5000
    if len(waveform) > max_points:
        step = len(waveform) // max_points
        time_plot = time[::step]
        waveform_plot = waveform[::step]
    else:
        time_plot = time
        waveform_plot = waveform
    
    fig.add_trace(go.Scatter(
        x=time_plot,
        y=waveform_plot,
        mode='lines',
        line=dict(color='#6C8EBF', width=1),
        name='Waveform'
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=10, b=30),
        xaxis_title="Time (s)",
        yaxis_title=None,
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, zeroline=True, zerolinecolor='gray'),
    )
    
    return fig


def render_audio_player(df: pd.DataFrame):
    """Render audio player section with play buttons for selected row."""
    st.subheader("Audio Player")
    
    # Check required columns exist
    if 'file_path' not in df.columns:
        st.warning("No 'file_path' column found in data")
        return
    
    # Get selected row from table (read directly from dataframe widget state)
    table_state = st.session_state.get("data_preview_table", None)
    if table_state and table_state.selection and table_state.selection.rows:
        row_idx = table_state.selection.rows[0]
    else:
        st.info("👆 Select row in the table to listen to the audio")
        return
    
    max_rows = len(df)
    row_idx = min(row_idx, max_rows - 1)
    row = df.iloc[row_idx]
    file_path = row['file_path']
    has_vad_mask = 'vad_mask' in df.columns
    
    st.markdown(f"**Selected Row:** {row_idx}")
    st.caption(f"File: {file_path}")
    
    # Pre-load audio (cached)
    try:
        vad_mask_raw = row['vad_mask'] if has_vad_mask else None
        # Convert vad_mask to tuple for caching (numpy arrays aren't hashable)
        vad_mask_tuple = tuple(vad_mask_raw) if vad_mask_raw is not None else None
        audio_data = prepare_audio_data(file_path, vad_mask_tuple)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return
    
    # Waveform visualization
    waveform_fig = create_waveform_plot(
        audio_data["waveform"],
        audio_data["sample_rate"],
        audio_data["vad_mask"]
    )
    st.plotly_chart(waveform_fig, width='stretch', config={'displayModeBar': False})
    
    if has_vad_mask:
        st.caption("🟩 Green regions = VAD mask (speech detected)")
    
    # Audio players
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original**")
        st.audio(audio_data["original_bytes"], format="audio/wav")
    
    with col2:
        if audio_data["vad_bytes"] is not None:
            st.write("**VAD Filtered**")
            st.audio(audio_data["vad_bytes"], format="audio/wav")
        else:
            st.info("No 'vad_mask' column available")


# =============================================================================
# Main Application
# =============================================================================

def main():
    st.title("📊 Threshold Analysis Dashboard")
    
    render_sidebar()
    df = get_dataframe()
    
    if df is None:
        st.info("👈 Upload a file or load a local file to get started")
        return
    
    # Load and merge ADFD predictions if available
    adfd_df = load_adfd_predictions()
    if adfd_df is not None:
        df = merge_adfd_predictions(df, adfd_df)
    
    # Determine which columns to display
    display_cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    numeric_cols = [c for c in display_cols if df[c].dtype in ['float64', 'float32']]
    
    # Get current quality thresholds from session state
    current_thresholds = {}
    for col_name in numeric_cols:
        default_val = DEFAULT_THRESHOLDS.get(col_name, 0.0)
        current_thresholds[col_name] = st.session_state.get(f"threshold_{col_name}", default_val)
    
    # Filter dataframe based on quality thresholds for ASV
    if current_thresholds:
        passing_mask = calculate_passing_mask(df, numeric_cols, current_thresholds)
        filtered_df = df[passing_mask]
    else:
        filtered_df = df
    
    # Render metrics with placeholders
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Rows", f"{len(filtered_df):,} / {len(df)}")
    asv_placeholder = metrics_cols[1].empty()
    yield_placeholder = metrics_cols[2].empty()
    adfd_placeholder = metrics_cols[3].empty()
    
    st.divider()
    
    # Speaker verification (uses filtered data)
    true_asv, false_asv = render_speaker_verification(filtered_df, asv_placeholder)
    
    # Worst results inspection
    if true_asv is not None and not true_asv.empty:
        st.divider()
        render_worst_results(true_asv, false_asv, filtered_df)
    
    st.divider()
    
    # Threshold analysis
    thresholds = {}
    if numeric_cols:
        thresholds = render_threshold_analysis(df, numeric_cols, yield_placeholder)
        st.divider()
    
    # Calculate and display ADFD negatives (rows below adfd threshold among quality-passing rows)
    if 'adfd_score' in df.columns and thresholds:
        # Get rows that pass quality thresholds (excluding adfd_score from quality check)
        quality_cols = [c for c in numeric_cols if c != 'adfd_score']
        if quality_cols:
            quality_passing_mask = calculate_passing_mask(df, quality_cols, thresholds)
            quality_passing_df = df[quality_passing_mask]
        else:
            quality_passing_df = df
        
        # Count how many have adfd_score below threshold
        adfd_threshold = thresholds.get('adfd_score', DEFAULT_THRESHOLDS.get('adfd_score', 0.5))
        adfd_negatives = (quality_passing_df['adfd_score'] < adfd_threshold).sum()
        total_quality_passing = len(quality_passing_df)
        adfd_neg_pct = (adfd_negatives / total_quality_passing * 100) if total_quality_passing > 0 else 0
        
        adfd_placeholder.metric(
            "ADFD Negatives",
            f"{adfd_negatives}/{total_quality_passing} ({adfd_neg_pct:.1f}%)"
        )
    
    # Audio player
    render_audio_player(df)
    
    st.divider()
    
    # Data preview
    render_data_preview(df, display_cols, numeric_cols, thresholds)


if __name__ == "__main__":
    main()
