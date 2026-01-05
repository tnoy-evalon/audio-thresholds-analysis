"""
Threshold Analysis Dashboard
A web-based GUI for analyzing dataframe data.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from utils.asv import asv_metrics, compute_asv
from utils.load_waveform import load_waveform
from utils.s3_utils import S3Path, read_s3_bytes_with_retry


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
    'squim_STOI', 'squim_PESQ', 'squim_SI-SDR'
]

# Default threshold values for numeric columns
DEFAULT_THRESHOLDS = {
    'vad_ratio': 0.2,
    'snr': 0.0,
    'squim_STOI': 0.6,
    'squim_PESQ': 1.5,
    'squim_SI-SDR': 0.0,
}

# Default threshold values for ASV models
DEFAULT_ASV_THRESHOLDS = {
    'WESpeakerONNX': 0.35,
    'WavLM': 0.90,
    'Titanet': 0.4,
    'combined_models': 0.5,
}


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
        try:
            df = load_data_from_upload(uploaded_file)
            st.session_state["df"] = df
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
        return df
    
    # Check for local file load request
    if st.session_state.get("_load_local") and st.session_state.get("_file_path"):
        try:
            df = load_data_from_path(st.session_state["_file_path"])
            st.session_state["df"] = df
            st.sidebar.success(f"Loaded: {st.session_state['_file_path']}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
        return df
    
    # Fall back to session state
    return st.session_state.get("df")


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
        
        st.text_input(
            "File path",
            value="s3://asv-data/analysis/predictions.pkl",
            help="Path to a local or S3 .parquet, .csv, or .pkl file",
            key="_file_path",
        )
        st.button("Load File", key="_load_local")


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
    fig = px.histogram(clean_data, nbins=10)
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
    return compute_asv(df, method=method, max_samples=max_samples)


def render_asv_model_histogram(model_name: str, true_similarities: pd.Series, false_similarities: pd.Series) -> float:
    """Render histogram and threshold input for a single ASV model.
    
    Returns the threshold value.
    """
    # Get current threshold (from session state or default)
    default_val = DEFAULT_ASV_THRESHOLDS.get(model_name, 0.5)
    current_threshold = st.session_state.get(f"asv_threshold_{model_name}", default_val)
    
    # Header
    st.markdown(f"**{model_name}**")
    
    # Overlaid histograms: true (green) and false (red)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=true_similarities.dropna(),
        nbinsx=20,
        name="True (same speaker)",
        marker_color="rgba(100, 200, 100, 0.6)",
    ))
    fig.add_trace(go.Histogram(
        x=false_similarities.dropna(),
        nbinsx=20,
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
        xaxis=dict(visible=True, title=None),
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
    st.markdown(f"**Accuracy:** {metrics['accuracy']:.1%}")
    st.caption(f"Precision: {metrics['precision']:.1%} · Recall: {metrics['recall']:.1%} · F1: {metrics['f1_score']:.1%}")
    
    return threshold


def render_speaker_verification(df: pd.DataFrame, asv_placeholder=None):
    """Render the speaker verification section."""
    st.subheader("Speaker Verification")
    
    # Check if we have embedding columns
    emb_columns = [col for col in df.columns if col.startswith("emb_")]
    if not emb_columns:
        st.warning("No embedding columns (emb_*) found in data")
        return
    
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
            options=["mean", "min", "max", "median"],
            index=0,
            key="asv_method"
        )
    
    # Compute ASV results (cached)
    # Use a hash of the dataframe to invalidate cache when data changes
    df_hash = str(hash(tuple(df['session_id'].tolist())))
    true_asv, false_asv = get_asv_results(df_hash, df, method, max_samples)
    
    if true_asv.empty:
        st.warning("No ASV results computed")
        return
    
    # Display histograms for each model (combined_models first if it exists)
    model_names = true_asv.columns.tolist()
    if 'combined_models' in model_names:
        model_names.remove('combined_models')
        model_names.insert(0, 'combined_models')
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


def render_data_preview(df: pd.DataFrame, display_cols: list[str], numeric_cols: list[str], thresholds: dict[str, float]):
    """Render the data preview table with rejected rows highlighted in red."""
    st.subheader("Data Preview")
    st.caption("Click on a row to select it for audio playback")
    
    preview_df = df[display_cols].head(100)
    
    if thresholds:
        # Calculate which rows pass thresholds
        passing_mask = calculate_passing_mask(df.head(100), numeric_cols, thresholds)
        
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
    
    max_rows = min(100, len(df))
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
    st.plotly_chart(waveform_fig, use_container_width=True, config={'displayModeBar': False})
    
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
    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric("Rows", f"{len(filtered_df):,} / {len(df)}")
    asv_placeholder = metrics_cols[1].empty()
    yield_placeholder = metrics_cols[2].empty()
    
    st.divider()
    
    # Speaker verification (uses filtered data)
    render_speaker_verification(filtered_df, asv_placeholder)
    
    st.divider()
    
    # Threshold analysis
    thresholds = {}
    if numeric_cols:
        thresholds = render_threshold_analysis(df, numeric_cols, yield_placeholder)
        st.divider()
    
    # Audio player
    render_audio_player(df)
    
    st.divider()
    
    # Data preview
    render_data_preview(df, display_cols, numeric_cols, thresholds)


if __name__ == "__main__":
    main()
