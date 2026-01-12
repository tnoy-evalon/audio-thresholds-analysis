import pandas as pd
import numpy as np
import random
from .similarity import compute_similarity
from .reference_samples import get_reference_samples_per_user


def compute_centroid_score(test_vector: np.ndarray, reference_vectors: list[np.ndarray]) -> float:
    """
    Compute speaker verification score using centroid-based approach.
    
    Args:
        test_vector: embedding vector for the test audio (1D array)
        reference_vectors: list of 1-3 embedding vectors for enrolled speaker
    
    Returns:
        score: float in (0, 1) — similarity to claimed speaker
    """
    if len(reference_vectors) == 0:
        return 0.0
    
    # Stack reference vectors and compute centroid
    ref_stack = np.stack(reference_vectors, axis=0)
    centroid = np.mean(ref_stack, axis=0)
    
    # Re-normalize the centroid
    centroid = centroid / np.linalg.norm(centroid)
    
    # Compute similarity between test vector and centroid
    score = compute_similarity(test_vector, centroid)
    
    return float(score)


def compute_reference_cohesion(reference_vectors: list[np.ndarray]) -> float | None:
    """
    Compute cohesion of reference vectors (how tightly clustered they are).
    
    Args:
        reference_vectors: list of embedding vectors for enrolled speaker
    
    Returns:
        cohesion: mean cosine similarity between all pairs of references,
                  or None if fewer than 2 references
    """
    n = len(reference_vectors)
    if n < 2:
        return None
    
    # Compute pairwise similarities for all pairs i < j
    pair_similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(reference_vectors[i], reference_vectors[j])
            pair_similarities.append(sim)
    
    return float(np.mean(pair_similarities))


def compute_similarity_variance(pairwise_similarities: list[float]) -> float:
    """
    Compute variance of pairwise similarities between test and references.
    
    Args:
        pairwise_similarities: list of similarity scores [s₁, s₂, ...]
    
    Returns:
        variance: float, with max possible value of 0.25 for similarities in (0,1)
    """
    if len(pairwise_similarities) < 2:
        return 0.0
    
    return float(np.var(pairwise_similarities))


def compute_confidence(
    reference_vectors: list[np.ndarray],
    pairwise_similarities: list[float]
) -> float | None:
    """
    Compute confidence score for the speaker verification result.
    
    Args:
        reference_vectors: list of embedding vectors for enrolled speaker
        pairwise_similarities: list of similarity scores between test and each reference
    
    Returns:
        confidence: float in (0, 1) — reliability of the score,
                    or None if only 1 reference (undefined confidence)
    """
    if len(reference_vectors) < 2:
        return None
    
    # Compute reference cohesion
    cohesion = compute_reference_cohesion(reference_vectors)
    if cohesion is None:
        return None
    
    # Compute similarity variance
    similarity_variance = compute_similarity_variance(pairwise_similarities)
    
    # Combine: confidence = cohesion * (1 - 2 * similarity_variance)
    # The 2x scaling maps variance range (0, 0.25) to (0, 0.5)
    confidence = cohesion * (1 - 2 * similarity_variance)
    
    # Clamp to (0, 1)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    
    return confidence


def compute_score_with_confidence(
    test_vector: np.ndarray,
    reference_vectors: list[np.ndarray]
) -> dict[str, float | None]:
    """
    Compute speaker verification score and confidence using the advanced approach.
    
    Args:
        test_vector: embedding vector for the test audio (1D array)
        reference_vectors: list of 1-3 embedding vectors for enrolled speaker
    
    Returns:
        dict with:
            - 'score': float in (0, 1) — similarity to claimed speaker
            - 'confidence': float in (0, 1) or None — reliability of the score
            - 'pairwise_similarities': list of individual similarities to each reference
            - 'cohesion': float or None — reference vector cohesion
    """
    if len(reference_vectors) == 0:
        return {
            'score': 0.0,
            'confidence': None,
            'pairwise_similarities': [],
            'cohesion': None
        }
    
    # Step 1: Compute pairwise similarities
    pairwise_similarities = [
        float(compute_similarity(test_vector, ref))
        for ref in reference_vectors
    ]
    
    # Step 2: Compute centroid-based score
    score = compute_centroid_score(test_vector, reference_vectors)
    
    # Step 3: Compute confidence (None if only 1 reference)
    confidence = compute_confidence(reference_vectors, pairwise_similarities)
    
    # Also return cohesion for diagnostics
    cohesion = compute_reference_cohesion(reference_vectors)
    
    return {
        'score': score,
        'confidence': confidence,
        'pairwise_similarities': pairwise_similarities,
        'cohesion': cohesion
    }


def _compute_asv_similarities(test_row: pd.Series, reference_rows: pd.DataFrame, method: str = "mean", max_samples: int | None = None) -> dict:
    """
    Compute the ASV score for a test row and a set of reference rows (legacy method).
    
    Returns a dict with:
        - Aggregated similarity per model (e.g., 'WavLM': 0.85)
        - 'combined_models': mean of all model similarities (if multiple models)
        - 'reference_session_ids': list of session_ids used as references
        - 'reference_similarities': dict mapping model_name -> list of individual similarities
    """
    # if the tested row appears in the reference rows, remove it from the reference rows
    if test_row["session_id"] in reference_rows["session_id"].values:
        reference_rows = reference_rows[reference_rows["session_id"] != test_row["session_id"]]
    
    # Limit the number of reference samples after filtering out the test row
    if max_samples is not None and len(reference_rows) > max_samples:
        reference_rows = reference_rows.head(max_samples)

    # compute the similarity between the test row and the reference rows
    emb_columns = [col for col in reference_rows.columns if col.startswith("emb_")]
    model_names = [e.removeprefix("emb_") for e in emb_columns]

    ret = {}
    reference_similarities = {}
    reference_session_ids = reference_rows["session_id"].tolist()
    
    for i, emb_column in enumerate(emb_columns):
        model_similarities = []

        for _, r in reference_rows.iterrows():
            similarity = compute_similarity(test_row[emb_column], r[emb_column])
            model_similarities.append(similarity)

        reference_similarities[model_names[i]] = model_similarities

        if method == "mean":
            ret[model_names[i]] = np.mean(model_similarities) if model_similarities else 0.0
        elif method == "max":
            ret[model_names[i]] = np.max(model_similarities) if model_similarities else 0.0
        elif method == "min":
            ret[model_names[i]] = np.min(model_similarities) if model_similarities else 0.0
        elif method == "median":
            ret[model_names[i]] = np.median(model_similarities) if model_similarities else 0.0
        else:
            raise ValueError(f"Invalid method: {method}")

    # Add combined_models after all model scores are computed
    if len(model_names) > 1:
        model_scores = [ret[m] for m in model_names]
        ret['combined_models'] = np.mean(model_scores)

    ret['reference_session_ids'] = reference_session_ids
    ret['reference_similarities'] = reference_similarities

    return ret


def _compute_asv_advanced(test_row: pd.Series, reference_rows: pd.DataFrame, max_samples: int | None = None) -> dict:
    """
    Compute the ASV score using centroid-based approach with confidence.
    
    Returns a dict with:
        - Score per model (e.g., 'WavLM': 0.85) - centroid-based similarity
        - Confidence per model (e.g., 'WavLM_confidence': 0.92)
        - Cohesion per model (e.g., 'WavLM_cohesion': 0.95)
        - 'combined_models': mean of all model scores (if multiple models)
        - 'combined_models_confidence': mean of all model confidences (if multiple models)
        - 'reference_session_ids': list of session_ids used as references
        - 'reference_similarities': dict mapping model_name -> list of individual similarities
    """
    # if the tested row appears in the reference rows, remove it from the reference rows
    if test_row["session_id"] in reference_rows["session_id"].values:
        reference_rows = reference_rows[reference_rows["session_id"] != test_row["session_id"]]
    
    # Limit the number of reference samples after filtering out the test row
    if max_samples is not None and len(reference_rows) > max_samples:
        reference_rows = reference_rows.head(max_samples)

    # Get embedding columns and model names
    emb_columns = [col for col in reference_rows.columns if col.startswith("emb_")]
    model_names = [e.removeprefix("emb_") for e in emb_columns]

    ret = {}
    reference_similarities = {}
    reference_session_ids = reference_rows["session_id"].tolist()
    confidences = {}
    
    for i, emb_column in enumerate(emb_columns):
        model_name = model_names[i]
        
        # Extract reference vectors for this model
        reference_vectors = [r[emb_column] for _, r in reference_rows.iterrows()]
        test_vector = test_row[emb_column]
        
        if len(reference_vectors) == 0:
            ret[model_name] = 0.0
            ret[f'{model_name}_confidence'] = None
            ret[f'{model_name}_cohesion'] = None
            reference_similarities[model_name] = []
            continue
        
        # Use the advanced scoring with confidence
        result = compute_score_with_confidence(test_vector, reference_vectors)
        
        ret[model_name] = result['score']
        ret[f'{model_name}_confidence'] = result['confidence']
        ret[f'{model_name}_cohesion'] = result['cohesion']
        reference_similarities[model_name] = result['pairwise_similarities']
        
        if result['confidence'] is not None:
            confidences[model_name] = result['confidence']

    # Add combined_models after all model scores are computed
    if len(model_names) > 1:
        model_scores = [ret[m] for m in model_names]
        ret['combined_models'] = np.mean(model_scores)
        
        # Combined confidence (mean of available confidences)
        if confidences:
            ret['combined_models_confidence'] = np.mean(list(confidences.values()))
        else:
            ret['combined_models_confidence'] = None

    ret['reference_session_ids'] = reference_session_ids
    ret['reference_similarities'] = reference_similarities

    return ret


def compute_asv(
    df: pd.DataFrame,
    method: str = "mean",
    max_samples: int = 3,
    use_advanced: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ASV (Automatic Speaker Verification) similarities for all samples.
    
    Args:
        df: DataFrame with embeddings and user_id/session_id columns
        method: Aggregation method for legacy approach ("mean", "max", "min", "median")
                Ignored when use_advanced=True
        max_samples: Maximum number of reference samples to use per comparison
        use_advanced: If True, use centroid-based scoring with confidence measurement
                      If False, use legacy aggregation-based scoring
    
    Returns:
        Tuple of (true_results, false_results) DataFrames
        - true_results: scores comparing each sample to its own user's references
        - false_results: scores comparing each sample to a random other user's references
    """
    # Get all reference samples per user (without limiting) - we'll limit per item in the loop
    reference_samples = get_reference_samples_per_user(df, max_samples=None)
    
    # Collect results in lists first, then create DataFrames at once (avoids FutureWarning)
    true_results = []
    true_indices = []
    for i, test_row in df.iterrows():
        ref_rows = reference_samples[test_row["user_id"]]
        if use_advanced:
            asv_similarities = _compute_asv_advanced(test_row, ref_rows, max_samples=max_samples)
        else:
            asv_similarities = _compute_asv_similarities(test_row, ref_rows, method, max_samples=max_samples)
        true_results.append(asv_similarities)
        true_indices.append(i)
    
    false_results = []
    false_indices = []
    for i, test_row in df.iterrows():
        other_user_ids = [user_id for user_id in reference_samples.keys() if user_id != test_row["user_id"]]
        random_user_id = random.choice(other_user_ids)
        ref_rows = reference_samples[random_user_id]
        if use_advanced:
            asv_similarities = _compute_asv_advanced(test_row, ref_rows, max_samples=max_samples)
        else:
            asv_similarities = _compute_asv_similarities(test_row, ref_rows, method, max_samples=max_samples)
        false_results.append(asv_similarities)
        false_indices.append(i)

    ret_true = pd.DataFrame(true_results, index=true_indices) if true_results else pd.DataFrame()
    ret_false = pd.DataFrame(false_results, index=false_indices) if false_results else pd.DataFrame()
    
    return ret_true, ret_false

def asv_metrics(true_preds: pd.Series | np.ndarray, false_preds: pd.Series | np.ndarray, threshold: float = 0.5) -> dict[str, float]:

    tp = np.count_nonzero(true_preds > threshold)
    fp = np.count_nonzero(false_preds > threshold)
    fn = np.count_nonzero(true_preds <= threshold)
    tn = np.count_nonzero(false_preds <= threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    ret = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "fnr": fnr,
    }
    return ret