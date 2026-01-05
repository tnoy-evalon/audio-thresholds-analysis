import pandas as pd
import numpy as np
import random
from .similarity import compute_similarity
from .reference_samples import get_reference_samples_per_user

def _compute_asv_similarities(test_row: pd.Series, reference_rows: pd.DataFrame, method: str = "mean") -> dict[str, float]:
    """
    Compute the ASV score for a test row and a set of reference rows.
    """
    # if the tested row appears in the reference rows, remove it from the reference rows
    if test_row["session_id"] in reference_rows["session_id"].values:
        reference_rows = reference_rows[reference_rows["session_id"] != test_row["session_id"]]

    # compute the similarity between the test row and the reference rows
    emb_columns = [col for col in reference_rows.columns if col.startswith("emb_")]
    model_names = [e.removeprefix("emb_") for e in emb_columns]

    ret = {}
    for i, emb_column in enumerate(emb_columns):
        model_similarities = []

        for _, r in reference_rows.iterrows():
            similarity = compute_similarity(test_row[emb_column], r[emb_column])
            model_similarities.append(similarity)

        if method == "mean":
            ret[model_names[i]] = np.mean(model_similarities)
        elif method == "max":
            ret[model_names[i]] = np.max(model_similarities)
        elif method == "min":
            ret[model_names[i]] = np.min(model_similarities)
        elif method == "median":
            ret[model_names[i]] = np.median(model_similarities)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        if len(model_names) > 1:
            ret['combined_models'] = np.mean(list(ret.values()))

    return ret

def compute_asv(df: pd.DataFrame, method: str = "mean", max_samples: int = 3) -> pd.DataFrame:
    reference_samples = get_reference_samples_per_user(df, max_samples=max_samples)
    
    # Collect results in lists first, then create DataFrames at once (avoids FutureWarning)
    true_results = []
    true_indices = []
    for i, test_row in df.iterrows():
        asv_similarities = _compute_asv_similarities(test_row, reference_samples[test_row["user_id"]], method)
        true_results.append(asv_similarities)
        true_indices.append(i)
    
    false_results = []
    false_indices = []
    for i, test_row in df.iterrows():
        other_user_ids = [user_id for user_id in reference_samples.keys() if user_id != test_row["user_id"]]
        random_user_id = random.choice(other_user_ids)
        asv_similarities = _compute_asv_similarities(test_row, reference_samples[random_user_id], method)
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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    ret = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
    return ret