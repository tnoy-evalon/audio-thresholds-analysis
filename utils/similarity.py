import numpy as np


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Return cosine similarity in [0.0, 1.0] as a Python float.

    Accepts tensors or numpy arrays.
    If embeddings contain batch dimension, pairwise similarity for each pair is returned.
    """

    assert emb1.shape == emb2.shape, "Embeddings must have the same shape"

    # Normalize last dim to unit vectors to ensure cosine in [-1,1]
    # emb1 = np.linalg.norm(emb1, axis=-1)
    # emb2 = np.linalg.norm(emb2, axis=-1)

    cosine_sim = np.dot(emb1.squeeze(), emb2.squeeze()) / (np.linalg.norm(emb1, axis=-1) * np.linalg.norm(emb2, axis=-1))
    if cosine_sim.shape == (1,):
        return cosine_sim[0]
    
    return cosine_sim

if __name__ == "__main__":
    emb1 = np.random.randn(1,128)
    emb2 = np.random.randn(1,128)
    similarity = compute_similarity(emb1, emb2)

    print(similarity)

    print(compute_similarity(emb1, emb1))