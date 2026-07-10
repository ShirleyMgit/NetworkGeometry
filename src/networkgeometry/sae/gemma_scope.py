import numpy as np
from networkgeometry.geometry.linalg import source_pcs, mean_center
from networkgeometry.geometry.circle_fit import fit_circle

def cluster_decoder(decoder: np.ndarray, threshold: float) -> list:
    normed = decoder / (np.linalg.norm(decoder, axis=0, keepdims=True) + 1e-12)
    similarity = normed.T @ normed
    m = decoder.shape[1]
    adjacency = (similarity >= threshold)
    seen, clusters = set(), []
    for start in range(m):
        if start in seen:
            continue
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node); component.append(node)
            stack.extend(np.where(adjacency[node])[0])
        clusters.append(sorted(component))
    return clusters

def cluster_circle_score(reconstructed: np.ndarray) -> float:
    matrix = reconstructed.T                        # (d, n_points)
    u = source_pcs(matrix)
    scores = (u[:, :2].T @ mean_center(matrix)).T
    return fit_circle(scores).normalized_residual

def load_sae(release: str, sae_id: str):
    from sae_lens import SAE
    return SAE.from_pretrained(release, sae_id)
