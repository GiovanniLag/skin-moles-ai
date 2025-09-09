import numpy as np
from typing import Dict

def compute_malignancy_index(
    probs: Dict[str, float],
    threshold: float = 0.05,
    hhi_weight: float = 0.6,
) -> Dict[str, float]:
    """
    Compute malignant/benign indices where HHI only *modifies* the raw sum.

    For each group G in {malignant, benign}:
        S_G   = sum of class probabilities in G with p >= threshold
        HHI_G = normalized Herfindahl-Hirschman index of {p_i / S_G}  (in [0,1])
        scale_G = (1 - hhi_weight) + hhi_weight * HHI_G
        weighted_G = S_G * scale_G

    The returned indices are:
        malignant = weighted_malignant / (weighted_malignant + weighted_benign)
        benign    = 1 - malignant

    Parameters
    ----------
    probs : dict of {str: float}
        Class probabilities (need not be exactly normalized).
    threshold : float, default=0.05
        Ignore classes with probability below this value.
    hhi_weight : float in [0,1], default=0.5
        0 -> no HHI effect (pure sums). 1 -> maximum HHI-based boost.

    Returns
    -------
    dict
        {"malignant": float, "benign": float} in [0, 1], summing to 1
        (both 0 if no class passes threshold).

    Notes
    -----
    Malignant labels recognized (case-insensitive): MEL(anoma), BCC, SCC.
    """
    malignant_keys = {
        "mel", "melanoma",
        "bcc", "basal cell carcinoma",
        "scc", "squamous cell carcinoma",
    }

    # Collect probs above threshold
    mal = [float(p) for k, p in probs.items() if p >= threshold and k.lower() in malignant_keys]
    ben = [float(p) for k, p in probs.items() if p >= threshold and k.lower() not in malignant_keys]

    def _sum(a):  # raw evidence
        return float(np.sum(a)) if a else 0.0

    def _hhi_norm(a):
        if not a:
            return 0.0
        s = float(np.sum(a))
        if s <= 0:
            return 0.0
        p = np.asarray(a, dtype=float) / s
        n = p.size
        if n == 1:
            return 1.0
        hhi = float(np.sum(p * p))                 # in [1/n, 1]
        return (hhi - 1.0 / n) / (1.0 - 1.0 / n)   # normalize to [0,1]

    mal_sum, ben_sum = _sum(mal), _sum(ben)
    mal_scale = (1.0 - hhi_weight) + hhi_weight * _hhi_norm(mal)
    ben_scale = (1.0 - hhi_weight) + hhi_weight * _hhi_norm(ben)

    mal_w = mal_sum * mal_scale
    ben_w = ben_sum * ben_scale

    total = mal_w + ben_w
    if total > 0:
        mal_idx = mal_w / total
        ben_idx = ben_w / total
    else:
        mal_idx = ben_idx = 0.0

    return {"malignant": float(mal_idx), "benign": float(ben_idx)}

def risk_factor(probs: dict[str, float]) -> str:
    """Determine risk factor based on the probabilities of the malignant classes.
    More malignant classes, such as melanoma, increase the risk factor.
    The risk factor is categorized as:
    - "Low" if risk score < 0.3
    - "Medium" if 0.3 <= risk score < 0.6
    - "High" if risk score >= 0.6
    The risk score is computed as a weighted sum of probabilities of malignant classes:
    - MEL (melanoma): weight 1.5
    - BCC (basal cell carcinoma): weight 1.3
    - SCC (squamous cell carcinoma): weight 1
    The function will search for both lowercase and uppercase keys and abbreviations.

    Parameters
    ----------
    probs : dict[str, float]
        A dictionary mapping class labels to their predicted probabilities.

    Returns
    -------
    str
        The risk factor as "Low", "Medium", or "High".
    """
    weights = {
        "mel": 1.5,
        "melanoma": 1.5,
        "bcc": 1.3,
        "basal cell carcinoma": 1.3,
        "scc": 1.0,
        "squamous cell carcinoma": 1.0,
    }
    risk_score = sum(prob * weights[label.lower()] for label, prob in probs.items() if label.lower() in weights)
    if risk_score < 0.30:
        return "Low"
    elif risk_score < 0.6:
        return "Medium"
    else:
        return "High"