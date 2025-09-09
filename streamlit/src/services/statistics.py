import numpy as np


def compute_malignancy_index(probs: dict[str, float]) -> float:
    """Compute a malignancy index from class probabilities.

    The malignancy index is defined as the sum of probabilities of malignant classes.
    The malignant classes are:
    - MEL (melanoma)
    - BCC (basal cell carcinoma)
    - SCC (squamous cell carcinoma)
    The function will search for both lowercase and uppercase keys and abbreviations.

    Parameters
    ----------
    probs : dict[str, float]
        A dictionary mapping class labels to their predicted probabilities.

    Returns
    -------
    float
        The computed malignancy index in the range [0, 1].
    """
    malignant_keys = {"mel", "melanoma", "bcc", "basal cell carcinoma", "scc", "squamous cell carcinoma"}
    malignancy_index = sum(prob for label, prob in probs.items() if label.lower() in malignant_keys)
    return float(np.clip(malignancy_index, 0.0, 1.0))


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