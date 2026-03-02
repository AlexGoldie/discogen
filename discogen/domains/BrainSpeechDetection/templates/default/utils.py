import numpy as np
from sklearn.metrics import precision_recall_curve


def _vectorized_macro_f1_from_thresholds(scores: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Compute macro-F1 for multiple thresholds at once using vectorized operations.

    Args:
        scores: prediction scores, shape (n,)
        y: binary labels, shape (n,)
        thresholds: candidate thresholds, shape (k,)

    Returns:
        macro_f1_scores: macro-F1 for each threshold, shape (k,)
    """
    y = y.astype(int).reshape(-1, 1)
    scores = scores.reshape(-1, 1)
    thresholds = thresholds.reshape(1, -1)
    preds = (scores >= thresholds).astype(int)

    tp = np.sum((preds == 1) & (y == 1), axis=0)
    fp = np.sum((preds == 1) & (y == 0), axis=0)
    tn = np.sum((preds == 0) & (y == 0), axis=0)
    fn = np.sum((preds == 0) & (y == 1), axis=0)

    f1_pos = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    f1_neg = (2 * tn) / (2 * tn + fn + fp + 1e-8)

    return 0.5 * (f1_pos + f1_neg)


def compute_macro_f1_score(logits, y):
    """Find threshold maximizing macro-F1 and return the best macro-F1.

    Uses vectorized computation across all candidate thresholds for speed.
    """
    scores = np.asarray(logits).astype(float)
    y = np.asarray(y).astype(int)

    _, __, pr_thresholds = precision_recall_curve(y, scores)

    macro_f1_scores = _vectorized_macro_f1_from_thresholds(scores, y, pr_thresholds)

    best_idx = np.argmax(macro_f1_scores)
    best_macro_f1 = float(macro_f1_scores[best_idx])
    best_thr = float(pr_thresholds[best_idx])

    return best_macro_f1, best_thr
