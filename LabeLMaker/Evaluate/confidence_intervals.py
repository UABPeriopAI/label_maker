"""
The code defines functions to compute performance metric values and their bootstrap-based confidence
intervals for classification tasks.

:param y_true: `y_true` refers to the true labels in a classification task. It is an array-like
object containing the actual class labels for the data points. In the provided example,
`y_true_example` is a list of true labels for a set of observations, where each label corresponds to
a specific class
:param y_pred: The `y_pred` parameter represents the predicted labels for your classification task.
It should be an array-like object containing the predicted labels for each corresponding sample in
`y_true`. In the provided example usage, `y_pred_example` is a list of predicted labels for the
dummy data samples
:param score_func: The `score_func` parameter in the `bootstrap_metric` function is a callable
function that computes a specific metric (e.g., accuracy, precision, recall, F1 score) based on the
true labels (`y_true`) and predicted labels (`y_pred`). It allows you to pass different scoring
functions
:param n_bootstraps: The `n_bootstraps` parameter in the provided code refers to the number of
bootstrap iterations to perform when estimating the confidence intervals for the performance
metrics. It determines how many times the bootstrap resampling process will be repeated to calculate
the mean score and confidence intervals for each metric, defaults to 1000 (optional)
:param alpha: The `alpha` parameter in the provided code represents the significance level used to
calculate the confidence intervals. In statistical hypothesis testing and confidence interval
construction, the significance level is the probability of rejecting the null hypothesis when it is
true
:return: The `compute_bootstrap_confidence_intervals` function returns a dictionary containing
performance metric values (Accuracy, Precision, Recall, and F1 Score) along with their corresponding
bootstrap mean and 95% confidence intervals. Each metric in the dictionary has the following
structure:
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def bootstrap_metric(y_true, y_pred, score_func, n_bootstraps=1000, alpha=0.05, **kwargs):
    """
    Computes the bootstrap distribution for a given metric.

    Parameters:
      y_true (array-like): True labels.
      y_pred (array-like): Predicted labels.
      score_func (callable): Function to compute the metric.
      n_bootstraps (int): Number of bootstrap samples.
      alpha (float): Significance level (for a 95% CI, alpha=0.05).
      **kwargs: Additional arguments to pass to score_func.

    Returns:
      mean_score (float): Mean score from bootstrap samples.
      ci (tuple): (lower bound, upper bound) as the confidence interval.
    """
    scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    for i in range(n_bootstraps):
        # sample indices with replacement
        indices = np.random.choice(n, n, replace=True)
        sample_y_true = y_true[indices]
        sample_y_pred = y_pred[indices]
        score = score_func(sample_y_true, sample_y_pred, **kwargs)
        scores.append(score)

    # Compute percentile bounds for the desired confidence level
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean_score = np.mean(scores)
    return mean_score, (lower, upper)


def compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=1000, alpha=0.05):
    """
    Computes performance metric values (accuracy, macro precision, recall, and F1) along
    with their bootstrap-based 95% confidence intervals.

    Parameters:
      y_true (array-like): True labels.
      y_pred (array-like): Predicted labels.
      n_bootstraps (int): Number of bootstrap iterations.
      alpha (float): Significance level.

    Returns:
      results (dict): Dictionary mapping each metric to its value, bootstrap mean, and 95% CI.
    """
    results = {}

    # Use macro-average for multiclass metrics.
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0),
        "Recall": lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0),
        "F1 Score": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    }

    # Compute values on full dataset and bootstrap estimates
    for metric_name, func in metrics.items():
        full_value = func(y_true, y_pred)
        bs_mean, ci = bootstrap_metric(y_true, y_pred, func, n_bootstraps=n_bootstraps, alpha=alpha)
        results[metric_name] = {"Value": full_value, "Bootstrap Mean": bs_mean, "95% CI": ci}
    return results


if __name__ == "__main__":
    # Example usage for bootstrapping metrics on dummy data:
    y_true_example = ["cat", "dog", "dog", "cat", "bird", "dog", "bird", "cat", "cat", "bird"]
    y_pred_example = ["cat", "cat", "dog", "cat", "bird", "dog", "bird", "dog", "cat", "bird"]
    results = compute_bootstrap_confidence_intervals(
        y_true_example, y_pred_example, n_bootstraps=1000, alpha=0.05
    )
    for metric, values in results.items():
        print(f"{metric}:")
        print(f"  Value           : {values['Value']:.4f}")
        print(f"  Bootstrap Mean  : {values['Bootstrap Mean']:.4f}")
        print(f"  95% CI          : ({values['95% CI'][0]:.4f}, {values['95% CI'][1]:.4f})\n")
