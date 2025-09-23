# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Score calibration module for uncertainty quantification confidence scores.

This module provides calibration methods to transform raw confidence scores
into better-calibrated probabilities using Platt Scaling and Isotonic Regression.
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional, List, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
from uqlm.utils.results import UQResult

Ignore_Columns = ["prompts", "responses", "sampled_responses", "raw_sampled_responses", "raw_responses", "logprobs"]


class ScoreCalibrator:
    """
    A class for calibrating confidence scores using Platt Scaling or Isotonic Regression.

    Confidence scores from uncertainty quantification methods may not be well-calibrated
    probabilities. This class provides methods to transform raw scores into calibrated
    probabilities that better reflect the true likelihood of correctness.

    Parameters
    ----------
    method : {'platt', 'isotonic'}, default='platt'
        The calibration method to use:
        - 'platt': Platt scaling using logistic regression
        - 'isotonic': Isotonic regression (non-parametric, monotonic)

    Attributes
    ----------
    method : str
        The calibration method used.
    calibrator_ : sklearn estimator
        The fitted calibration model.
    is_fitted_ : bool
        Whether the calibrator has been fitted.
    """

    def __init__(self, method: Literal["platt", "isotonic"] = "platt"):
        self.method = method
        self.calibrators = {}
        self.is_fitted_ = False

    def fit(self, uq_result: UQResult, correct_indicators: Union[List[bool], List[int], np.ndarray]) -> None:
        """
        Fit the calibration model using scores and binary correctness labels.

        Parameters
        ----------
        uq_result : UQResult
            The UQResult object to fit the calibrator on.
        correct_indicators : array-like of shape (n_samples,)
            Binary labels indicating correctness (True/False or 1/0).

        Returns
        -------
        self : ScoreCalibrator
            The fitted calibrator instance.
        """
        correct_indicators = np.array(correct_indicators, dtype=int)
        if not np.all(np.isin(correct_indicators, [0, 1])):
            raise ValueError("correct_indicators must be binary (True/False or 1/0)")

        for scorer in uq_result.data:
            if scorer not in Ignore_Columns:
                scores = np.array(uq_result.data[scorer])
                if len(scores) != len(correct_indicators):
                    raise ValueError("scores and correct_indicators must have the same length")

                if not np.all((scores >= 0) & (scores <= 1)):
                    raise ValueError("scores must be between 0 and 1 inclusive")

                if self.method == "platt":
                    # Reshape scores to 2D array for LogisticRegression
                    scores_2d = scores.reshape(-1, 1)
                    self.calibrators[scorer] = LogisticRegression()
                    self.calibrators[scorer].fit(scores_2d, correct_indicators)
                elif self.method == "isotonic":
                    self.calibrators[scorer] = IsotonicRegression(out_of_bounds="clip")
                    self.calibrators[scorer].fit(scores, correct_indicators)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted_ = True

    def transform(self, uq_result: UQResult) -> None:
        """
        Transform raw scores into calibrated probabilities.

        Parameters
        ----------
        uq_result : UQResult
            The UQResult object to transform.

        Returns
        -------
        None
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before transform")

        tmp = {}
        for scorer in uq_result.data:
            if scorer not in Ignore_Columns:
                scores = np.array(uq_result.data[scorer])
                if self.method == "platt":
                    # LogisticRegression needs 2D input and returns probabilities for class 1
                    scores_2d = scores.reshape(-1, 1)
                    tmp["calibrated_" + scorer] = self.calibrators[scorer].predict_proba(scores_2d)[:, 1]
                elif self.method == "isotonic":
                    # IsotonicRegression can handle 1D arrays
                    tmp["calibrated_" + scorer] = self.calibrators[scorer].predict(scores)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
        uq_result.data.update(tmp)

    def fit_transform(self, uq_result: UQResult, correct_indicators: Union[List[bool], List[int], np.ndarray]) -> None:
        """
        Fit the calibrator and transform the scores in one step.

        Parameters
        ----------
        uq_result : UQResult
            The UQResult object to fit and transform.
        correct_indicators : array-like of shape (n_samples,)
            Binary labels indicating correctness (True/False or 1/0).

        Returns
        -------
        None
        """
        self.fit(uq_result, correct_indicators)
        self.transform(uq_result)

    def evaluate_calibration(self, uq_result: UQResult, correct_indicators: Union[List[bool], List[int], np.ndarray], plot: bool = True, axes: Optional[tuple] = None) -> dict:
        """
        Evaluate the calibration quality of the scores.

        Parameters
        ----------
        uq_result : UQResult
            The UQResult object to evaluate.
        correct_indicators : array-like of shape (n_samples,)
            Binary labels indicating correctness (True/False or 1/0).
        plot : bool, default=True
            Whether to plot the reliability diagram.
        axes : tuple of matplotlib.axes.Axes, optional
            Tuple of (reliability_ax, distribution_ax) for plotting.
            If None and plot=True, creates new figure.

        Returns
        -------
        metrics : dict
            Dictionary containing calibration metrics for each scorer.
        """
        if len(uq_result.data["responses"]) != len(correct_indicators):
            raise ValueError("uq_result.data and correct_indicators must have the same length")

        metrics = {}
        for scorer in uq_result.data:
            if scorer not in Ignore_Columns:
                metrics[scorer] = evaluate_calibration(uq_result.data[scorer], correct_indicators, plot=plot, axes=axes, title=scorer)
        return pd.DataFrame(metrics).T


def evaluate_calibration(scores: Union[List[float], np.ndarray], correct_indicators: "Union[List[int], np.ndarray]", n_bins: int = 10, plot: bool = True, axes: "Union[tuple, None]" = None, title: str = None) -> dict:
    """
    Evaluate the calibration quality of scores.

    Parameters
    ----------
    scores : np.ndarray of shape (n_samples,)
        Confidence scores (raw or calibrated).
    correct_indicators : np.ndarray of shape (n_samples,)
        Binary indicators (0/1) of whether each response was correct.
    n_bins : int, default=10
        Number of bins for reliability diagram.
    plot : bool, default=True
        Whether to plot the reliability diagram.
    axes : tuple of matplotlib.axes.Axes, optional
        Tuple of (reliability_ax, distribution_ax) for plotting.
        If None and plot=True, creates new figure.
    title : str, optional
        Title of the plot.
    Returns
    -------
    metrics : dict
        Dictionary containing calibration metrics:
        - 'average_confidence': Mean of confidence scores
        - 'average_accuracy': Mean of correct indicators
        - 'calibration_gap': Absolute difference between average confidence and accuracy
        - 'brier_score': Brier score (lower is better)
        - 'log_loss': Log loss (lower is better)
        - 'ece': Expected Calibration Error
        - 'mce': Maximum Calibration Error
    """
    scores = np.array(scores)
    correct_indicators = np.array(correct_indicators)
    avg_confidence = scores.mean()
    avg_accuracy = correct_indicators.mean()
    calibration_gap = abs(avg_confidence - avg_accuracy)

    # Store metrics for table display
    metrics = {"average_confidence": avg_confidence, "average_accuracy": avg_accuracy, "calibration_gap": calibration_gap}

    # Calculate Brier score and log loss
    brier = brier_score_loss(correct_indicators, scores)

    # Handle edge case where all labels are the same (log_loss requires at least 2 classes)
    if len(np.unique(correct_indicators)) == 1:
        # For single class, log loss is not well-defined, so we calculate it manually
        # Log loss = -1/N * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
        eps = 1e-15  # Small epsilon to avoid log(0)
        scores_clipped = np.clip(scores, eps, 1 - eps)
        if correct_indicators[0] == 1:  # All correct
            logloss = -np.mean(np.log(scores_clipped))
        else:  # All incorrect
            logloss = -np.mean(np.log(1 - scores_clipped))
    else:
        logloss = log_loss(correct_indicators, scores)

    # Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)

    # Create bins enclosing the range [0, 1] of confidence scores
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_boundaries[0] = 0 - np.finfo(float).eps  # Ensure scores of exactly 0 are included
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    mce = 0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (scores > bin_lower) & (scores <= bin_upper)
        prob_in_bin = in_bin.mean()

        if prob_in_bin > 0:
            accuracy_in_bin = correct_indicators[in_bin].mean()
            avg_confidence_in_bin = scores[in_bin].mean()

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)

    metrics.update({"brier_score": brier, "log_loss": logloss, "ece": ece, "mce": mce})

    if plot:
        _plot_reliability_diagram(bin_boundaries, bin_counts, bin_accuracies, axes=axes, title=title)

    return metrics


def _plot_reliability_diagram(bin_boundaries: np.ndarray, bin_counts: list, bin_accuracies: list, axes: tuple = None, title: str = None):
    """Plot reliability diagram for calibration assessment."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if title is not None:
            fig.suptitle(f"Plots for {title}")
        show_plot = True
    else:
        show_plot = False

    ax1, ax2 = axes

    n_bins = len(bin_boundaries) - 1

    # Create bin boundary labels
    bin_labels = [f"({bin_boundaries[i]:.1f}, {bin_boundaries[i + 1]:.1f}]" for i in range(n_bins)]

    # Calculate bin midpoints for perfect calibration line
    bin_midpoints = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)]

    # Reliability diagram
    # Perfect calibration line: where confidence = accuracy for each bin
    ax1.plot(range(n_bins), bin_midpoints, "k--", label="Perfect calibration")
    ax1.bar(range(n_bins), bin_accuracies, alpha=0.7, label="Actual accuracy", width=0.8)
    ax1.set_xlabel("Confidence bin")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Reliability Diagram")
    ax1.set_xticks(range(n_bins))
    ax1.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Score distribution
    ax2.bar(range(n_bins), bin_counts, alpha=0.7)
    ax2.set_xlabel("Confidence bin")
    ax2.set_ylabel("Number of samples")
    ax2.set_title("Score Distribution")
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    if show_plot:
        plt.tight_layout()
        plt.show()


# def fit_and_evaluate_calibrators(scores: Union[List[float], np.ndarray], correct_labels: Union[List[bool], List[int], np.ndarray], test_size: float = 0.2, random_state: Optional[int] = None, methods: Optional[List[str]] = None) -> tuple[dict, pd.DataFrame, dict]:
#     """
#     Fit and evaluate multiple calibration methods on the same data.

#     Parameters
#     ----------
#     scores : array-like
#         Training confidence scores.
#     correct_labels : array-like
#         Training binary correctness labels (True/False or 1/0).
#     test_size : float, default=0.2
#         Proportion of data to use for testing.
#     random_state : int, optional
#         Random state for train-test split.
#     methods : list of str, optional
#         List of calibration methods to compare. Default is ['platt', 'isotonic'].

#     Returns
#     -------
#     calibrators : dict
#         Dictionary containing fitted ScoreCalibrator instances for each method.
#         Keys are method names ('platt', 'isotonic'), values are fitted calibrators.

#     metrics_df : pd.DataFrame
#         DataFrame comparing calibration metrics across methods with columns:

#         - 'brier_score' : float
#             Brier score (lower is better) - measures accuracy of probabilistic predictions
#         - 'log_loss' : float
#             Logarithmic loss (lower is better) - penalizes confident wrong predictions
#         - 'ece' : float
#             Expected Calibration Error - average difference between confidence and accuracy
#         - 'mce' : float
#             Maximum Calibration Error - worst-case calibration error across bins

#         Index contains method names: 'platt', 'isotonic'.

#     figures : dict
#         Dictionary containing matplotlib figures:
#         - 'transformation_comparison' : Figure showing original vs transformed scores
#         - '{method}_calibration' : Figure showing calibration plots for each method

#     See Also
#     --------
#     ScoreEvaluator.evaluate_calibration : Detailed metric definitions and computation
#     """
#     if methods is None:
#         methods = ["platt", "isotonic"]
#     results = []
#     calibrated_scores_dict = {}
#     figures = {}
#     calibrators = {}

#     # Convert to numpy arrays
#     scores = np.array(scores)
#     correct_labels = np.array(correct_labels, dtype=int)

#     # Train-test split
#     train_scores, test_scores, train_labels, test_labels = train_test_split(scores, correct_labels, test_size=test_size, random_state=random_state, stratify=correct_labels)

#     # Create evaluator instance
#     transformation_data = {}
#     for method in methods:
#         calibrator = ScoreCalibrator(method=method)
#         calibrator.fit(train_scores, train_labels)
#         calibrators[method] = calibrator
#         eval_scores = calibrator.transform(test_scores)
#         calibrated_scores_dict[method] = eval_scores
#         transformation_data[method] = eval_scores

#         metrics = evaluate_calibration(eval_scores, test_labels, plot=False, axes=None)
#         metrics["method"] = method
#         results.append(metrics)

#     # Create figures
#     # 1. Transformation comparison
#     fig, axes = plt.subplots(1, 1, figsize=(3.5, 3.5))
#     _plot_original_vs_transformed(test_scores, transformation_data, ax=axes)
#     plt.close(fig)
#     figures["transformation_comparison"] = fig

#     # 2. Calibration plots for each method
#     for method in methods:
#         eval_scores = calibrated_scores_dict[method]
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#         fig.suptitle(f"Calibration Plots for {method} Scores")
#         evaluate_calibration(eval_scores, test_labels, plot=True, axes=axes)
#         plt.close(fig)
#         figures[f"{method}_calibration"] = fig

#     return calibrators, pd.DataFrame(results).set_index("method"), figures


# def _plot_original_vs_transformed(original_scores: np.ndarray, transformed_data: Union[np.ndarray, dict], ax: plt.Axes = None, **kwargs):
#     """
#     Plot original vs transformed scores with a perfect calibration reference line.

#     Parameters
#     ----------
#     original_scores : np.ndarray
#         The original (uncalibrated) scores.
#     transformed_data : dict
#         multiple transformations with method names as keys.
#     ax : matplotlib.axes.Axes, optional
#         Axes to plot on. If None, creates new figure.
#     kwargs : dict
#         Optional keyword arguments for customization.
#     """
#     title = kwargs.get("title", "Score Transformation Comparison")
#     alpha = kwargs.get("alpha", 0.5)
#     s = kwargs.get("s", 10)

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6))

#     # Handle both single transformation and multiple transformations
#     colors = {"platt": "blue", "isotonic": "orange"}
#     for method, transformed_scores in transformed_data.items():
#         color = colors.get(method, None)
#         ax.scatter(original_scores, transformed_scores, alpha=alpha, s=s, label=f"{method.title()} Calibration", color=color)

#     ax.set_xlabel("Original Scores")
#     ax.set_ylabel("Transformed Scores")
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True, alpha=0.3)