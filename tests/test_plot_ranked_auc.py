import pytest
from unittest.mock import patch
from uqlm.utils.plots import plot_ranked_auc
from uqlm.utils.results import UQResult


@pytest.fixture
def string_response_uq_result():
    return UQResult(result={"data": {"responses": ["The Eiffel Tower is in Berlin.", "Water boils at 100°C.", "The moon is made of cheese.", "Paris is the capital of France."], "semantic_negentropy": [0.9, 0.1, 0.8, 0.2], "normalized_probability": [0.85, 0.15, 0.75, 0.25], "ensemble_scores": [0.88, 0.12, 0.78, 0.22]}, "metadata": {}})


# Input validation tests
def test_missing_correct_indicators(string_response_uq_result):
    with pytest.raises(ValueError, match="correct_indicators must be provided"):
        plot_ranked_auc(string_response_uq_result, None)


def test_length_mismatch(string_response_uq_result):
    with pytest.raises(ValueError, match="correct_responses must be the same length"):
        plot_ranked_auc(string_response_uq_result, [True, False])


def test_invalid_metric_type(string_response_uq_result):
    with pytest.raises(ValueError, match="metric_type must be one of"):
        plot_ranked_auc(string_response_uq_result, [False, True, False, True], metric_type="invalid")


# Plot rendering tests
@patch("matplotlib.pyplot.show")
def test_plot_auroc_only(mock_show, string_response_uq_result):
    plot_ranked_auc(string_response_uq_result, [False, True, False, True], metric_type="auroc")
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_plot_auprc_only(mock_show, string_response_uq_result):
    plot_ranked_auc(string_response_uq_result, [False, True, False, True], metric_type="auprc")
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_plot_both_metrics(mock_show, string_response_uq_result):
    plot_ranked_auc(string_response_uq_result, [False, True, False, True], metric_type="both")
    mock_show.assert_called_once()


# Scorer categorization and naming
@patch("matplotlib.pyplot.show")
def test_scorer_categorization_and_naming(mock_show):
    uq_result = UQResult(result={"data": {"responses": ["A", "B", "C", "D"], "semantic_negentropy": [0.9, 0.1, 0.8, 0.2], "normalized_probability": [0.85, 0.15, 0.75, 0.25], "ensemble_scores": [0.88, 0.12, 0.78, 0.22]}, "metadata": {}})
    correct = [False, True, False, True]

    plot_ranked_auc(uq_result, correct, scorers_names=["semantic_negentropy", "normalized_probability", "ensemble_scores"], metric_type="auroc")
    mock_show.assert_called_once()


# Default scorer selection (scorers_names=None)
@patch("matplotlib.pyplot.show")
def test_default_scorer_selection(mock_show):
    uq_result = UQResult(result={"data": {"responses": ["A", "B", "C", "D"], "semantic_negentropy": [0.9, 0.1, 0.8, 0.2], "normalized_probability": [0.85, 0.15, 0.75, 0.25], "ensemble_scores": [0.88, 0.12, 0.78, 0.22], "prompts": ["Q1", "Q2", "Q3", "Q4"]}, "metadata": {}})
    correct = [False, True, False, True]

    plot_ranked_auc(uq_result, correct, scorers_names=None, metric_type="auroc")
    mock_show.assert_called_once()
