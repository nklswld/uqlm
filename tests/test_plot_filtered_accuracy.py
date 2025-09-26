import pytest
from unittest.mock import patch
from uqlm.utils import plot_filtered_accuracy
from uqlm.utils.results import UQResult

# Dummy Method_Names and Ignore_Columns for testing
Method_Names = {"semantic_negentropy": "Semantic Negentropy", "normalized_probability": "Normalized Probability", "ensemble_scores": "Ensemble Scores"}
Ignore_Columns = ["responses"]


@pytest.fixture
def sample_uq_result():
    return UQResult(result={"data": {"responses": ["The Eiffel Tower is in Berlin.", "Water boils at 100°C.", "The moon is made of cheese.", "Paris is the capital of France."], "semantic_negentropy": [0.9, 0.1, 0.8, 0.2], "normalized_probability": [0.85, 0.15, 0.75, 0.25], "ensemble_scores": [0.88, 0.12, 0.78, 0.22]}, "metadata": {}})


def test_missing_correct_indicators_raises(sample_uq_result):
    with pytest.raises(ValueError, match="correct_indicators must be provided"):
        plot_filtered_accuracy(sample_uq_result, None)


def test_mismatched_length_raises(sample_uq_result):
    wrong_length = [1, 0]
    with pytest.raises(ValueError, match="correct_responses must be the same length"):
        plot_filtered_accuracy(sample_uq_result, wrong_length)


@patch("matplotlib.pyplot.show")
def test_plot_runs_successfully(mock_show, sample_uq_result):
    correct = [0, 1, 0, 1]
    plot_filtered_accuracy(sample_uq_result, correct)
    assert mock_show.called


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_plot_saves_image(mock_show, mock_savefig, sample_uq_result, tmp_path):
    correct = [0, 1, 0, 1]
    out_path = tmp_path / "accuracy_plot.png"
    plot_filtered_accuracy(sample_uq_result, correct, write_path=str(out_path))
    mock_savefig.assert_called_once_with(str(out_path), dpi=300)


@patch("matplotlib.pyplot.show")
def test_plot_with_specific_scorers(mock_show, sample_uq_result):
    correct = [0, 1, 0, 1]
    plot_filtered_accuracy(sample_uq_result, correct, scorers_names=["semantic_negentropy", "ensemble_scores"])
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_with_custom_title_and_font(mock_show, sample_uq_result):
    correct = [0, 1, 0, 1]
    plot_filtered_accuracy(sample_uq_result, correct, title="Custom Accuracy Plot", fontsize=12, fontname="Arial")
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_excludes_ignore_columns(mock_show):
    uq_result = UQResult(
        result={
            "data": {
                "responses": ["A", "B", "C", "D"],
                "semantic_negentropy": [0.9, 0.1, 0.8, 0.2],
                "normalized_probability": [0.85, 0.15, 0.75, 0.25],
                "ensemble_scores": [0.88, 0.12, 0.78, 0.22],
                "metadata": [0, 0, 0, 0],  # Should be ignored
            },
            "metadata": {},
        }
    )
    Ignore_Columns.append("metadata")
    correct = [0, 1, 0, 1]
    plot_filtered_accuracy(uq_result, correct, scorers_names=["semantic_negentropy", "normalized_probability", "ensemble_scores"])
    Ignore_Columns.remove("metadata")  # Clean up
    assert mock_show.called
