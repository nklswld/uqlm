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

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from uqlm.utils.score_calibrator import ScoreCalibrator, _plot_original_vs_transformed, evaluate_calibration, _plot_reliability_diagram
from uqlm.utils.results import UQResult


class TestScoreCalibrator:
    """Test suite for ScoreCalibrator class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample calibration data for testing."""
        np.random.seed(42)
        n_samples = 100
        scores = np.random.beta(2, 2, n_samples)  # Scores between 0 and 1
        uq_result = UQResult(result={"data": {"judge_1": scores}})
        # Create labels with some correlation to scores
        correct_labels = np.random.binomial(1, scores * 0.7 + 0.15, n_samples)
        return uq_result, correct_labels

    @pytest.fixture
    def perfect_calibration_data(self):
        """Generate perfectly calibrated data for testing."""
        np.random.seed(123)
        n_samples = 100
        scores = np.random.uniform(0, 1, n_samples)
        uq_result = UQResult(result={"data": {"judge_1": scores}})
        correct_labels = np.random.binomial(1, scores, n_samples)
        return uq_result, correct_labels
        return scores, correct_labels

    def test_init_default(self):
        """Test ScoreCalibrator initialization with default parameters."""
        calibrator = ScoreCalibrator()
        assert calibrator.method == "platt"
        assert calibrator.calibrators == {}
        assert calibrator.is_fitted_ is False

    def test_init_invalid_method(self):
        """Test ScoreCalibrator initialization with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            calibrator = ScoreCalibrator(method="invalid")
            calibrator.fit(uq_result=UQResult(result={"data": {"judge_1": [0.1, 0.9]}}), correct_indicators=[0, 1])

    def test_fit_and_transform_platt(self, sample_data):
        """Test fitting and transforming with Platt scaling method."""
        uq_result, labels = sample_data
        calibrator = ScoreCalibrator(method="platt")

        # Test fit
        calibrator.fit(uq_result, labels)
        assert calibrator.is_fitted_ is True
        assert calibrator.calibrators["judge_1"] is not None
        assert hasattr(calibrator.calibrators["judge_1"], "predict_proba")

        # Test transform
        calibrator.transform(uq_result)
        assert len(uq_result.data["calibrated_judge_1"]) == len(uq_result.data["judge_1"])
        assert np.all((uq_result.data["calibrated_judge_1"] >= 0) & (uq_result.data["calibrated_judge_1"] <= 1))

    def test_fit_and_transform_isotonic(self, sample_data):
        """Test fitting and transforming with isotonic regression method."""
        uq_result, labels = sample_data
        calibrator = ScoreCalibrator(method="isotonic")

        # Test fit
        calibrator.fit(uq_result, labels)
        assert calibrator.is_fitted_ is True
        assert calibrator.calibrators["judge_1"] is not None
        assert hasattr(calibrator.calibrators["judge_1"], "predict")

        # Test transform
        calibrator.transform(uq_result)
        assert len(uq_result.data["calibrated_judge_1"]) == len(uq_result.data["judge_1"])
        assert np.all((uq_result.data["calibrated_judge_1"] >= 0) & (uq_result.data["calibrated_judge_1"] <= 1))

    def test_fit_mismatched_lengths(self):
        """Test fit with mismatched input lengths."""
        calibrator = ScoreCalibrator()
        uq_result = UQResult(result={"data": {"judge_1": [0.1, 0.5, 0.9]}})
        labels = [0, 1]  # Different length

        with pytest.raises(ValueError, match="scores and correct_indicators must have the same length"):
            calibrator.fit(uq_result, labels)

    def test_fit_invalid_labels(self):
        """Test fit with invalid label values."""
        calibrator = ScoreCalibrator()
        uq_result = UQResult(result={"data": {"judge_1": [0.1, 0.5, 0.9]}})
        labels = [0, 1, 2]  # Invalid label value

        with pytest.raises(ValueError, match=r"correct_indicators must be binary \(True/False or 1/0\)"):
            calibrator.fit(uq_result, labels)

    def test_fit_invalid_scores(self):
        """Test fit with invalid score values."""
        calibrator = ScoreCalibrator()
        uq_result = UQResult(result={"data": {"judge_1": [0.1, 0.5, 1.5]}})  # Score > 1
        labels = [0, 1, 1]

        with pytest.raises(ValueError, match="scores must be between 0 and 1 inclusive"):
            calibrator.fit(uq_result, labels)

    def test_fit_negative_scores(self):
        """Test fit with negative score values."""
        calibrator = ScoreCalibrator()
        uq_result = UQResult(result={"data": {"judge_1": [-0.1, 0.5, 0.9]}})  # Negative score
        labels = [0, 1, 1]

        with pytest.raises(ValueError, match="scores must be between 0 and 1 inclusive"):
            calibrator.fit(uq_result, labels)

    def test_transform_not_fitted(self):
        """Test transform before fitting."""
        calibrator = ScoreCalibrator()
        uq_result = UQResult(result={"data": {"judge_1": [0.1, 0.5, 0.9]}})

        with pytest.raises(ValueError, match="Calibrator must be fitted before transform"):
            calibrator.transform(uq_result)

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        uq_result, labels = sample_data
        calibrator = ScoreCalibrator()

        calibrator.fit_transform(uq_result, labels)

        assert calibrator.is_fitted_ is True
        assert len(uq_result.data["calibrated_judge_1"]) == len(uq_result.data["judge_1"])
        assert np.all((uq_result.data["calibrated_judge_1"] >= 0) & (uq_result.data["calibrated_judge_1"] <= 1))

    def test_boolean_labels(self, sample_data):
        """Test with boolean labels."""
        uq_result, labels = sample_data
        bool_labels = labels.astype(bool)
        calibrator = ScoreCalibrator()

        calibrator.fit(uq_result, bool_labels)
        calibrator.transform(uq_result)

        assert len(uq_result.data["calibrated_judge_1"]) == len(uq_result.data["judge_1"])
        assert calibrator.is_fitted_ is True

    def test_edge_case_single_sample(self):
        """Test with single sample - should raise error for insufficient data."""
        calibrator = ScoreCalibrator()
        uq_result = UQResult(result={"data": {"judge_1": [0.5]}})
        labels = [1]

        # Single sample with only one class should raise an error
        with pytest.raises(ValueError):
            calibrator.fit(uq_result=uq_result, correct_indicators=labels)

    def test_minimal_valid_dataset(self):
        """Test with minimal valid dataset (2 samples, 2 classes)."""
        calibrator = ScoreCalibrator()
        uq_result = UQResult(result={"data": {"judge_1": [0.3, 0.7]}})
        labels = [0, 1]

        calibrator.fit(uq_result=uq_result, correct_indicators=labels)
        calibrator.transform(uq_result=uq_result)

        assert len(uq_result.data["calibrated_judge_1"]) == 2
        assert np.all((uq_result.data["calibrated_judge_1"] >= 0) & (uq_result.data["calibrated_judge_1"] <= 1))


# class TestFitAndEvaluateCalibrators:
#     """Test suite for fit_and_evaluate_calibrators function."""

#     @pytest.fixture
#     def sample_data(self):
#         """Generate sample calibration data for testing."""
#         np.random.seed(42)
#         n_samples = 200
#         scores = np.random.beta(2, 2, n_samples)
#         correct_labels = np.random.binomial(1, scores * 0.7 + 0.15, n_samples)
#         return UQResult(result={"data": {"judge_1": scores}}), correct_labels

#     def test_default_parameters(self, sample_data):
#         """Test with default parameters."""
#         uq_result, labels = sample_data

#         calibrators, metrics_df, figures = fit_and_evaluate_calibrators(uq_result, labels)

#         # Check calibrators
#         assert isinstance(calibrators, dict)
#         assert "platt" in calibrators
#         assert "isotonic" in calibrators
#         assert all(cal.is_fitted_ for cal in calibrators.values())

#         # Check metrics
#         assert isinstance(metrics_df, pd.DataFrame)
#         assert len(metrics_df) == 2  # platt and isotonic
#         assert "brier_score" in metrics_df.columns
#         assert "ece" in metrics_df.columns
#         assert "mce" in metrics_df.columns

#         # Check figures
#         assert isinstance(figures, dict)
#         assert "transformation_comparison" in figures
#         assert "platt_calibration" in figures
#         assert "isotonic_calibration" in figures

#     def test_custom_random_state(self, sample_data):
#         """Test with custom random state."""
#         uq_result, labels = sample_data

#         # Run twice with same random state
#         result1 = fit_and_evaluate_calibrators(uq_result, labels, random_state=42)
#         result2 = fit_and_evaluate_calibrators(uq_result, labels, random_state=42)

#         # Results should be identical
#         pd.testing.assert_frame_equal(result1[1], result2[1])

#     def test_single_method(self, sample_data):
#         """Test with single calibration method."""
#         uq_result, labels = sample_data

#         calibrators, metrics_df, figures = fit_and_evaluate_calibrators(uq_result, labels, methods=["platt"])

#         assert len(calibrators) == 1
#         assert "platt" in calibrators
#         assert len(metrics_df) == 1
#         assert "platt_calibration" in figures
#         assert "isotonic_calibration" not in figures

#     def test_invalid_method(self, sample_data):
#         """Test with invalid method."""
#         uq_result, labels = sample_data

#         with pytest.raises(ValueError):
#             fit_and_evaluate_calibrators(uq_result, labels, methods=["invalid"])

#     def test_small_dataset(self):
#         """Test with very small dataset."""
#         uq_result = UQResult(result={"data": {"judge_1": [0.1, 0.9, 0.3, 0.7]}})
#         labels = [0, 1, 0, 1]

#         calibrators, metrics_df, _ = fit_and_evaluate_calibrators(uq_result, labels, test_size=0.5)

#         assert len(calibrators) == 2
#         assert len(metrics_df) == 2


class TestPlotOriginalVsTransformed:
    """Test suite for _plot_original_vs_transformed function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for plotting tests."""
        np.random.seed(42)
        original = np.random.uniform(0, 1, 50)
        transformed_platt = original * 0.8 + 0.1  # Simple transformation
        transformed_isotonic = np.clip(original * 1.2 - 0.1, 0, 1)
        return original, {"platt": transformed_platt, "isotonic": transformed_isotonic}

    def test_plot_with_default_params(self, sample_data):
        """Test plotting with default parameters."""
        original, transformed = sample_data

        fig, ax = plt.subplots()
        _plot_original_vs_transformed(original, transformed, ax=ax)

        assert len(ax.collections) == 2  # Two scatter plots
        assert ax.get_xlabel() == "Original Scores"
        assert ax.get_ylabel() == "Transformed Scores"
        assert ax.legend_ is not None

        plt.close(fig)

    def test_plot_with_custom_kwargs(self, sample_data):
        """Test plotting with custom keyword arguments."""
        original, transformed = sample_data

        fig, ax = plt.subplots()
        _plot_original_vs_transformed(original, transformed, ax=ax, title="Custom Title", alpha=0.8, s=20)

        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_plot_single_method(self, sample_data):
        """Test plotting with single transformation method."""
        original, transformed = sample_data

        fig, ax = plt.subplots()
        single_transformed = {"platt": transformed["platt"]}
        _plot_original_vs_transformed(original, single_transformed, ax=ax)

        assert len(ax.collections) == 1  # One scatter plot
        plt.close(fig)


class TestIntegration:
    """Integration tests for the calibration module."""

    # def test_end_to_end_workflow(self):
    #     """Test complete calibration workflow."""
    #     # Generate realistic overconfident data
    #     np.random.seed(42)
    #     n_samples = 500
    #     raw_scores = np.random.beta(0.5, 0.5, n_samples)
    #     raw_scores = 0.3 + 0.7 * raw_scores  # Shift to be overconfident
    #     correct_labels = np.random.binomial(1, 0.2 + 0.4 * raw_scores, n_samples)

    #     # Fit calibrators
    #     calibrators, metrics_df, figures = fit_and_evaluate_calibrators(raw_scores, correct_labels, test_size=0.4, random_state=42)

    #     # Check that calibration improves metrics
    #     # Both methods should have reasonable performance
    #     assert len(calibrators) == 2
    #     assert all(0 <= row["brier_score"] <= 1 for _, row in metrics_df.iterrows())
    #     assert all(0 <= row["ece"] <= 1 for _, row in metrics_df.iterrows())

    #     # Test individual calibrator usage
    #     platt_calibrator = calibrators["platt"]
    #     new_scores = [0.1, 0.5, 0.9]
    #     calibrated = platt_calibrator.transform(new_scores)

    #     assert len(calibrated) == 3
    #     assert np.all((calibrated >= 0) & (calibrated <= 1))

    # def test_perfect_calibration_data(self):
    #     """Test with perfectly calibrated data."""
    #     np.random.seed(123)
    #     n_samples = 300
    #     scores = np.random.uniform(0, 1, n_samples)
    #     correct_labels = np.random.binomial(1, scores, n_samples)

    #     calibrators, metrics_df, figures = fit_and_evaluate_calibrators(scores, correct_labels, random_state=42)

    #     # With perfect calibration, ECE should be relatively low
    #     assert all(row["ece"] < 0.2 for _, row in metrics_df.iterrows())

    # def test_extreme_overconfidence(self):
    #     """Test with extremely overconfident scores."""
    #     n_samples = 200
    #     scores = np.full(n_samples, 0.95)  # Very high confidence
    #     correct_labels = np.random.binomial(1, 0.3, n_samples)  # Low accuracy

    #     calibrators, _, _ = fit_and_evaluate_calibrators(scores, correct_labels, random_state=42)

    #     # Calibration should help reduce overconfidence
    #     platt_calibrator = calibrators["platt"]
    #     calibrated_scores = platt_calibrator.transform(scores)

    #     # Calibrated scores should be lower than original for this case
    #     assert np.mean(calibrated_scores) < np.mean(scores)

    def test_isotonic_monotonicity(self):
        """Test that isotonic regression produces monotonic transformation."""
        np.random.seed(42)
        scores = np.linspace(0, 1, 100)
        # Create labels with some noise but generally increasing with scores
        correct_labels = np.random.binomial(1, np.clip(scores + np.random.normal(0, 0.1, 100), 0, 1))

        calibrator = ScoreCalibrator(method="isotonic")
        calibrator.fit(uq_result=UQResult(result={"data": {"judge_1": scores}}), correct_indicators=correct_labels)

        # Test on sorted scores
        test_scores = np.linspace(0, 1, 50)
        uq_result = UQResult(result={"data": {"judge_1": test_scores}})
        calibrator.transform(uq_result)

        # Should be monotonic (non-decreasing)
        assert np.all(np.diff(uq_result.data["calibrated_judge_1"]) >= -1e-10)  # Allow for small numerical errors

class TestScoreEvaluator:
    """Test suite for ScoreEvaluator class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample calibration data for testing."""
        np.random.seed(42)
        n_samples = 100
        scores = np.random.beta(2, 2, n_samples)  # Scores between 0 and 1
        # Create labels with some correlation to scores
        correct_labels = np.random.binomial(1, scores * 0.7 + 0.15, n_samples)
        return scores, correct_labels

    @pytest.fixture
    def perfect_calibration_data(self):
        """Generate perfectly calibrated data for testing."""
        np.random.seed(123)
        n_samples = 1000  # Larger sample for more stable perfect calibration
        scores = np.random.uniform(0, 1, n_samples)
        correct_labels = np.random.binomial(1, scores, n_samples)
        return scores, correct_labels

    @pytest.fixture
    def overconfident_data(self):
        """Generate overconfident scores (high confidence, low accuracy)."""
        scores = np.array([0.9, 0.95, 0.8, 0.85, 0.9, 0.92, 0.88, 0.93, 0.87, 0.91])
        correct_labels = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])  # 30% accuracy
        return scores, correct_labels

    @pytest.fixture
    def underconfident_data(self):
        """Generate underconfident scores (low confidence, high accuracy)."""
        scores = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.12, 0.18, 0.22, 0.28, 0.16])
        correct_labels = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1])  # 90% accuracy
        return scores, correct_labels

    def test_evaluate_calibration_basic_metrics(self, sample_data):
        """Test basic calibration metrics calculation."""
        scores, correct_labels = sample_data

        metrics = evaluate_calibration(scores, correct_labels, plot=False)

        # Check that all expected metrics are present
        expected_keys = {"average_confidence", "average_accuracy", "calibration_gap", "brier_score", "log_loss", "ece", "mce"}
        assert set(metrics.keys()) == expected_keys

        # Check metric ranges
        assert 0 <= metrics["average_confidence"] <= 1
        assert 0 <= metrics["average_accuracy"] <= 1
        assert metrics["calibration_gap"] >= 0
        assert metrics["brier_score"] >= 0
        assert metrics["log_loss"] >= 0
        assert 0 <= metrics["ece"] <= 1
        assert 0 <= metrics["mce"] <= 1

    def test_evaluate_calibration_perfect_case(self, perfect_calibration_data):
        """Test calibration metrics with perfectly calibrated data."""
        scores, correct_labels = perfect_calibration_data

        metrics = evaluate_calibration(scores, correct_labels, plot=False)

        # Perfect calibration should have small calibration gap and ECE
        # Note: Due to randomness, we allow some tolerance
        assert metrics["calibration_gap"] < 0.05
        assert metrics["ece"] < 0.05

    def test_evaluate_calibration_overconfident(self, overconfident_data):
        """Test calibration metrics with overconfident scores."""
        scores, correct_labels = overconfident_data

        metrics = evaluate_calibration(scores, correct_labels, plot=False)

        # Overconfident data should have high calibration gap
        assert metrics["average_confidence"] > metrics["average_accuracy"]
        assert metrics["calibration_gap"] > 0.5  # Should be significantly overconfident

    def test_evaluate_calibration_underconfident(self, underconfident_data):
        """Test calibration metrics with underconfident scores."""
        scores, correct_labels = underconfident_data

        metrics = evaluate_calibration(scores, correct_labels, plot=False)

        # Underconfident data should have high calibration gap
        assert metrics["average_confidence"] < metrics["average_accuracy"]
        assert metrics["calibration_gap"] > 0.5  # Should be significantly underconfident

    def test_input_type_validation(self):
        """Test that function handles various input types correctly."""
        # Test with different numeric types: lists, tuples, arrays
        scores_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        correct_list = [0, 0, 1, 1, 1]
        scores_tuple = (0.1, 0.5, 0.9)
        correct_tuple = (0, 1, 1)
        scores_array = np.array([0.2, 0.6, 0.8])
        correct_array = np.array([0, 1, 1])

        # All should work without error
        for scores, correct_labels in [(scores_list, correct_list), (scores_tuple, correct_tuple), (scores_array, correct_array)]:
            metrics = evaluate_calibration(scores, correct_labels, plot=False)
            assert isinstance(metrics, dict)
            assert 0 <= metrics["average_confidence"] <= 1

    def test_single_class_scenarios(self):
        """Test edge cases with single class data and extreme values."""
        # All correct, high confidence
        scores = np.array([0.9, 0.95, 0.99])
        correct_labels = np.array([1, 1, 1])

        metrics = evaluate_calibration(scores, correct_labels, plot=False)

        assert metrics["average_accuracy"] == 1.0
        assert metrics["calibration_gap"] >= 0
        assert np.isfinite(metrics["log_loss"])
        assert metrics["log_loss"] > 0

        # All incorrect, low confidence
        scores = np.array([0.1, 0.05, 0.01])
        correct_labels = np.array([0, 0, 0])

        metrics = evaluate_calibration(scores, correct_labels, plot=False)

        assert metrics["average_accuracy"] == 0.0
        assert np.isfinite(metrics["log_loss"])
        assert metrics["log_loss"] > 0

    def test_evaluate_calibration_single_sample(self):
        """Test with single sample."""
        scores = np.array([0.7])
        correct_labels = np.array([1])

        metrics = evaluate_calibration(scores, correct_labels, plot=False)

        assert metrics["average_confidence"] == 0.7
        assert metrics["average_accuracy"] == 1.0
        assert abs(metrics["calibration_gap"] - 0.3) < 1e-10  # Use tolerance for floating point
        # Log loss should be finite for single sample case
        assert np.isfinite(metrics["log_loss"])

    def test_evaluate_calibration_boundary_scores(self):
        """Test with boundary confidence scores (0 and 1)."""
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        correct_labels = np.array([0, 1, 0, 1])

        metrics = evaluate_calibration(scores, correct_labels, plot=False)

        # Should handle boundary cases without error
        assert isinstance(metrics, dict)
        assert 0 <= metrics["ece"] <= 1

    def test_evaluate_calibration_different_bin_counts(self, sample_data):
        """Test with different numbers of bins."""
        scores, correct_labels = sample_data

        for n_bins in [5, 10, 20]:
            metrics = evaluate_calibration(scores, correct_labels, n_bins=n_bins, plot=False)

            # Should work with different bin counts
            assert isinstance(metrics, dict)
            assert 0 <= metrics["ece"] <= 1

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    def test_plotting_functionality(self, mock_show, mock_tight_layout, mock_subplots, sample_data):
        """Test plotting functionality with automatic and custom axes."""
        scores, correct_labels = sample_data

        # Test automatic plotting
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])

        _ = evaluate_calibration(scores, correct_labels, plot=True)

        # Verify plotting functions were called
        mock_subplots.assert_called_once_with(1, 2, figsize=(12, 5))
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()
        assert mock_ax1.plot.called
        assert mock_ax1.bar.called
        assert mock_ax2.bar.called

        # Reset mocks for custom axes test
        mock_show.reset_mock()
        mock_subplots.reset_mock()
        mock_tight_layout.reset_mock()

        # Test custom axes
        custom_ax1 = MagicMock()
        custom_ax2 = MagicMock()
        custom_axes = (custom_ax1, custom_ax2)

        _ = evaluate_calibration(scores, correct_labels, plot=True, axes=custom_axes)

        # When custom axes provided, show() should not be called
        mock_show.assert_not_called()
        mock_subplots.assert_not_called()
        mock_tight_layout.assert_not_called()

        # Verify custom axes were used
        assert custom_ax1.plot.called
        assert custom_ax1.bar.called
        assert custom_ax2.bar.called

    def test_ece_calculation_manual(self):
        """Test ECE calculation with manually verified data."""
        # Create simple data where we can manually calculate ECE
        scores = np.array([0.1, 0.1, 0.9, 0.9])  # Two bins: [0.0-0.5], [0.5-1.0]
        correct_labels = np.array([0, 1, 0, 1])  # 50% accuracy in each bin

        metrics = evaluate_calibration(scores, correct_labels, n_bins=2, plot=False)

        # Manual ECE calculation:
        # Bin 1 [0.0-0.5]: avg_conf=0.1, accuracy=0.5, weight=0.5, contrib=|0.1-0.5|*0.5=0.2
        # Bin 2 [0.5-1.0]: avg_conf=0.9, accuracy=0.5, weight=0.5, contrib=|0.9-0.5|*0.5=0.2
        # ECE = 0.2 + 0.2 = 0.4
        expected_ece = 0.4

        assert abs(metrics["ece"] - expected_ece) < 0.01

    def test_mce_calculation_manual(self):
        """Test MCE calculation with manually verified data."""
        # Create data where one bin has much worse calibration than others
        scores = np.array([0.1, 0.1, 0.9, 0.9])
        correct_labels = np.array([0, 0, 0, 0])  # All wrong

        metrics = evaluate_calibration(scores, correct_labels, n_bins=2, plot=False)

        # MCE should be the maximum calibration error across bins
        # Bin 2 [0.5-1.0]: |0.9 - 0.0| = 0.9 (worst bin)
        # Bin 1 [0.0-0.5]: |0.1 - 0.0| = 0.1
        expected_mce = 0.9

        assert abs(metrics["mce"] - expected_mce) < 0.01
        # Log loss should be finite even with all incorrect labels
        assert np.isfinite(metrics["log_loss"])

    def test_empty_bins_handling(self):
        """Test handling of empty bins in ECE/MCE calculation."""
        # Create scores that don't populate all bins
        scores = np.array([0.1, 0.2, 0.8, 0.9])  # No scores in middle bins
        correct_labels = np.array([0, 1, 0, 1])

        metrics = evaluate_calibration(scores, correct_labels, n_bins=10, plot=False)

        # Should handle empty bins without error
        assert isinstance(metrics["ece"], float)
        assert isinstance(metrics["mce"], float)
        assert 0 <= metrics["ece"] <= 1
        assert 0 <= metrics["mce"] <= 1

    def test_plot_reliability_diagram_methods(self):
        """Test the _plot_reliability_diagram method with and without axes."""
        bin_boundaries = np.array([0, 0.5, 1.0])
        bin_counts = [50, 50]
        bin_accuracies = [0.3, 0.7]

        # Test with provided axes
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        axes = (mock_ax1, mock_ax2)

        _plot_reliability_diagram(bin_boundaries, bin_counts, bin_accuracies, axes=axes)

        # Verify plotting methods were called
        mock_ax1.plot.assert_called()
        mock_ax1.bar.assert_called()
        mock_ax2.bar.assert_called()

        # Verify labels and titles were set
        mock_ax1.set_xlabel.assert_called()
        mock_ax1.set_ylabel.assert_called()
        mock_ax1.set_title.assert_called()
        mock_ax2.set_xlabel.assert_called()
        mock_ax2.set_ylabel.assert_called()
        mock_ax2.set_title.assert_called()

        # Test without axes (creates new figure)
        with patch("matplotlib.pyplot.subplots") as mock_subplots, patch("matplotlib.pyplot.tight_layout") as mock_tight_layout, patch("matplotlib.pyplot.show") as mock_show:
            mock_fig = MagicMock()
            mock_ax1_new = MagicMock()
            mock_ax2_new = MagicMock()
            mock_subplots.return_value = (mock_fig, [mock_ax1_new, mock_ax2_new])

            _plot_reliability_diagram(bin_boundaries, bin_counts, bin_accuracies, axes=None)

            # Verify new figure was created and shown
            mock_subplots.assert_called_once_with(1, 2, figsize=(12, 5))
            mock_tight_layout.assert_called_once()
            mock_show.assert_called_once()

    def test_edge_cases_and_precision(self):
        """Test numeric precision and edge cases with boundary values."""
        # Test with very small differences
        scores = np.array([0.5000001, 0.5000002, 0.4999999])
        correct_labels = np.array([1, 0, 1])

        metrics = evaluate_calibration(scores, correct_labels, plot=False)

        # Should handle small numeric differences without error
        assert isinstance(metrics["calibration_gap"], float)
        assert not np.isnan(metrics["calibration_gap"])

        # Test scores exactly at bin boundaries
        scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        correct_labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1])

        metrics = evaluate_calibration(scores, correct_labels, n_bins=10, plot=False)

        # Should handle boundary scores correctly
        assert isinstance(metrics["ece"], float)
        assert not np.isnan(metrics["ece"])
        assert not np.isinf(metrics["ece"])