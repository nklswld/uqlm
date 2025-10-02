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
from unittest.mock import Mock, MagicMock
from uqlm.utils.nli import NLI


class TestNLIInitialization:
    """Test NLI class initialization with different model types."""

    def test_default_initialization(self):
        """Test that NLI initializes with default HuggingFace model."""
        nli = NLI()
        assert nli.is_hf_model is True
        assert nli.max_length == 2000
        assert nli.label_mapping == ["contradiction", "neutral", "entailment"]
        assert nli.tokenizer is not None
        assert nli.model is not None

    def test_custom_hf_model_initialization(self):
        """Test NLI initialization with custom HuggingFace model name."""
        nli = NLI(nli_model_name="microsoft/deberta-base-mnli")
        assert nli.is_hf_model is True
        assert nli.tokenizer is not None
        assert nli.model is not None

    def test_langchain_model_initialization(self):
        """Test NLI initialization with LangChain model."""
        mock_llm = Mock()
        nli = NLI(nli_llm=mock_llm)
        assert nli.is_hf_model is False
        assert nli.model == mock_llm
        assert nli.tokenizer is None
        assert nli.device is None

    def test_langchain_takes_precedence(self):
        """Test that nli_llm takes precedence over nli_model_name when both provided."""
        mock_llm = Mock()
        nli = NLI(nli_model_name="some-model", nli_llm=mock_llm)
        assert nli.is_hf_model is False
        assert nli.model == mock_llm

    def test_custom_max_length(self):
        """Test initialization with custom max_length."""
        nli = NLI(max_length=1000)
        assert nli.max_length == 1000

    def test_initialization_with_device(self):
        """Test initialization with device specification."""
        # Note: Not testing actual GPU allocation, just parameter passing
        nli = NLI(device="cpu")
        assert nli.device == "cpu"


class TestNLIPredictHuggingFace:
    """Test NLI prediction methods with HuggingFace models."""

    def test_predict_returns_probabilities(self):
        """Test that predict returns probability array by default."""
        nli = NLI()
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)  # [contradiction, neutral, entailment]
        assert result.sum() == pytest.approx(1.0)  # Probabilities should sum to 1
        assert np.all(result >= 0) and np.all(result <= 1)  # All probs in [0,1]

    def test_predict_returns_class_label(self):
        """Test that predict returns class label when return_probabilities=False."""
        nli = NLI()
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", return_probabilities=False)

        assert isinstance(result, str)
        assert result in ["contradiction", "neutral", "entailment"]

    def test_predict_entailment_case(self):
        """Test prediction on an entailment case."""
        nli = NLI()
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", return_probabilities=False)
        # Identical statements should be entailment
        assert result == "entailment"

    def test_predict_with_similar_sentences(self):
        """Test prediction with semantically similar sentences."""
        nli = NLI()
        result = nli.predict(hypothesis="A person is walking.", premise="A human is taking a walk.", return_probabilities=True)

        # Should favor entailment (index 2)
        assert result[0, 2] > result[0, 0]  # entailment > contradiction

    def test_predict_with_contradictory_sentences(self):
        """Test prediction with contradictory sentences."""
        nli = NLI()
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is red.", return_probabilities=False)

        # Colors contradict, but may be predicted as neutral depending on model
        assert result in ["contradiction", "neutral"]

    def test_predict_with_neutral_sentences(self):
        """Test prediction with unrelated sentences."""
        nli = NLI()
        result = nli.predict(hypothesis="The sky is blue.", premise="The grass is green.", return_probabilities=False)

        # Unrelated facts should be neutral
        assert result in ["neutral", "entailment"]


class TestNLIPredictLangChain:
    """Test NLI prediction methods with LangChain models."""

    def test_predict_probabilities_with_langchain(self):
        """Test that LangChain model returns probabilities."""
        mock_llm = Mock()
        
        # Mock responses for p_false, p_neutral, p_true queries
        mock_response_1 = Mock()
        mock_response_1.content = "No"
        mock_response_1.response_metadata = {}
        
        mock_response_2 = Mock()
        mock_response_2.content = "No"
        mock_response_2.response_metadata = {}
        
        mock_response_3 = Mock()
        mock_response_3.content = "Yes"
        mock_response_3.response_metadata = {}
        
        mock_llm.invoke.side_effect = [mock_response_1, mock_response_2, mock_response_3]

        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", return_probabilities=True)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        assert result.sum() == pytest.approx(1.0)
        assert mock_llm.invoke.call_count == 3  # Called once for each probability

    def test_predict_probabilities_with_logprobs(self):
        """Test that LangChain model uses logprobs when available."""
        mock_llm = Mock()
        
        # Mock responses with logprobs (OpenAI-style)
        mock_response_1 = Mock()
        mock_response_1.content = "No"
        mock_response_1.response_metadata = {
            'logprobs': {
                'content': [{
                    'top_logprobs': [
                        {'token': 'yes', 'logprob': -3.0},  # exp(-3.0) ≈ 0.0498
                        {'token': 'no', 'logprob': -0.1}    # exp(-0.1) ≈ 0.9048
                    ]
                }]
            }
        }
        
        mock_response_2 = Mock()
        mock_response_2.content = "No"
        mock_response_2.response_metadata = {
            'logprobs': {
                'content': [{
                    'top_logprobs': [
                        {'token': 'yes', 'logprob': -2.0},
                        {'token': 'no', 'logprob': -0.5}
                    ]
                }]
            }
        }
        
        mock_response_3 = Mock()
        mock_response_3.content = "Yes"
        mock_response_3.response_metadata = {
            'logprobs': {
                'content': [{
                    'top_logprobs': [
                        {'token': 'yes', 'logprob': -0.1},  # High probability
                        {'token': 'no', 'logprob': -3.0}
                    ]
                }]
            }
        }
        
        mock_llm.invoke.side_effect = [mock_response_1, mock_response_2, mock_response_3]
        
        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", return_probabilities=True)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        assert result.sum() == pytest.approx(1.0)
        # Since we're using logprobs, values should not be binary (0/1)
        # They should be extracted from the actual logprob values
        assert not np.any((result == 0.0) | (result == 1.0))
        assert mock_llm.invoke.call_count == 3

    def test_predict_class_with_langchain(self):
        """Test that LangChain model returns class label."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "entailment"
        mock_llm.invoke.return_value = mock_response

        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", return_probabilities=False)

        assert result == "entailment"
        assert mock_llm.invoke.call_count == 1

    def test_predict_handles_unclear_response(self):
        """Test that unclear LLM responses are handled gracefully."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "I don't know"
        mock_llm.invoke.return_value = mock_response

        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", return_probabilities=False)

        # Should default to neutral on unclear response
        assert result == "neutral"

    def test_predict_probabilities_handles_exception(self):
        """Test that exceptions during probability prediction are handled."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API Error")

        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", return_probabilities=True)

        # Should return uniform probabilities on error
        assert isinstance(result, np.ndarray)
        # Use allclose for 2D array comparison since pytest.approx doesn't support nested structures
        assert np.allclose(result, [[1 / 3, 1 / 3, 1 / 3]])

    def test_predict_class_handles_exception(self):
        """Test that exceptions during class prediction are handled."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API Error")

        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", return_probabilities=False)

        # Should default to neutral on error
        assert result == "neutral"


class TestNLIEdgeCases:
    """Test edge cases and error handling."""

    def test_long_text_truncation_warning(self):
        """Test that warning is issued for texts exceeding max_length."""
        nli = NLI(max_length=50)
        long_text = "This is a very long sentence " * 20

        with pytest.warns(UserWarning, match="Maximum response length exceeded"):
            nli.predict(hypothesis=long_text, premise="Short text")

    def test_empty_strings(self):
        """Test prediction with empty strings."""
        nli = NLI()
        result = nli.predict(hypothesis="", premise="")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)

    def test_special_characters(self):
        """Test prediction with special characters."""
        nli = NLI()
        result = nli.predict(hypothesis="The cost is $100.", premise="The price is $100.")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)

    def test_multilingual_text(self):
        """Test prediction with non-English text (if model supports it)."""
        nli = NLI()
        result = nli.predict(hypothesis="Le ciel est bleu.", premise="Le ciel est bleu.")

        # Should still return valid probabilities even if not well-calibrated
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)


class TestNLIComparison:
    """Test comparing HuggingFace and LangChain models on same inputs."""

    def test_both_models_return_same_format_probabilities(self):
        """Test that both model types return probabilities in same format."""
        mock_llm = Mock()

        # Mock responses for probability queries
        responses = [Mock() for _ in range(3)]
        responses[0].content = "Yes"  # contradiction
        responses[1].content = "No"  # neutral
        responses[2].content = "No"  # entailment
        mock_llm.invoke.side_effect = responses

        hf_nli = NLI()
        llm_nli = NLI(nli_llm=mock_llm)

        hypothesis = "The sky is blue."
        premise = "The sky is blue."

        hf_result = hf_nli.predict(hypothesis=hypothesis, premise=premise)
        llm_result = llm_nli.predict(hypothesis=hypothesis, premise=premise)

        # Both should return same shape
        assert hf_result.shape == llm_result.shape == (1, 3)
        # Both should be valid probability distributions
        assert hf_result.sum() == pytest.approx(1.0)
        assert llm_result.sum() == pytest.approx(1.0)

    def test_both_models_return_same_format_classes(self):
        """Test that both model types return class labels in same format."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "entailment"
        mock_llm.invoke.return_value = mock_response

        hf_nli = NLI()
        llm_nli = NLI(nli_llm=mock_llm)

        hypothesis = "The sky is blue."
        premise = "The sky is blue."

        hf_result = hf_nli.predict(hypothesis=hypothesis, premise=premise, return_probabilities=False)
        llm_result = llm_nli.predict(hypothesis=hypothesis, premise=premise, return_probabilities=False)

        # Both should return strings
        assert isinstance(hf_result, str)
        assert isinstance(llm_result, str)
        # Both should be valid labels
        assert hf_result in ["contradiction", "neutral", "entailment"]
        assert llm_result in ["contradiction", "neutral", "entailment"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
