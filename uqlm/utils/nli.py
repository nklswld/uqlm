from typing import Any, Optional
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from uqlm.utils.prompts.entailment_prompts import get_entailment_prompt


class NLI:
    def __init__(self, nli_model_name: Optional[str] = "microsoft/deberta-large-mnli", nli_llm: Optional[BaseChatModel] = None, max_length: int = 2000, device: Any = None) -> None:
        """
        A class to compute NLI predictions.

        Parameters
        ----------
        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which HuggingFace NLI model to use. Must be acceptable input to
            AutoTokenizer.from_pretrained() and AutoModelForSequenceClassification.from_pretrained().
            Ignored if nli_llm is provided.

        nli_llm : BaseChatModel, default=None
            A LangChain chat model for LLM-based NLI inference. If provided, takes precedence over nli_model_name.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError. Only applies to HuggingFace models.

        device : torch.device input or torch.device object, default=None
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to
            leverage the GPU. Only applies to HuggingFace models.
        """
        # Prioritize nli_llm if provided, otherwise use nli_model_name
        self.is_hf_model = nli_llm is None
        self.max_length = max_length
        self.label_mapping = ["contradiction", "neutral", "entailment"]

        if self.is_hf_model:
            # Initialize HuggingFace model
            if nli_model_name is None:
                raise ValueError("Must specify either nli_model_name or nli_llm.")
            self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
            self.device = device
            self.model = model.to(self.device) if self.device else model
        else:
            # LangChain model
            self.device = None
            self.tokenizer = None
            self.model = nli_llm

    def predict(self, hypothesis: str, premise: str, return_probabilities: bool = True) -> Any:
        """
        This method computes NLI predictions on the provided inputs.

        Parameters
        ----------
        hypothesis : str
            The hypothesis text for NLI classification.

        premise : str
            The premise text for NLI classification.

        return_probabilities : bool, default=True
            If True, returns probabilities for all three classes [contradiction, neutral, entailment].
            If False, returns a single class label string ("contradiction", "neutral", or "entailment").

        Returns
        -------
        numpy.ndarray or str
            If return_probabilities=True: numpy array with 3 probabilities for [contradiction, neutral, entailment]
            If return_probabilities=False: string with the predicted class label
        """
        if self.is_hf_model:
            return self._predict_hf(hypothesis, premise, return_probabilities)
        else:
            return self._predict_langchain(hypothesis, premise, return_probabilities)

    def _predict_hf(self, hypothesis: str, premise: str, return_probabilities: bool = True) -> Any:
        """
        Perform NLI prediction using a HuggingFace model.

        Parameters
        ----------
        hypothesis : str
            The hypothesis text for NLI classification.

        premise : str
            The premise text for NLI classification.

        return_probabilities : bool, default=True
            If True, returns probabilities for all three classes.
            If False, returns the predicted class label.

        Returns
        -------
        numpy.ndarray or str
            If return_probabilities=True: numpy array with 3 probabilities
            If return_probabilities=False: string with the predicted class label
        """
        if len(hypothesis) > self.max_length or len(premise) > self.max_length:
            warnings.warn("Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length")

        concat = hypothesis[0 : self.max_length] + " [SEP] " + premise[0 : self.max_length]
        encoded_inputs = self.tokenizer(concat, padding=True, return_tensors="pt")

        if self.device:
            encoded_inputs = {name: tensor.to(self.device) for name, tensor in encoded_inputs.items()}

        logits = self.model(**encoded_inputs).logits
        np_logits = logits.detach().cpu().numpy() if self.device else logits.detach().numpy()
        probabilities = np.exp(np_logits) / np.exp(np_logits).sum(axis=-1, keepdims=True)

        if return_probabilities:
            return probabilities
        else:
            # Return the class with highest probability
            predicted_class_idx = probabilities.argmax(axis=1)[0]
            return self.label_mapping[predicted_class_idx]

    def _predict_langchain(self, hypothesis: str, premise: str, return_probabilities: bool = True) -> Any:
        """
        Perform NLI prediction using a LangChain BaseChatModel.

        This method queries the LLM three times (once for each class) to estimate probabilities,
        or once if only the class label is needed.

        Parameters
        ----------
        hypothesis : str
            The hypothesis text (claim) for NLI classification.

        premise : str
            The premise text (source text) for NLI classification.

        return_probabilities : bool, default=True
            If True, estimates probabilities for all three classes by querying the LLM 3 times.
            If False, performs a single query to determine the most likely class.

        Returns
        -------
        numpy.ndarray or str
            If return_probabilities=True: numpy array with 3 estimated probabilities for [contradiction, neutral, entailment]
            If return_probabilities=False: string with the predicted class label
        """
        if return_probabilities:
            # Query the LLM for each class to get probability estimates
            # We'll use Yes/No prompts for each class and interpret the response
            probabilities = []

            for style in ["p_false", "p_neutral", "p_true"]:  # contradiction, neutral, entailment
                prompt = get_entailment_prompt(claim=hypothesis, source_text=premise, style=style)
                messages = [SystemMessage("You are a helpful assistant that evaluates natural language inference relationships."), HumanMessage(prompt)]

                try:
                    response = self.model.invoke(messages)
                    response_text = response.content.strip().lower()

                    # Try to extract probability from logprobs if available
                    prob = self._extract_yes_probability_from_response(response, response_text)

                    probabilities.append(prob)
                except Exception as e:
                    warnings.warn(f"Error during LangChain NLI inference: {e}. Assigning uniform probabilities.")
                    return np.array([[1 / 3, 1 / 3, 1 / 3]])

            # Normalize probabilities
            probabilities = np.array(probabilities)
            if probabilities.sum() > 0:
                probabilities = probabilities / probabilities.sum()
            else:
                probabilities = np.array([1 / 3, 1 / 3, 1 / 3])

            return probabilities.reshape(1, 3)

        else:
            # Single query to determine the class
            # Use a general entailment prompt that asks for classification
            prompt = get_entailment_prompt(claim=hypothesis, source_text=premise, style="nli_classification")

            messages = [SystemMessage("You are a helpful assistant that evaluates natural language inference relationships."), HumanMessage(prompt)]

            try:
                response = self.model.invoke(messages)
                response_text = response.content.strip().lower()

                # Parse response to get class label
                if "entailment" in response_text:
                    return "entailment"
                elif "contradiction" in response_text:
                    return "contradiction"
                elif "neutral" in response_text:
                    return "neutral"
                else:
                    warnings.warn(f"Unclear NLI response from LangChain model: '{response_text}'. Defaulting to 'neutral'.")
                    return "neutral"
            except Exception as e:
                warnings.warn(f"Error during LangChain NLI inference: {e}. Defaulting to 'neutral'.")
                return "neutral"
    
    def _extract_yes_probability_from_response(self, response: Any, response_text: str) -> float:
        """
        Extract the probability of "Yes" from a LangChain response.
        
        If logprobs are available, uses the actual token probability.
        Otherwise, falls back to binary classification based on response text.
        
        Supports both OpenAI and Vertex AI (Gemini) logprobs formats.
        
        Parameters
        ----------
        response : AIMessage or similar
            The response object from the LangChain model
        response_text : str
            The lowercased content of the response
        
        Returns
        -------
        float
            Probability estimate for "Yes" answer (between 0 and 1)
        """
        # Try to extract from logprobs if available
        if hasattr(response, 'response_metadata') and response.response_metadata:
            
            # Vertex AI (Gemini) style logprobs
            if 'logprobs_result' in response.response_metadata:
                logprobs_result = response.response_metadata['logprobs_result']
                if logprobs_result and len(logprobs_result) > 0:
                    first_token_data = logprobs_result[0]
                    token = first_token_data.get('token', '').strip().lower()
                    logprob = first_token_data.get('logprob', None)
                    
                    if logprob is not None:
                        prob = np.exp(logprob)
                        
                        # For Vertex AI, we get the logprob of the actual generated token
                        # So we need to interpret based on what token was generated
                        if token in ['yes', 'yes.', 'yes,']:
                            # Token is "yes", so probability of "yes" is high
                            return prob
                        elif token in ['no', 'no.', 'no,']:
                            # Token is "no", so probability of "yes" is low
                            return 1.0 - prob
            
            # OpenAI-style logprobs
            if 'logprobs' in response.response_metadata:
                logprobs_data = response.response_metadata['logprobs']
                if logprobs_data and 'content' in logprobs_data:
                    content_logprobs = logprobs_data['content']
                    if content_logprobs and len(content_logprobs) > 0:
                        # Get the first token's top logprobs
                        first_token = content_logprobs[0]
                        if 'top_logprobs' in first_token:
                            top_logprobs = first_token['top_logprobs']
                            # Look for "Yes", "yes", "YES", "No", "no", "NO" tokens
                            yes_prob = 0.0
                            no_prob = 0.0
                            
                            for logprob_item in top_logprobs:
                                token = logprob_item.get('token', '').strip().lower()
                                logprob = logprob_item.get('logprob', None)
                                
                                if logprob is not None:
                                    prob = np.exp(logprob)
                                    if token in ['yes', 'yes.', 'yes,']:
                                        yes_prob = max(yes_prob, prob)
                                    elif token in ['no', 'no.', 'no,']:
                                        no_prob = max(no_prob, prob)
                            
                            # If we found yes/no probabilities, normalize them
                            total = yes_prob + no_prob
                            if total > 0:
                                return yes_prob / total
        
        # Fallback: binary classification based on text
        if "yes" in response_text[:10]:  # Check first 10 chars for yes
            return 1.0
        elif "no" in response_text[:10]:
            return 0.0
        else:
            # If unclear, assign neutral probability
            return 0.5
