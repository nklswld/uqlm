from uqlm.black_box.baseclass.claim_scorer import ClaimScorer
from typing import List, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from uqlm.black_box.nli import NLIScorer

class LUQScorer(ClaimScorer):

    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli", device: Any = None, max_length: int = 2000):
        self.nli_model_name = nli_model_name
        self.device = device
        self.nli_scorer = NLIScorer(device=device, 
                                    nli_model_name=nli_model_name,
                                    max_length=max_length)
        

    def evaluate(self, claims: List[str], sampled_responses: List[str]) -> List[float]:
        """
        Evaluate the LUQ score for a list of claims and sampled responses.
        """
        num_claims = len(claims)
        num_samples = len(sampled_responses)
        scores = np.zeros((num_claims, num_samples))
        for claim_idx in range(num_claims):
            claim = claims[claim_idx]
            for sample_idx, sample in enumerate(sampled_responses):
                nli_proba = self.nli_scorer.predict(claim, sample) # assuming this checks claim is entailed in the sample
                nli_label = self.nli_scorer.label_mapping[nli_proba.argmax(axis=1)[0]]
                if nli_label == "entailment":
                    score_ = 1
                elif nli_label == "neutral":
                    score_ = 0.5
                else:
                    score_ = 0
                print(f"claim: {claim}, sample: {sample}, nli_proba: {nli_proba}, nli_label: {nli_label}, score: {score_}")
                scores[claim_idx, sample_idx] = score_
        print(scores)
        scores_per_claim = np.mean(scores, axis=-1) 
        return scores_per_claim  
    