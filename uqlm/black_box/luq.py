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

from typing import List, Any, Tuple
import numpy as np
from uqlm.black_box.nli import NLIScorer
from uqlm.black_box.baseclass.claims_scorer import ClaimsScorer

class LUQScorer(ClaimsScorer):
    """
    LUQScorer is a class that evaluates the LUQ score for a list of claim sets and sampled responses.
    """
    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli", device: Any = None, max_length: int = 2000):
        self.nli_model_name = nli_model_name
        self.device = device
        self.nli_scorer = NLIScorer(device=device, 
                                    nli_model_name=nli_model_name,
                                    max_length=max_length)
        

    def evaluate(self, claim_sets: List[List[str]], sampled_responses: List[str]) -> Tuple[List[float], List[np.ndarray]]:
        """
        Evaluate the LUQ score for a list of claims and sampled responses.
        """
        luq_score = np.zeros(len(claim_sets))
        entailment_scores = []
        for claim_set_idx, claim_set in enumerate(claim_sets):
            num_claims = len(claim_set)
            num_samples = len(sampled_responses)
            scores = np.zeros((num_claims, num_samples))
            for claim_idx in range(num_claims):
                claim = claim_set[claim_idx]
                for sample_idx, sample in enumerate(sampled_responses):
                    nli_proba = self.nli_scorer.predict(sample, claim)
                    nli_label = self.nli_scorer.label_mapping[nli_proba.argmax(axis=1)[0]]
                    if nli_label == "entailment":
                        score_ = 1
                    elif nli_label == "neutral":
                        score_ = 0.5
                    else:
                        score_ = 0
                    print(f"claim: {claim} \n sample: {sample} \n nli_proba: {nli_proba} \n nli_label: {nli_label} \n score: {score_}")
                    scores[claim_idx, sample_idx] = score_
                print("--------------------------------")
            print(scores)
            entailment_scores.append(scores)
            scores_per_claim = np.mean(scores, axis=-1)
            luq_score[claim_set_idx] = scores_per_claim.mean()
            print(f'LUQ score for claim set {claim_set_idx}: {luq_score[claim_set_idx]}')
        return luq_score, entailment_scores
    