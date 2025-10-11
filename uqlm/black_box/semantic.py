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


import math
import numpy as np
from collections import deque, Counter
from typing import Any, Dict, List, Optional
from transformers import logging
import time
from rich.progress import Progress
from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer
from uqlm.utils.nli import NLI

logging.set_verbosity_error()


class SemanticScorer(SimilarityScorer):
    def __init__(self, device: Any = None, verbose: bool = False, nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000) -> None:
        """
        A class to computing NLI-based confidence scores. This class offers two types of confidence scores, namely
        noncontradiction probability :footcite:`chen2023quantifyinguncertaintyanswerslanguage` and semantic entropy
        :footcite:`farquhar2024detectinghallucinations`.

        Parameters
        ----------
        device : torch.device input or torch.device object, default=None
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to
            leverage the GPU.

        verbose : bool, default=False
            Specifies whether to print verbose status updates of NLI scoring process

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        """
        self.verbose = verbose
        self.nli_model = NLI(nli_model_name=nli_model_name, max_length=max_length, device=device)
        self.label_mapping = ["contradiction", "neutral", "entailment"]

    def evaluate(self, responses: List[str], sampled_responses: List[List[str]], responses_logprobs: List[List[Dict[str, Any]]] = None, sampled_responses_logprobs: List[List[List[Dict[str, Any]]]] = None, use_best: bool = False, compute_entropy: bool = False, best_response_selection: str = "discrete", progress_bar: Optional[Progress] = None) -> Dict[str, Any]:
        """
        Evaluate confidence scores on LLM responses.

        Parameters
        ----------
        responses : list of strings
            Original LLM response

        sampled_responses : list of list of strings
            Sampled candidate responses to be compared to the original response

        responses_logprobs : list of list of dicts, default=None
            Log probabilities of the original response

        sampled_responses_logprobs : list of list of list of dicts, default=None
            Log probabilities of the sampled responses

        use_best : bool
            Specifies whether to swap the original response for the uncertainty-minimized response
            based on semantic entropy clusters.

        compute_entropy : bool, default=False
            Specifies whether to include semantic entropy in returned result

        best_response_selection : str, default="discrete"
            Specifies the type of entropy confidence score to compute best response. Must be one of "discrete" or "token-based".

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        Dict
            Dictionary containing mean NLI and (optionally) semantic entropy scores.
            The dictionary will also contain original and multiple responses, updated if `use_best` is True
        """
        self.num_responses = len(sampled_responses[0])
        self.logprobs, self.multiple_logprobs = responses_logprobs, sampled_responses_logprobs
        observed_consistency_data = {"noncontradiction": [], "discrete_semantic_entropy": [], "tokenprob_semantic_entropy": [], "responses": responses, "sampled_responses": sampled_responses}

        def _process_i(i, response):
            oc_result_i = self._observed_consistency_i(original=response, candidates=sampled_responses[i], i=i, use_best=use_best, compute_entropy=compute_entropy)
            observed_consistency_data["noncontradiction"].append(oc_result_i["nli_score_i"])
            observed_consistency_data["discrete_semantic_entropy"].append(oc_result_i["discrete_semantic_entropy"])
            observed_consistency_data["tokenprob_semantic_entropy"].append(oc_result_i["tokenprob_semantic_entropy"])
            responses[i] = oc_result_i["response"]  # Replace with optimized response if use_best
            sampled_responses[i] = oc_result_i["candidates"]  # Replace with updated candidates if use_best

        if progress_bar:
            progress_task = progress_bar.add_task("  - Scoring responses with NLI...", total=len(responses))
        for i, response in enumerate(responses):
            _process_i(i, response)
            if progress_bar:
                progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)

        if use_best:
            observed_consistency_data["responses"] = responses
            observed_consistency_data["sampled_responses"] = sampled_responses
        return observed_consistency_data

    def _observed_consistency_i(self, original: str, candidates: List[str], i: int = None, use_best: bool = False, compute_entropy: bool = False) -> Dict[str, Any]:
        """
        Compute observed consistency score on the provided original response and multiple candidates.
        """
        scores = {}
        nli_scores = []
        best_response = original
        discrete_semantic_entropy, tokenprob_semantic_entropy = None, None
        if compute_entropy or use_best:
            all_responses = [original] + candidates
            all_logprobs = [self.logprobs[i]] + self.multiple_logprobs[i] if (self.logprobs and self.multiple_logprobs) else None
            tmp = self._semantic_entropy_process(candidates=all_responses, i=i, logprobs_results=all_logprobs)
            best_response, discrete_semantic_entropy, scores, tokenprob_semantic_entropy = tmp
            if use_best:
                all_responses.remove(best_response)
                candidates = all_responses

        for candidate in candidates:
            if (candidate, best_response) in scores:
                nli_score = scores[(candidate, best_response)]
            else:
                nli_score = self._get_nli_results(response1=best_response, response2=candidate)["score"]
            nli_scores.append(nli_score)

        return {"nli_score_i": np.mean(nli_scores), "candidates": candidates, "response": best_response, "discrete_semantic_entropy": discrete_semantic_entropy, "tokenprob_semantic_entropy": tokenprob_semantic_entropy}

    def _semantic_entropy_process(self, candidates: List[str], i: int = None, logprobs_results: List[List[Dict[str, Any]]] = None, best_response_selection: str = "discrete") -> Any:
        """
        Executes complete process for semantic entropy and returns best response, SE score, and dictionary
        of NLI scores for response pairs
        """
        if self.verbose and i is not None:
            print("Question No. - ", i + 1)
        tokenprob_response_probabilities, response_probabilities = self._compute_response_probabilities(logprobs_results=logprobs_results, num_responses=len(candidates))
        clustered_responses, cluster_indices, nli_scores = self._cluster_responses(responses=candidates)
        # Compute discrete semantic entropy
        cluster_probabilities = self._compute_cluster_probabilities(response_probabilities=response_probabilities, cluster_indices=cluster_indices)
        best_response = self._default_best_response_selection(clustered_responses=clustered_responses, cluster_probabilities=cluster_probabilities)
        discrete_semantic_entropy = self._compute_semantic_entropy(cluster_probabilities=cluster_probabilities)

        # Compute token-level semantic entropy
        tokenprob_semantic_entropy = None
        if tokenprob_response_probabilities:
            tokenprob_cluster_probabilities = self._compute_cluster_probabilities(response_probabilities=tokenprob_response_probabilities, cluster_indices=cluster_indices)
            tokenprob_semantic_entropy = self._compute_semantic_entropy(cluster_probabilities=tokenprob_cluster_probabilities)
            if best_response_selection == "token-based":
                best_response = self._default_best_response_selection(clustered_responses=clustered_responses, cluster_probabilities=tokenprob_cluster_probabilities)

        return (best_response, discrete_semantic_entropy, nli_scores, tokenprob_semantic_entropy)

    def _default_best_response_selection(self, clustered_responses: List[List[str]], cluster_probabilities: List[float]) -> str:
        """Select the best response from the clustered responses based on the cluster probabilities"""
        return clustered_responses[cluster_probabilities.index(max(cluster_probabilities))][0]

    def _compute_cluster_probabilities(self, response_probabilities: List[float], cluster_indices: List[List[int]]) -> List[float]:
        """Compute cluster probabilities"""
        cluster_probabilities = [0] * len(cluster_indices)
        for i, cluster_index in enumerate(cluster_indices):
            cluster_probabilities[i] = sum([response_probabilities[j] for j in cluster_index])
        return self._normalize_cluster_probabilities(cluster_probabilities=cluster_probabilities)

    def _compute_response_probabilities(self, logprobs_results: List[List[Dict[str, Any]]], num_responses: int = None) -> List[float]:
        """Compute response probabilities"""
        uniform_response_probabilities = [1 / num_responses] * num_responses
        tokenprob_response_probabilities = [self.avg_logprob(logprobs_i) if logprobs_i else np.nan for logprobs_i in logprobs_results] if logprobs_results else None
        return tokenprob_response_probabilities, uniform_response_probabilities

    def _get_nli_results(self, response1: str, response2: str) -> Dict[str, Any]:
        """This method computes mean NLI score and determines whether entailment exists."""
        if response1 == response2:
            avg_nli_score, entailment = 1, True
        else:
            left = self.nli_model.predict(hypothesis=response1, premise=response2)
            left_label = left.ternary_label

            right = self.nli_model.predict(hypothesis=response2, premise=response1)
            right_label = right.ternary_label
            s1 = 1 - left.contradiction_probability
            s2 = 1 - right.contradiction_probability

            entailment = left_label == "entailment" or right_label == "entailment"
            avg_nli_score = (s1 + s2) / 2
        return {"score": avg_nli_score, "entailment": entailment}

    def _cluster_responses(self, responses: List[str]) -> Any:
        """
        This method create clusters from a list of responses based on the semantic meaning of each response.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses

        Returns
        ----------
        A list of lists, where each list represents a cluster.
        """
        clusters, cluster_indices = [deque([responses[0]])], [deque([0])]
        nli_scores = {}
        entailments = {}
        for i in range(1, len(responses)):
            new_cluster_indicator = True
            for j, cluster in enumerate(clusters):
                key, rev_key = (cluster[0], responses[i]), (responses[i], cluster[0])
                if key in nli_scores:
                    # Do not recompute if pair already assessed
                    entailment = entailments[key]
                else:
                    # Compute nli score and entailment if pair not yet assessed
                    nli_result = self._get_nli_results(response1=cluster[0], response2=responses[i])
                    score, entailment = nli_result["score"], nli_result["entailment"]
                    nli_scores[key], nli_scores[rev_key] = score, score
                    entailments[key], entailments[rev_key] = entailment, entailment
                if entailment:
                    new_cluster_indicator = False
                    cluster.append(responses[i])
                    cluster_indices[j].append(i)

            if new_cluster_indicator:
                clusters.append(deque([responses[i]]))
                cluster_indices.append(deque([i]))

        # Arrange cluster so that first element is mode (if exists) else longest
        clusters = [self._sort_responses(list(cluster)) for cluster in clusters]

        # Normalize cluster probabilities
        # cluster_probabilities = self._normalize_cluster_probabilities(cluster_probabilities=cluster_probabilities)
        return clusters, cluster_indices, nli_scores

    def _normalize_entropy(self, entropy_values):
        return [e / math.log(self.num_responses + 1) for e in entropy_values]

    @staticmethod
    def _compute_semantic_entropy(cluster_probabilities: List[float]) -> float:
        """
        Helper function to compute semantic entropy score from cluster probabilities
        """
        return abs(sum([p * math.log(p) if p > 0.0 else 0 for p in cluster_probabilities]))

    @staticmethod
    def avg_logprob(logprobs: List[Dict[str, Any]]) -> float:
        "Compute average logprob"
        return np.prod([np.exp(d["logprob"]) for d in logprobs])

    @staticmethod
    def _normalize_cluster_probabilities(cluster_probabilities: List[float]) -> float:
        """Normalize cluster probabilities"""
        total_probability = sum(cluster_probabilities)
        return [cp_i / total_probability for cp_i in cluster_probabilities]

    @staticmethod
    def _sort_responses(responses: List[str]) -> List[str]:
        """Sorts responses in a cluster"""
        counter = Counter(responses)
        mode_str, count = counter.most_common(1)[0]
        if count > 1:
            return sorted(responses, key=lambda x: (x != mode_str, x))
        else:
            return sorted(responses, key=len, reverse=True)

