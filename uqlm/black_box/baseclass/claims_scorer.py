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

from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import numpy as np

@dataclass
class ClaimScores:
    """
    ClaimsScore is a dataclass that contains the aggregated score and the raw scores for each claim set.
    """
    aggregated_score: List[float]
    raw_scores: List[np.ndarray]
    


class ClaimScorer(ABC):
    """Abstract class for text similarity scorers"""

    @abstractmethod
    def __init__(self):
        """Abstract constructor method"""
        pass

    @abstractmethod
    def evaluate(self, claim_sets: List[List[str]], sampled_responses: List[str]) -> ClaimScores:
        """Abstract method for metric computation"""
        pass

