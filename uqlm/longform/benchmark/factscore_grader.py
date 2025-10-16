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


from typing import List, Optional
from rich.progress import Progress
from uqlm.utils.response_generator import ResponseGenerator
from uqlm.utils.prompts.factscore_prompts import FACTSCORE_SYSTEM_PROMPT


class FactScoreGrader:
    def __init__(self, llm, max_calls_per_min: int = None):
        self.rg = ResponseGenerator(llm, max_calls_per_min=max_calls_per_min)
        self.system_prompt = FACTSCORE_SYSTEM_PROMPT

    def construct_entailment_prompt(self, claim: str, answer: str):
        return f"""
            Context: {answer}
            Claim: {claim}
            Is the claim supported by the context above?
            Answer only Yes or No:
            """

    def _str_to_bool(self, response) -> bool:
        """Parse LLM response to extract Yes/No answer and convert to boolean"""
        response_text = response.strip().lower()
        if "yes" in response_text:
            return True
        elif "no" in response_text:
            return False
        else:
            return False

    def _format_grades(self, flat_grades_list: List[str], reference_structure: List[List[str]]) -> List[bool]:
        """
        Reshape a flat list into a nested list structure that matches the reference structure.

        Args:
            flat_list: A flat list of elements with length equal to the sum of all inner list lengths in reference_structure
            reference_structure: A list of lists with varying inner list lengths

        Returns:
            A nested list with the same structure as reference_structure, containing elements from flat_list
        """
        formatted_grades = []
        flat_index = 0
        for inner_list in reference_structure:
            inner_length = len(inner_list)
            new_inner_list = flat_grades_list[flat_index : flat_index + inner_length]
            new_inner_list_bool = [self._str_to_bool(r) for r in new_inner_list]
            formatted_grades.append(new_inner_list_bool)
            flat_index += inner_length
        return formatted_grades

    async def grade_claims(self, claim_sets: List[List[str]], answers: List[str], progress_bar: Optional[Progress] = None) -> List[List[bool]]:
        prompts = []
        indices = []
        for i, (claim_set, answer) in enumerate(zip(claim_sets, answers)):
            for j, claim in enumerate(claim_set):
                prompt = self.construct_entailment_prompt(claim=claim, answer=answer)
                prompts.append(prompt)
                indices.append((i, j))

        self.generations = await self.rg.generate_responses(prompts=prompts, system_prompt=self.system_prompt, progress_bar=progress_bar)
        self.responses = self.generations["data"]["response"]
        formatted_grade_lists = self._format_grades(flat_grades_list=self.responses, reference_structure=claim_sets)
        return formatted_grade_lists
