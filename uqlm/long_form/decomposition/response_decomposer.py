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

import asyncio
from typing import Dict, List, Optional
from uqlm.utils.prompt_templates import get_claim_breakdown_template
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
import re


class ResponseDecomposer:
    def __init__(self, claim_decomposition_llm: Optional[BaseChatModel] = None) -> None:
        """
        Class for decomposing responses into individual claims or sentences. This class is used as an intermediate
        step for longform UQ methods.

        Parameters
        ----------
        claim_decomposition_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.
        """
        self.claim_decomposition_llm = claim_decomposition_llm

    async def decompose_claims(self, responses: List[str], progress_bar: Optional[Progress] = None) -> List[Dict[str, str]]:
        """
        Parameters
        ----------
        responses: list[str] | str
            LLM response that will be decomposed into independent claims.

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses
        """
        if not self.claim_decomposition_llm:
            raise ValueError("llm must be provided to decompose responses into claims")
        if progress_bar:
            self.progress_task = progress_bar.add_task(" - Decomposing responses into claims...", total=len(responses))
        tasks = [self._get_claims_from_response(response=response, progress_bar=progress_bar) for response in responses]
        return await asyncio.gather(*tasks)

    async def _get_claims_from_response(self, response: str, progress_bar: Optional[Progress] = None) -> Dict[str, str]:
        decomposed_response = await self.claim_decomposition_llm.ainvoke(get_claim_breakdown_template(response))
        if progress_bar:
            progress_bar.update(self.progress_task, advance=1)
        return re.split(r"### ", decomposed_response.content)[1:]

    def decompose_sentences(self, responses: List[str], progress_bar: Optional[Progress] = None) -> List[dict]:
        """
        Parameters
        ----------
        responses: list[str] | str
            LLM response that will be decomposed into independent claims.

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses
        """
        if progress_bar:
            progress_task = progress_bar.add_task(" - Decomposing responses into sentences...", total=len(responses))

        sentence_lists = []
        for response in responses:
            if progress_bar:
                progress_bar.update(progress_task, advance=1)
            sentence_lists.append(self._get_sentences_from_response(response))
        return sentence_lists

    def _get_sentences_from_response(self, text: str) -> list[str]:
        """
        A more sophisticated approach inspired by NLTK's sentence tokenizer.
        Uses multiple passes and heuristics.
        """
        text = re.sub(r"(\d+)\.(\d+)", r"\1<DECIMAL>\2", text)
        abbrev_pattern = r"\b(?:mr|mrs|ms|dr|prof|sr|jr|vs|etc|inc|ltd|corp|co|st|ave|blvd|rd|ph\.d|m\.d|b\.a|m\.a|u\.s|u\.k|n\.y|l\.a|d\.c)\."

        abbreviations = re.finditer(abbrev_pattern, text, re.IGNORECASE)
        protected_text = text
        offset = 0

        for match in abbreviations:
            start, end = match.span()
            start += offset
            end += offset
            replacement = match.group().replace(".", "<DOT>")
            protected_text = protected_text[:start] + replacement + protected_text[end:]
            offset += len(replacement) - len(match.group())

        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern, protected_text.strip())

        for i, sentence in enumerate(sentences):
            sentence = sentence.replace("<DOT>", ".")
            sentence = sentence.replace("<DECIMAL>", ".")
            sentences[i] = sentence.strip()
        return sentences
