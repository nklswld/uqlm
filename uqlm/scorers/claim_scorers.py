from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier, UQResult

class LUQ(UncertaintyQuantifier):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm=llm)

    def generate_and_score(self, prompts: List[str], num_responses: int = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        pass

    def score(self, responses: List[str], sampled_responses: List[List[str]], show_progress_bars: Optional[bool] = True) -> UQResult:
        pass