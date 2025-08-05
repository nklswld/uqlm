from typing import Any, List, Optional
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier, UQResult
from uqlm.black_box import LUQScorer
from langchain_core.language_models.chat_models import BaseChatModel

class LongFormUQ(UncertaintyQuantifier):
    def __init__(self,
                 llm: Optional[BaseChatModel] = None,
                 scorers: Optional[List[str]] = None,
                 device: Any = None,
                 nli_model_name: str = "microsoft/deberta-large-mnli",
                 system_prompt: str = "You are a helpful assistant.",
                 max_calls_per_min: Optional[int] = None,
                 sampling_temperature: float = 1.0,
                 use_n_param: bool = False,
                 max_length: int = 2000,
                 postprocessor: Optional[Any] = None,
                 # verbose: bool = False,
                ) -> None:
        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
        self.prompts = None
        self.max_length = max_length
        # self.verbose = verbose
        self.sampling_temperature = sampling_temperature
        self.nli_model_name = nli_model_name
        self._validate_scorers(scorers)

    def _validate_scorers(self, scorers: Optional[List[str]]) -> None:
        self.scorer_objects = {}
        if scorers is None:
            scorers = self.default_long_form_names
        for scorer in scorers:
            if scorer == "luq":
                self.scorer_objects["luq"] = LUQScorer(nli_model_name=self.nli_model_name, device=self.device, max_length=self.max_length)
            elif scorer == "graph_based":
                pass
            else:
                raise ValueError(f"Invalid scorer: {scorer}")
        self.scorers = scorers
    
    async def decompose(self, responses: List[str]) -> List[List[str]]:
        # async response utils.postprocessors.claims_postprocessor(response)
        pass

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        pass

    def _generate_responses(self, prompts: List[str], num_responses: int = 5, show_progress_bars: Optional[bool] = True) -> List[str]:
        # generate responses
        # decompose responses into claim sets
        # generate candidate responses
        # score 
        pass
    
    def score(self, claim_sets: List[List[str]], sampled_responses: List[List[str]], show_progress_bars: Optional[bool] = True, _display_header: bool = True) -> UQResult:
        # LUQScorer.evaluate()
        pass