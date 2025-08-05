from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier, UQResult
from uqlm.black_box import LUQScorer
from uqlm.utils.postprocessors import claims_postprocessor

class LongFormUQ(UncertaintyQuantifier):
    def __init__(self,
                 claim_decomposition_llm: Optional[BaseChatModel],
                 generation_llm: Optional[BaseChatModel] = None,
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
        super().__init__(llm=generation_llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
        self.max_length = max_length
        # self.verbose = verbose
        self.sampling_temperature = sampling_temperature
        self.nli_model_name = nli_model_name
        self._validate_scorers(scorers)
        self.claim_decomposition_llm = claim_decomposition_llm
        self.prompts = None
        self.responses = None
        self.claim_sets = None
        self.sampled_responses = None
        self.num_responses = None
        self.scores_dict = {}

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

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        pass

    def score(self,
              responses: List[str], 
              sampled_responses: List[List[str]]
              ) -> UQResult:
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = len(sampled_responses[0])
        self.scores_dict = {k: [] for k in self.scorer_objects}
        self.claim_sets = claims_postprocessor(llm=self.claim_decomposition_llm, responses=responses)
        # self._construct_progress_bar(show_progress_bars)
        # self._display_scoring_header(show_progress_bars and _display_header)
        
        for scorer_key, scorer_object in self.scorer_objects.items():
            self.scores_dict[scorer_key] = scorer_object.evaluate(self.claim_sets, self.sampled_responses).to_dict()
        result = self._construct_result()
        # self._stop_progress_bar()
        # self.progress_bar = None  # if re-run ensure the same progress object is not used
        return result
    
    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {"responses": self.responses, 
                "claim_sets": self.claim_sets,
                "sampled_responses": self.sampled_responses}
        if self.prompts:
            data["prompts"] = self.prompts
        data.update(self.scores_dict)
        result = {"data": data, "metadata": {"temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses, "scorers": self.scorers}}
        return UQResult(result) 