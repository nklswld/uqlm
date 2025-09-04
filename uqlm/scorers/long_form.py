import warnings
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier, UQResult, DEFAULT_LONG_FORM_SCORERS
from uqlm.black_box import LUQScorer
from uqlm.utils.decomposer import ResponseDecomposer

SENTENCE_BASED_SCORERS = ["luq"]
CLAIM_BASED_SCORERS = ["luq_atomic"]

class LongFormUQ(UncertaintyQuantifier):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        scorers: Optional[List[str]] = None,
        claim_decomposition_llm: Optional[BaseChatModel],
        device: Any = None,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        system_prompt: str = "You are a helpful assistant.",
        max_calls_per_min: Optional[int] = None,
        sampling_temperature: float = 1.0,
        use_n_param: bool = False,
        max_length: int = 2000,
        postprocessor: Optional[Any] = None,
        return_responses: str = "all",
    ) -> None:
        """
        Class for longform uncertainty quantification. Implements claimwise analogs of other UQ-based 
        scorers for improved performance with longform tasks.

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        scorers : subset of {"luq", "luq_atomic"}, default=None
            Specifies which black box (consistency) scorers to include. If None, defaults to
            ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim"]. 
            
        claim_decomposition_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel` to be used for decomposing responses into individual claims 
            or 'factoids'. If a scorer that requires claim decomposition (e.g., "luq_atomic") is specified 
            and claim_decomposition_llm is None, the provided `llm` will be used for claim decomposition.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Applies to 'luq', 'luq_atomic'
            scorers. Pass a torch.device to leverage GPU.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs before black-box comparisons.

        return_responses : str, default="all"
            If a postprocessor is used, specifies whether to return only postprocessed responses, only raw responses,
            or both. Specified with 'postprocessed', 'raw', or 'all', respectively.

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        """
        
        super().__init__(llm=generation_llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
        self.max_length = max_length
        self.sampling_temperature = sampling_temperature
        self.nli_model_name = nli_model_name
        self.claim_decomposition_llm = claim_decomposition_llm
        self.default_long_form_scorers = DEFAULT_LONG_FORM_SCORERS
        self._validate_scorers(scorers)
        self.decomposer = ResponseDecomposer(claim_decomposition_llm=claim_decomposition_llm if claim_decomposition_llm else llm)
        self.prompts = None
        self.responses = None
        self.claim_sets = None
        self.sentence_sets = None
        self.sampled_responses = None
        self.num_responses = None
        self.scores_dict = {}

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate LLM responses, sampled LLM (candidate) responses, and compute confidence scores with specified scorers for the provided prompts.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        show_progress_bars : bool, default=True
            If True, displays progress bars while generating and scoring responses

        Returns
        -------
        UQResult
            UQResult containing data (prompts, responses, and scores) and metadata
        """
        self.prompts = prompts
        self.num_responses = num_responses

        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts=prompts, progress_bar=self.progress_bar)
        sampled_responses = await self.generate_candidate_responses(prompts=prompts, progress_bar=self.progress_bar)
        result = self.score(responses=responses, sampled_responses=sampled_responses, show_progress_bars=show_progress_bars)
        return result

    async def score(self, responses: List[str], sampled_responses: List[List[str]], show_progress_bars: Optional[bool] = True, _display_header: bool = True) -> UQResult:
        """
        Compute confidence scores with specified scorers on provided LLM responses. Should only be used if responses and sampled responses
        are already generated. Otherwise, use `generate_and_score`.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses for the prompts.

        sampled_responses : list of list of str, default=None
            A list of lists of sampled LLM responses for each prompt. These will be used to compute consistency scores by comparing to
            the corresponding response from `responses`.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        UQResult
            UQResult containing data (responses and scores) and metadata
        """
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = len(sampled_responses[0])
        
        self._construct_progress_bar(show_progress_bars)
        self._display_scoring_header(show_progress_bars and _display_header)

        if not set(self.scorers).isdisjoint(CLAIM_BASED_SCORERS):
            self.claim_sets = await decomposer.decompose_claims(responses=responses, progress_bar=self.progress_bar)
        if not set(self.scorers).isdisjoint(SENTENCE_BASED_SCORERS):
            self.sentence_sets = await decomposer.decompose_sentences(responses=responses, progress_bar=self.progress_bar)
            
        self.scores_dict = {k: [] for k in self.scorer_objects}
        for scorer_key, scorer_object in self.scorer_objects.items():
            decomposed_responses = self.claim_sets if scorer_key in CLAIM_BASED_SCORERS else self.sentence_sets
            self.scores_dict[scorer_key] = scorer_object.evaluate(decomposed_responses, sampled_responses=self.sampled_responses, progress_bar=self.progress_bar).to_dict()
        result = self._construct_result()
        self._stop_progress_bar()
        self.progress_bar = None 
        return result

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {"responses": self.responses, "claim_sets": self.claim_sets, "sampled_responses": self.sampled_responses}
        if self.prompts:
            data["prompts"] = self.prompts
        data.update(self.scores_dict)
        result = {"data": data, "metadata": {"temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses, "scorers": self.scorers}}
        return UQResult(result)

    def _validate_scorers(self, scorers: Optional[List[str]]) -> None:
        """Validate scorers"""
        if "luq_atomic" in scorers and "luq" in scorers:
            warnings.warn(
                "Redundant metrics configuration: Using both 'luq' and 'luq_atomic' "
                "increases computation time without providing additional information. "
                "Use only one or the other for substantially faster runtime.",
                UserWarning
            )
        self.scorer_objects = {}
        if scorers is None:
            scorers = self.default_long_form_scorers
        for scorer in scorers:
            if scorer in ["luq", "luq_atomic"]:
                self.scorer_objects[scorer] = LUQScorer(nli_model_name=self.nli_model_name, device=self.device, max_length=self.max_length)
            else:
                raise ValueError(
                    f"""
                    Invalid scorer: {scorer}. Must be one of ["luq", "luq_atomic"]
                    """
                )
        self.scorers = scorers