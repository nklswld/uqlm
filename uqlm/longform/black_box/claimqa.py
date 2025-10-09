import contextlib
import io
import re
import numpy as np
from typing import List, Optional, Any
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.response_generator import ResponseGenerator
from uqlm.longform.decomposition.response_decomposer import ResponseDecomposer
from uqlm.utils.prompt_templates import get_factoid_template, get_question_template, get_answer_template, get_claim_breakdown_template
from uqlm.utils.results import UQResult
from uqlm.scorers import BlackBoxUQ


class ClaimQAScorer():
    def __init__(self, llm: BaseChatModel, llm_decomposer: BaseChatModel=None, llm_questioner: BaseChatModel=None, black_box_scorers: Optional[List[str]] = None, response_template: str = "atomic", device: Any = None, system_prompt: str="You are a helpful assistant.", max_calls_per_min: int=1000, use_n_param: bool=False, num_factoids: int = 7, num_questions: int = 2, num_claim_qa_responses: int = 2):
        """
        Initialize the ClaimQAScorer.

        Parameters
        ----------
        llm : BaseChatModel
            The original LLM to use for generating responses.
        llm_decomposer : BaseChatModel
            The LLM to use for decomposing the claims.
        llm_questioner : BaseChatModel
            The LLM to use for generating questions.
        response_template : str, default="atomic"
            The template to use for generating responses. Choose from "atomic" or "factoid".
        system_prompt : str
            The system prompt to use for generating responses. Default is "You are a helpful assistant."
        max_calls_per_min : int
            The maximum number of calls per minute to the LLM.
        use_n_param : bool
            Whether to use the n parameter for the LLM.
        num_factoids : int, default=7
            The number of factoids to generate for each response.
        num_questions : int, default=2
            The number of questions to generate for each factoid.
        num_claim_qa_responses : int, default=2
            The number of responses to generate for each claim-inverted question.
        """
        self.llm = llm
        self.llm_decomposer = llm_decomposer if llm_decomposer is not None else llm
        self.llm_questioner = llm_questioner if llm_questioner is not None else self.llm_decomposer
        self.bb_object = BlackBoxUQ(llm=llm, scorers=black_box_scorers, device=device, max_calls_per_min=max_calls_per_min)
        self.system_prompt = system_prompt
        self.max_calls_per_min = max_calls_per_min
        self.use_n_param = use_n_param
        self.num_factoids = num_factoids
        self.num_questions = num_questions
        self.num_claim_qa_responses = num_claim_qa_responses
        if response_template == "atomic":
            self.response_template = get_claim_breakdown_template
        elif response_template == "factoid":
            self.response_template = get_factoid_template
        else:
            raise ValueError("""response_template must be either "atomic" or "factoid".""")

    async def generate_and_score(self, prompts: List[str], progress_bar: Optional[Progress] = None):
        """
        Generate and score the responses.

        Parameters
        ----------
        prompts : List[str]
            A list of prompts to generate responses from LLM.
        progress_bar : Optional[Progress], default=None
            A progress bar to display the progress of the generation.
        """
        self.prompts = prompts
        responses = await self._generate_responses(llm=self.llm, prompts=prompts, count=1, progress_bar=progress_bar)
        self.responses = responses["responses"]
        return await self.score(responses=responses["responses"], progress_bar=progress_bar)

    async def score(self, responses: List[str], progress_bar: Optional[Progress] = None):
        """
        Evaluate the QuesAns scores for a given set of factoids.

        Parameters
        ----------
        responses : List[str]
            A list of responses to be scored. 
        progress_bar : Optional[Progress], default=None
            A progress bar to display the progress of the evaluation.
        """
        # Store responses if not already set
        if not hasattr(self, 'responses'):
            self.responses = responses
        if progress_bar:
            progress_task = progress_bar.add_task("  - Decomposing responses into factoids...", total=len(responses))

        # TODO: Appropriately implement self.num_factoids to generate the number of claims/factoids
        decomposer = ResponseDecomposer(claim_decomposition_llm=self.llm_decomposer, response_template=self.response_template)
        for i, response in enumerate(responses):
            self.factoids = await decomposer.decompose_claims(responses=[response], progress_bar=progress_bar)

        if progress_bar:
            progress_task = progress_bar.add_task("  - Computing ClaimQA Score...", total=len(self.factoids))
        self.response_scores = {key: [] for key in self.bb_object.scorers}
        for i, response in enumerate(responses):
            tmp = await self._compute_factoid_scores(original_prompt=self.prompts[i], original_response=responses[i], factoids=self.factoids[i])
            for key in self.response_scores:
                self.response_scores[key].append(tmp[key])
            if progress_bar:
                progress_bar.update(progress_task, advance=1)
        print("Factoids: ", self.factoids)
        print(" ")
        print("Response-level scores: ", self.response_scores)
        return self._construct_result()

    async def _compute_factoid_scores(self, original_prompt: str, original_response: str, factoids: List[List[str]]) -> List[float]:
        """
        Compute the ClaimQA scores for a given set of factoids.

        Parameters
        ----------
        original_prompt : str
            The original prompt to be scored.
        original_response : str
            The original response to be scored.
        factoids : List[str]
            A list of factoids to be evaluated.

        Returns
        -------
        List[float]
            A dictionary of ClaimQA scores for the given set of factoids.
        """
        claim_qa_scores = {key: [] for key in self.bb_object.scorers}
        print("Factoids for ith response: ", factoids)
        print(" ")
        for factoid_i in factoids[:self.num_factoids]: # TODO: Appropriately implement self.num_factoids to generate the number of claims/factoids
            #Generate questions on each factoid
            create_question_prompt = get_question_template(original_response, factoid_i, self.num_questions)
            print("Prompt: ", create_question_prompt)
            print(" ")
            claim_questions_result = await self._generate_responses(llm=self.llm_questioner, prompts=[create_question_prompt])
            claim_questions = re.split(r"### ", claim_questions_result["responses"][0])[1:]
            print("Questions: ", claim_questions)
            print(" ")
            final_questions = [get_answer_template(original_question=original_prompt, original_response=original_response, claim_question=claim_question) for claim_question in claim_questions]
            print("Final questions: ", final_questions)
            print(" ")
            if len(final_questions) == 0:
                print("No questions generated for factoid: ", factoid_i, " -- returning nan")
                for key in claim_qa_scores:
                    claim_qa_scores[key].append(np.nan)
                continue

            #Generate and score answers on each question
            bb_result = await self.bb_object.generate_and_score(prompts=final_questions, num_responses=self.num_claim_qa_responses, show_progress_bars=False)
            for key in claim_qa_scores:
                claim_qa_scores[key].append(np.mean(bb_result.to_dict()["data"][key]))
        return {key: np.mean(claim_qa_scores[key]) for key in claim_qa_scores}

    async def _generate_responses(self, llm, prompts: List[str], count: int = 1, progress_bar: Optional[Progress] = None) -> List[str]:
        """Helper function to generate responses with LLM.
        
        Parameters
        ----------
        llm : BaseChatModel
            The LLM to use for generating responses.
        prompts : List[str]
            A list of prompts to generate responses from LLM.
        count : int
            The number of responses to generate.
        progress_bar : Optional[Progress], default=None
            A progress bar to display the progress of the generation.

        Returns
        -------
        List[str]
            A list of responses generated by the LLM.
        """
        try:
            generator_object = ResponseGenerator(llm=llm, max_calls_per_min=self.max_calls_per_min, use_n_param=self.use_n_param)
            with contextlib.redirect_stdout(io.StringIO()):
                generations = await generator_object.generate_responses(prompts=prompts, count=count, system_prompt=self.system_prompt, progress_bar=progress_bar)
        except Exception:
            if progress_bar:
                progress_bar.stop()
            raise
        return {"responses": generations["data"]["response"], "logprobs": generations["metadata"]["logprobs"]}

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        prompts = getattr(self, 'prompts', [])
        responses = getattr(self, 'responses', [])
        factoid_scores = getattr(self, 'factoid_scores', [])
        data = {"prompts": prompts, "responses": responses, "factoid_scores": factoid_scores}
        result = {"data": data}
        return UQResult(result)
