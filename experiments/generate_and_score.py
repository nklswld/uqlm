import torch
import asyncio
from dotenv import load_dotenv, find_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI

from uqlm import UQEnsemble
from uqlm.utils import load_example_dataset

from utils import grade_responses


async def main():
    # Set the torch device
    if torch.cuda.is_available():  # NVIDIA GPU
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # macOS
        device = torch.device("mps")
    else:
        device = torch.device("cpu")  # CPU

    # 1. Define a list of benchmarks and number of prompts
    num_prompts = 1000
    num_responses = 15
    benchmark_names = ["svamp", "gsm8k", "ai2_arc", "csqa", "nq_open", "popqa"]
    INSTRUCTION = {
        "svamp": "When you solve this math problem only return the answer with no additional text.\n",
        "gsm8k": "When you solve this math problem only return the answer with no additional text.\n",
        "ai2_arc": "You will be given a multiple choice question. Return only the letter of the response with no additional text or explanation.\n",
        "csqa": "You will be given a multiple choice question. Return only the letter of the response with no additional text or explanation.\n",
        "nq_open": "You will be given a question. Return only the answer as concisely as possible without providing an explanation.\n",
        "popqa": "You will be given a question. Return only the answer as concisely as possible without providing an explanation.\n",
    }

    # 2. Define a list of LLM instances
    gemini_15_flash = ChatVertexAI(model_name="gemini-1.5-flash", temperature=1)

    # gemini_10_pro = ChatVertexAI(model_name="gemini-1.0-pro", temperature=1) # THIS MODEL IS NOW RETIRED. Kept here as a placeholder.

    load_dotenv(find_dotenv())
    gpt_35 = AzureChatOpenAI(
        deployment_name="exai-gpt-35-turbo-16k",
        openai_api_type="azure",
        openai_api_version="2023-07-01-preview",
        temperature=1,
    )

    gpt_4o = AzureChatOpenAI(
        deployment_name="gpt-4o",
        openai_api_type="azure",
        openai_api_version="2024-02-15-preview",
        temperature=1,
    )

    llms = {
        # "gemini_10_pro": gemini_10_pro,
        "gemini_15_flash": gemini_15_flash,
        "gpt_35": gpt_35,
        "gpt_4o": gpt_4o,
    }

    # 3. Define Scorers
    black_box_scorers = [
        "semantic_negentropy",
        "noncontradiction",
        "exact_match",
        "bert_score",
        "cosine_sim",
    ]
    white_box_scorers = [
        "normalized_probability",
        "min_probability",
    ]
    judges = [gemini_15_flash, gpt_35, gpt_4o]

    scorers = {
        # "gemini_10_pro": black_box_scorers + white_box_scorers + judges,
        "gemini_15_flash": black_box_scorers + white_box_scorers + judges,
        "gpt_35": black_box_scorers + judges,
        "gpt_4o": black_box_scorers + white_box_scorers + judges,
    }

    # 4. Generate response and compute score for each scorer.
    for name in benchmark_names:
        data = load_example_dataset(name=name, n=num_prompts)
        prompts = [INSTRUCTION[name] + prompt for prompt in data.question]
        for llm_name in llms:
            uqe = UQEnsemble(
                llm=llms[llm_name],
                device=device,
                max_calls_per_min=175,
                use_n_param=True if llm_name[:3] == "gpt" else False,
                scorers=scorers[llm_name],
            )

            results = await uqe.generate_and_score(
                prompts=prompts,
                num_responses=num_responses,
            )
            result_df = results.to_df()
            result_df["answer"] = data["answer"]
            result_df["response_correct"] = grade_responses(name=name, df=result_df)
            result_df.to_parquet("main_results/{}_{}.parquet".format(llm_name, name))


asyncio.run(main())
