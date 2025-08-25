import os
import ast
import random
import torch
import pandas as pd
import numpy as np
from uqlm import BlackBoxUQ


def main():
    device = torch.device("cuda")

    benchmark_names = ["svamp", "gsm8k", "ai2_arc", "csqa", "nq_open", "popqa"]

    model_names = [
        # "gemini_10_pro"
        "gemini_15_flash",
        "gpt_35",
        "gpt_4o",
    ]

    scorers = [
        "noncontradiction",
        "semantic_negentropy",
        "cosine_sim",
        "exact_match",
        "bert_score",
    ]

    values_of_m = [1, 3, 5, 10, 15]

    def run_num_responses_experiment(read_path, write_path):
        """Execute responses to measure"""
        read_data = pd.read_parquet(read_path)
        write_data = read_data.copy()[["prompt", "response", "sampled_responses"]]
        responses = write_data.response.to_list()
        sampled_responses_raw = write_data.sampled_responses.to_list()
        sampled_responses_raw = [
            random.sample(list(sr), max(values_of_m)) for sr in sampled_responses_raw
        ]

        for num_responses in values_of_m:
            if isinstance(
                sampled_responses_raw[0], np.ndarray
            ):  # handle different data types that parquet selected
                sampled_responses = [
                    (list(sr))[0:num_responses] for sr in sampled_responses_raw
                ]
            elif isinstance(sampled_responses_raw[0], str):
                sampled_responses = [
                    list(ast.literal_eval(sr))[0:num_responses]
                    for sr in sampled_responses_raw
                ]
            elif isinstance(sampled_responses_raw[0], list):
                sampled_responses = [
                    (list(sr))[0:num_responses] for sr in sampled_responses_raw
                ]

            bbuq = BlackBoxUQ(scorers=scorers, device=device, use_best=False)
            result = bbuq.score(
                responses=responses, sampled_responses=sampled_responses
            )
            result_df = result.to_df()

            for col in scorers:
                write_data[col + str(num_responses)] = result_df[col]
        write_data.to_parquet(write_path)

    num_results_dict = {}
    current_path = os.getcwd()
    for benchmark_name in benchmark_names:
        num_results_dict[benchmark_name] = {}
        for model_name in model_names:
            read_path = os.path.join(
                current_path, f"main_results/{model_name}_{benchmark_name}.parquet"
            )
            write_path = os.path.join(
                current_path,
                f"num_responses_results/{model_name}_{benchmark_name}.parquet",
            )
            run_num_responses_experiment(read_path, write_path)


if __name__ == "__main__":
    main()
