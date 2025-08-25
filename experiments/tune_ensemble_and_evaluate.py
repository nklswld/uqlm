import os
import asyncio
import pandas as pd
from utils import compute_metric_values, extract_scores_from_df, kfold_crossvalidation


async def main():
    # 1. Define relevant variables
    n_trails = 1000  # number of trails for ensemble optimation problem
    current_directory = os.getcwd()
    data_directory = (
        current_directory + "/main_results"
    )  # Define path to data directory
    benchmark_names = ["svamp", "gsm8k", "ai2_arc", "csqa", "nq_open", "popqa"]
    llms = [
        # "gemini_10_pro"
        "gemini_15_flash",
        "gpt_35",
        "gpt_4o",
    ]

    results = {llm_name: {name: {} for name in benchmark_names} for llm_name in llms}
    for name in benchmark_names:
        for llm_name in llms:
            file_path = "{}/{}_{}.parquet".format(data_directory, llm_name, name)
            df = pd.read_parquet(file_path)

            # 2. Implement k-fold cross validation
            # 2a. Tune UQEnsemble on training dataset
            # 2b. Compute metrics (f1-score, precision, recall, rocauc) on hold out set
            x_scores = extract_scores_from_df(df)
            correct_indicators = df["response_correct"].to_list()
            results[llm_name][name]["Ensemble"] = kfold_crossvalidation(
                _score=x_scores,
                _correct_indicators=correct_indicators,
                n_params=len(x_scores),
                n_t=n_trails,
                k=5,
            )

            # 3. Compute metrics for all the other techniques
            results[llm_name][name].update(
                compute_metric_values(correct_indicators, df)
            )

            # 4. Store results in separate file for every pair of llm and benchmark
            store_df = pd.DataFrame(
                results[llm_name][name]
            )  # This dataframe includes method name and metric values
            store_df.to_parquet(
                "{}/{}_{}_metric_values.parquet".format(data_directory, llm_name, name)
            )


asyncio.run(main())
