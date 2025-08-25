## Anonymized Code for "Uncertainty Quantification for Language Models: Standardizing and Evaluating Black-Box, White-Box, LLM Judge, and Ensemble Scorers". Submitted to NeurIPS 2025 Workshop *Evaluating the Evolving LLM Lifecycle*.

To replicate our experiments, run the following steps.

#### 1. Install the library
To install the library, navigate to this directory and run the following in the terminal:
```bash
pip install uqlm
```
Note this will install the latest version of our `uqlm` package. To review the source code and supporting documentation, please visit our [anonymized repository](https://anonymous.4open.science/r/uqlm-FBF3).

#### 2. Set up LLM credentials
Running our experiments without code changes requires the following LLMs
- GPT-3.5 instance on Azure
- GPT-4o instance on Azure
- Gemini-1.5-Flash instance on VertexAI

API keys will need to be configured accordingly. If you do not have access to these models and would like to run our experiments with different LLMs, replace the `AzureChatOpenAI` and/or `ChatVertexAI` objects with any [LangChain Chat model](https://js.langchain.com/docs/integrations/chat/) of your choice.

#### 3. Run scripts
Scripts are to be run, from `~/experiments` directory, in this order:
- `generate_and_score.py`
- `tune_ensemble_and_evaluate.py`
- `blackbox_nsamples_experiments.py`