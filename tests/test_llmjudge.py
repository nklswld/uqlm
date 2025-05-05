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

import pytest
import json
from uqlm.judges import LLMJudge
from langchain_openai import AzureChatOpenAI

datafile_path = "tests/data/scorers/llmjudge_results_file.json"
with open(datafile_path, "r") as f:
    data = json.load(f)

mock_object = AzureChatOpenAI(
    deployment_name="YOUR-DEPLOYMENT",
    temperature=1,
    api_key="SECRET_API_KEY",
    api_version="2024-05-01-preview",
    azure_endpoint="https://mocked.endpoint.com",
)


@pytest.mark.asyncio
async def test_judge(monkeypatch):
    PROMPTS = data["prompts"]
    RESPONSES = data["responses"]

    # Initiate LLMJudge class object
    judge = LLMJudge(llm=mock_object)
    print("data keys: ", data.keys())

    async def mock_generate_responses(*args, **kwargs):
        print(data.keys())
        return {
            "data": {
                "prompt": data["judge_result"]["judge_prompts"],
                "response": data["judge_result"]["judge_responses"],
            }
        }

    monkeypatch.setattr(judge, "generate_responses", mock_generate_responses)

    result = await judge.judge_responses(prompts=PROMPTS, responses=RESPONSES)

    assert result["judge_prompts"] == data["judge_result"]["judge_prompts"]

    extract_answer = judge._extract_answers(
        responses=data["judge_result"]["judge_responses"]
    )
    assert extract_answer == data["extract_answer"]

    assert result["scores"] == data["judge_result"]["scores"]
