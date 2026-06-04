# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import io
import logging
import os
import random
import re

import requests
from openai import OpenAI
from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)
openai_api_key = "EMPTY"
#b4af2-bc2f-0e352e7b17e6/project-f9befcd0-b0a2-4299-b8d6-974ad13595d9/user-fcaffd46-e912-43be-9cb5-1f326be31ef5/vscode/f0e557a9-a89c-41a0-8636-0b0f6d018f1f/c0797186-85ab-4789-bb8b-f7bcbbddbcef/proxy/8000/v1"
openai_api_base = "https://ai-notebook-inspir/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-53bc9e7a-d826-4323-88f8-8526c3b740a0/user-fcaffd46-e912-43be-9cb5-1f326be31ef5/vscode/6eac16bf-e08c-4825-ad68-4fe2af211736/56990e8d-6ded-443c-b5c2-083b951bfb5f/proxy/8002/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = ""


prompt="""### Workflow and Output Format
You must follow these steps in order. Every conversation turn should start from Step 1.
---
### **Step 1: Think**
- **This is the starting point for every turn.**
- Carefully analyze the user's query.
- Break down the problem and formulate a plan.
- **Enclose the entire reasoning process in `<think>...</think>` tags.**
---
### **Step 2: Act (Tool Call)**
- If your plan requires information you don't have, call **one single tool**.
- Enclose the tool call in `<tool_call>...</tool_call>` tags.
- If you can answer without tools, **skip this step** and move directly to Step 4.
---
### **Step 3: Observe and Think Again**
- This is the **critical step** where you analyze the output from the tool call and decide the next action.
- After the tool call, you will receive the tool's output (observation).
- You **MUST** start a new thought process in `<think>...</think>` tags to analyze this output.
- Based on your analysis, decide:
    - **A) Is the information insufficient?**
        → If yes, formulate the next step and go back to **Step 2** to call another tool.
    - **B) Is the information sufficient?**
        → If yes, proceed to **Step 4** to summarize the gathered information.
---
### **Step 4: Summarize**
- **This step is executed only when the decision in Step 3 is to provide the final answer.**
- Summarize all the gathered information using `<summary>...</summary>` tags:
    1. **Extract key points**: Identify and condense the most relevant information from previous steps.
    2. **Eliminate redundancy**: Remove repetitive or unnecessary details to maintain clarity.
    3. **Verify consistency**: Ensure the summary aligns with the original query and avoids hallucinations.
- The summary should be concise and focused, serving as a foundation for the final answer.
- Once the summary is complete, proceed to **Step 5** for the final answer.
---
### **Step 5: Answer**
- **This step is executed after the summary (Step 4) is completed.**
- Formulate the final user-facing answer based on the summary.
- Your answer should not exceed 30 words.
- **Enclose your final answer in `<answer>...</answer>` tags.**

---
Here is the question and image:
<image>
"""


import re
from typing import Optional, Dict, Any


def evaluate_answer_accuracy(
        question_text: str,
        normalized_gt: str,
        normalized_answer: str,
        model_name: str,
        client: Any,
        logger: Any,
        extra_info: Optional[Dict] = None
) -> float:


    judge_prompt = """
    Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
    First, I will give examples of each grade, and then you will grade a new example.

    The following are examples of CORRECT predicted answers.
    ‘‘‘
    Question: What are the names of Barack Obama's children?
    Gold target: Malia Obama and Sasha Obama
    Predicted answer 1: sasha and malia obama
    Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
    Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
    ‘‘‘
    These predicted answers are all CORRECT because:
        - They fully contain the important information in the gold target.
        - They do not contain any information that contradicts the gold target.
        - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
        - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

    The following are examples of INCORRECT predicted answers.
    ‘‘‘
    Question: What are the names of Barack Obama's children?
    Gold target: Malia and Sasha
    Predicted answer 1: Malia.
    Predicted answer 2: Malia, Sasha, and Susan.
    Predicted answer 3: Barack Obama does not have any children.
    Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
    Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
    Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
    Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
    ‘‘‘
    These predicted answers are all INCORRECT because:
        - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.

    The following are examples of NOT_ATTEMPTED predicted answers.
    ‘‘‘
    Question: What are the names of Barack Obama's children?
    Gold target: Malia and Sasha
    Predicted answer 1: I don't know.
    Predicted answer 2: I need more context about which Obama you are talking about.
    Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
    Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
    ‘‘‘
    These predicted answers are all NOT_ATTEMPTED because:
        - The important information in the gold target is not included in the answer.
        - No statements in the answer contradict the gold target.

    Also note the following things:
    - The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
        - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
    - Do not punish predicted answers if they omit information that would be clearly inferred from the question.
        - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include " California".
        - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - Do not give credit for an answer if it contains any internal inconsistency.
        - For example, consider the question: "How many NBA players have scored 60 or more points in a regular season game since 2024?" with the gold answer "8". A response is INCORRECT if it states "8 players" but lists 7 or 9, or if it initially says "8 players" but later contradicts this by concluding 7 or 9.

    Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
    ‘‘‘
    Question: {question}
    Gold target: {correct_answer}
    Predicted answer: {response}
    ‘‘‘
    Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    C: NOT_ATTEMPTED

    Just return the letters "A", "B", or "C", with no text around it.
    """.strip()

    user_prompt = judge_prompt.format(
        question=question_text,
        correct_answer=normalized_gt,
        response=normalized_answer
    )

    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Lower temperature for more deterministic judgement
        )
        response = chat_response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f" [WARNING] Chat completion request failed: {e}")
        return 0.0

    response = response.replace("\\n", "\n").replace("\\r", "\r")


    cleaned = re.sub(r"<think>.*?</think>|<think>|</think>", "", response, flags=re.DOTALL | re.IGNORECASE)

    # 去掉多余空行和空格
    cleaned = cleaned.strip()

    # 根据清理后的响应返回分数
    if cleaned == "A":
        accuracy_score = 1.0
    elif cleaned == "B":
        accuracy_score = 0.0
    elif cleaned == "C":
        accuracy_score = 0.0
    else:
        accuracy_score = 0.0

    return accuracy_score


# print(f"Accuracy score: {accuracy_score}")  # 应该输出 1.0
class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        question = row_dict[self.prompt_key][0]["content"]
        row_dict[self.prompt_key] = [
            {
                "role": "system",
                # We don't need tool description, because custom_chat_template will add it.
                "content": (
                    "You are a helpful assistant. You can call functions to assist with the user query. "
                    "Important: You must call only one function at a time. After each function call, "
                    "wait for the execution result before making the next function call if needed."
                ),
            },
            {
                "role": "user",
                "content": prompt + row_dict[self.prompt_key][0]["content"],
            },
        ]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                # images = [Image.open(io.BytesIO(image["bytes"])) for image in row_dict_images]
                images = Image.open(io.BytesIO(row_dict_images)).convert("RGB")
                images = images.resize((448, 448))
                images = [images]
                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # noqa: E501
                multi_modal_data["image"] = images

            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        if "category" in row_dict:
            if row_dict["category"] is not None:
                row_dict["extra_info"]["category"] = row_dict["category"]
        if "level" in row_dict:
            if row_dict["level"] is not None:
                row_dict["extra_info"]["level"] = row_dict["level"]
        if "data_source" in row_dict:
            if row_dict["data_source"] is not None:
                row_dict["extra_info"]["data_source"] = row_dict["data_source"]
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        row_dict["extra_info"]["question"] = question

        # add index for each prompt
        # 传递给工具的参数
        tools_kwargs = {
            "web_image_to_image_search": {
                "create_kwargs": {"ids": row_dict['data_id']},
                # "execute_kwargs": {},
                # "calc_reward_kwargs": {},
                # "release_kwargs": {},
            }
        }
        print('tools_kwargs')
        print(tools_kwargs)
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict


import re
import math


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    # Initialize tracking variables
    is_format_error = False


    segments = solution_str.split("</tool_response>")


    if len(segments) > 1:

        for segment in segments[1:]:

            cleaned_segment = re.sub(r"^\s*(assistant\s*)?", "", segment, flags=re.IGNORECASE)


            if cleaned_segment and not cleaned_segment.startswith("<think>"):
                is_format_error = True

                break

    # 1. Check <think> tag format
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2 or count_think_2 == 0:
        is_format_error = True

    # 2. Extract answer text with multiple fallback strategies
    answer_text = ""
    predict_no_think = (
        solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
    )

    # Check <answer> tag format
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    # Strategy 1: Try to extract from <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        is_format_error = True
        # Strategy 2: Fallback to content after tool responses
        tool_response_match = re.search(
            r"</tool_response>\s*assistant\s*\n(.*?)$", predict_no_think, re.DOTALL | re.MULTILINE
        )
        if tool_response_match:
            answer_text = tool_response_match.group(1).strip()
        else:
            # Strategy 3: Fallback to content after </think>
            if "</think>" in solution_str:
                remaining_content = predict_no_think
                remaining_content = re.sub(r"<tool_call>.*?</tool_call>", "", remaining_content, flags=re.DOTALL)
                remaining_content = re.sub(r"<tool_response>.*?</tool_response>", "", remaining_content,
                                           flags=re.DOTALL)
                remaining_content = re.sub(r"\b(user|assistant)\b", "", remaining_content)
                answer_text = remaining_content.strip()
            else:
                # Strategy 4: fallback to entire solution
                answer_text = solution_str.strip()

    # Clean up answer text
    answer_text = answer_text.strip()
    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()

    # --- SCORES CALCULATION ---

    # 3. Accuracy Score (Weight: 0.7 in final stage)
    def normalize_text(text):
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.strip().lower())

    normalized_answer = normalize_text(answer_text)
    normalized_gt = normalize_text(ground_truth)

    question_text = extra_info.get("question", "") if extra_info else ""
    accuracy_score = 1.0 if normalized_answer == normalized_gt else 0.0
    accuracy_score = evaluate_answer_accuracy(
        question_text=question_text,
        normalized_gt=normalized_gt,
        normalized_answer=normalized_answer,
        model_name="",
        client=client,
        logger=logger
    )

    # Penalize overlong answers
    if len(answer_text) >= 500:
        accuracy_score = 0.0
        is_format_error = True

    # ------------------------------------
    # 4. Tool Usage Efficiency Score (Smooth Gaussian Reward)
    # ------------------------------------
    tool_calls = re.findall(r"<tool_call>.*?</tool_call>", solution_str, re.DOTALL)
    tool_count = len(tool_calls)

    def gaussian_reward(x, mu, sigma):
        return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    if tool_count > 6:
        tool_efficiency_score = 0.1  #
    else:
        if accuracy_score >= 0.8:
#BGAS parameters need to be updated here
            tool_efficiency_score = gaussian_reward(
                tool_count,
                mu=2, 
                sigma=3  
            )
        else:

            tool_efficiency_score = 0.8 * gaussian_reward(
                tool_count,
                mu=4,
                sigma=1.2
            )

    # ------------------------------------
    # 5. Summary Score
    # ------------------------------------
    summary_score = 0.0
    summary_blocks = re.findall(r"<summary>(.*?)</summary>", solution_str, re.DOTALL)

    if len(summary_blocks) == 1:
        summary_content = summary_blocks[0].strip()
        if summary_content and len(summary_content) >= 10:
            summary_score = 0.9
    elif len(summary_blocks) > 1:
        summary_score = 0.0
        is_format_error = True

    # ------------------------------------
    # 6. Format Score
    # ------------------------------------

    format_score = 0.0 if is_format_error else 1.0

    if is_format_error or not answer_text:
        logger.debug(
            f"Format issue detected:\n"
            f"Solution: {solution_str[:200]}...\n"
            f"Extracted answer: '{answer_text}'\n"
            f"Format error: {is_format_error}\n"
            f"Tool count: {tool_count}\n"
            f"Tool efficiency score: {tool_efficiency_score}\n"
            f"Summary blocks: {len(summary_blocks)}"
        )

    # Determine using full scoring or accuracy-only
    flag = True
    if "category" not in extra_info:
        flag = False

    # Final weighted score calculation
    if flag == False:
        final_score = accuracy_score
    else:
        final_score = (
                0.7 * accuracy_score +
                0.2 * (format_score +summary_score) +
                0.1 * tool_efficiency_score
        )

    return final_score



