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
import json
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
openai_api_base = ""

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = "judge"
# if openai_api_base:
#     try:
#         response = requests.get(f"{openai_api_base}/models")
#         response.raise_for_status()
#         models = response.json()
#         if models.get("data"):
#             model_name = models["data"][0]["id"]
#         else:
#             logger.warning("No models found at the specified API base for reward scoring.")
#     except (requests.exceptions.RequestException, KeyError, IndexError) as e:
#         logger.warning(f"Failed to get model from {openai_api_base}: {e}. Reward scoring will be disabled.")
#要求回复简短的prompt
prompt_new="""# Workflow and Output Format
You must follow these steps in order. In every conversation turn, you start from Step 1.

**Step 1: Think (Plan)**
* **This is the starting point for every turn.**
* Analyze the user's query and all available information (including previous observations) carefully.
* **Evaluate the query's difficulty and nature.** Determine if the question can be answered *directly* or if it *requires external information* (e.g., facts, real-time data).
* Formulate a plan. Your plan must decide on **one** of two courses of action:
    1.  **Call a tool:** If your evaluation shows you **need more information** (e.g., for complex, factual, or real-time questions).
    2.  **Provide a final answer:** If your evaluation shows you have **sufficient information** (e.g., for simple questions, or tasks that don't require external data).
* Your entire reasoning process must be enclosed in `<think>...</think>` tags.

**Step 2: Act (Tool Call)**
* **Execute this step ONLY if your Step 1 plan was to call a tool.**
* Call the **one single tool** decided upon in your plan.
* The tool call must be enclosed in `<tool_call>...</tool_call>` tags.
* **Important: If you call a tool, you must STOP and wait for the observation. Do NOT proceed to Step 4.**

**Step 3: Observe (Tool Output)**
* **You will only enter this step after a tool call.**
* You will receive the tool's output (observation).
* After receiving the output, you **MUST** go back to **Step 1 (Think)** to analyze the new information and decide the next step (e.g., call another tool or provide the final answer).

**Step 4: Answer (Final Response)**
* **Execute this step ONLY if your Step 1 plan was to provide a final answer.**
* In your **Step 1 Think block**, you must have already synthesized all information and planned the content of your response.
* Formulate the **clear, focused, and concise user-facing final answer** based on that plan.
* Your final answer must be enclosed in `<answer>...</answer>` tags.
Here is the question and image:
<image>"""



#要求回复详细的prompt
prompt="""# Workflow and Output Format
You must follow these steps in order. In every conversation turn, you start from Step 1.

**Step 1: Think (Plan)**
* **This is the starting point for every turn.**
* Analyze the user's query and all available information (including previous observations) carefully.
* Formulate a plan. Your plan must decide on **one** of two courses of action:
    1.  **Call a tool:** If you need more information.
    2.  **Provide a final answer:** If you have sufficient information.
* Your entire reasoning process must be enclosed in `<think>...</think>` tags.

**Step 2: Act (Tool Call)**
* **Execute this step ONLY if your Step 1 plan was to call a tool.**
* Call the **one single tool** decided upon in your plan.
* The tool call must be enclosed in `<tool_call>...</tool_call>` tags.
* **Important: If you call a tool, you must STOP and wait for the observation. Do NOT proceed to Step 4.**

**Step 3: Observe (Tool Output)**
* **You will only enter this step after a tool call.**
* You will receive the tool's output (observation).
* After receiving the output, you **MUST** go back to **Step 1 (Think)** to analyze the new information and decide the next step (e.g., call another tool or provide the final answer).

**Step 4: Answer (Final Response)**
* **Execute this step ONLY if your Step 1 plan was to provide a final answer.**
* In your **Step 1 Think block**, you must have already synthesized all information and planned the content of your response.
* Formulate the **comprehensive, detailed, and user-facing final answer** based on that plan.
* Your final answer must be enclosed in `<answer>...</answer>` tags.
Here is the question and image:
<image>
"""



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
                image = Image.open(io.BytesIO(row_dict_images)).convert("RGB")
                image = image.resize((448, 448))  # ✅ 限制分辨率
                images = [image]
                multi_modal_data["image"] = images

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # noqa: E501


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
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
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
                row_dict["extra_info"]["category"]=row_dict["category"]
        if "level" in row_dict:
            if row_dict["level"] is not None:
                row_dict["extra_info"]["level"]=row_dict["level"]
        if "data_source" in row_dict:
            if row_dict["data_source"] is not None:
                row_dict["extra_info"]["data_source"]=row_dict["data_source"]
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
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute reward score for model solutions with robust handling of various formats.

    The score is a weighted combination of three components, ranging from 0 to 1:
    - Accuracy Score (0.7 weight): Whether the final answer is semantically correct.
    - Tool Score (0.2 weight): Whether the model made the correct decision to use (or not use) a tool.
    - Format Score (0.1 weight): Whether the output follows the expected tag format.
    """

    # Initialize tracking variables
    is_format_error = False

    # 1. Check <think> tag format
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
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
            # Strategy 3: Fallback to content after </think> tag
            if "</think>" in solution_str:
                remaining_content = predict_no_think
                remaining_content = re.sub(r"<tool_call>.*?</tool_call>", "", remaining_content, flags=re.DOTALL)
                remaining_content = re.sub(r"<tool_response>.*?</tool_response>", "", remaining_content, flags=re.DOTALL)
                remaining_content = re.sub(r"\b(user|assistant)\b", "", remaining_content)
                answer_text = remaining_content.strip()
            else:
                # Strategy 4: Use the entire solution as a last resort
                answer_text = solution_str.strip()

    # Clean up answer text
    answer_text = answer_text.strip()
    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()

    # --- SCORES CALCULATION (0-1 range for each component) ---

    # 3. Accuracy Score (Weight: 0.6)
    def normalize_text(text):
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.strip())

    normalized_answer = normalize_text(answer_text)
    normalized_gt = normalize_text(ground_truth)
    question_text = extra_info.get("question", "") if extra_info else ""
    if not client or not model_name:
        logger.warning("Reward function client not initialized or model name not found.")
        return 0.0

    judge_prompt = """
    Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
    First, I will give examples of each grade, and then you will grade a new example.

    The following are examples of CORRECT predicted answers.
    ‘‘‘
    Question: What are the names of Barack Obama’s children?
    Gold target: Malia Obama and Sasha Obama
    Predicted answer 1: sasha and malia obama
    Predicted answer 2: most people would say Malia and Sasha, but I’m not sure and would have to double check
    Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
    ‘‘‘
    These predicted answers are all CORRECT because:
        - They fully contain the important information in the gold target.
        - They do not contain any information that contradicts the gold target.
        - Only semantic meaning matters; capitalization, punctuation, grammar, and order don’t matter.
        - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

    The following are examples of INCORRECT predicted answers.
    ‘‘‘
    Question: What are the names of Barack Obama’s children?
    Gold target: Malia and Sasha
    Predicted answer 1: Malia.
    Predicted answer 2: Malia, Sasha, and Susan.
    Predicted answer 3: Barack Obama does not have any children.
    Predicted answer 4: I think it’s either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
    Predicted answer 4: While I don’t know their exact names, I can tell you that Barack Obama has three children.
    Predicted answer 5: It’s possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
    Predicted answer 6: It may be the case that Obama’s child is named James. However, it’s recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
    ‘‘‘
    These predicted answers are all INCORRECT because:
        - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i’m not sure, i think") are also considered incorrect.

    The following are examples of NOT_ATTEMPTED predicted answers.
    ‘‘‘
    Question: What are the names of Barack Obama’s children?
    Gold target: Malia and Sasha
    Predicted answer 1: I don’t know.
    Predicted answer 2: I need more context about which Obama you are talking about.
    Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
    Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I’m not sure about the other one.
    ‘‘‘
    These predicted answers are all NOT_ATTEMPTED because:
        - The important information in the gold target is not included in the answer.
        - No statements in the answer contradict the gold target.
        
    Also note the following things:
    - The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
        - For example, consider the question "What episode did Derek and Meredith get legally married in Grey’s Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
    - Do not punish predicted answers if they omit information that would be clearly inferred from the question.
        - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include " California".
        - Consider the question "What award did A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL ’24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - Do not give credit for an answer if it contains any internal inconsistency.
        - For example, consider the question: "How many NBA players have scored 60 or more points in a regular season game since 2024?" with the gold answer "8". A response is INCORRECT if it states "8 players" but lists 7 or 9, or if it initially says "8 players" but later contradicts this by concluding 7 or 9.

    Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don’t apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
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
    user_prompt = judge_prompt.format(question=question_text, correct_answer=normalized_gt, response=normalized_answer)
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

    # 移除 <think>...</think>、孤立的 <think>、孤立的 </think>
    cleaned = re.sub(r"<think>.*?</think>|<think>|</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    
    # 去掉多余空行和空格
    cleaned = cleaned.strip()
    max_answer_word_count = 100
    answer_word_count = len(normalized_answer.split()) if normalized_answer else 0

    if cleaned == "A":
        accuracy_score = 1.0
    elif cleaned == "B":
        accuracy_score = 0.0
    elif cleaned == "C":
        accuracy_score = 0.0
    else:
        accuracy_score = 0.0

    if accuracy_score > 0.0 and answer_word_count > max_answer_word_count:
        accuracy_score = 0.0

    # 4. Tool Score (Weight: 0.3) - NEW LOGIC
    tool_call_matches = re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", solution_str, re.DOTALL)
    has_tool_usage = bool(tool_call_matches)
    illegal_tool_used = False

    def _validate_search(arguments):
        if not isinstance(arguments, dict):
            return False
        if set(arguments.keys()) != {"query_list"}:
            return False
        query_list = arguments.get("query_list")
        if not isinstance(query_list, list) or not query_list:
            return False
        return all(isinstance(item, str) and item.strip() for item in query_list)

    def _validate_web_image(arguments):
        if not isinstance(arguments, dict):
            return False
        if set(arguments.keys()) != {"img_idx"}:
            return False
        img_idx = arguments.get("img_idx")
        return isinstance(img_idx, str) and img_idx == "0"

    allowed_tool_validators = {
        "search": _validate_search,
        "web_image_to_image_search": _validate_web_image,
    }

    tool_call_counts = {}

    for tool_payload in tool_call_matches:
        try:
            tool_info = json.loads(tool_payload)
        except json.JSONDecodeError:
            illegal_tool_used = True
            continue
        tool_name = tool_info.get("name")
        if not tool_name or tool_name not in allowed_tool_validators:
            illegal_tool_used = True
            continue
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
        if tool_name == "web_image_to_image_search" and tool_call_counts[tool_name] > 1:
            illegal_tool_used = True
            continue
        arguments = tool_info.get("arguments")
        validator = allowed_tool_validators[tool_name]
        if not validator(arguments):
            illegal_tool_used = True

    flag = True
    tool_score = 0.0 # Default to incorrect decision
    if extra_info and "category" in extra_info:
        if extra_info["category"] == None:
            extra_info["category"] = "search_required"
        if extra_info["category"] == "search_required":
            if has_tool_usage :
                tool_score = 1.0  # Correct decision: Used tool when required
            else:
                tool_score = 0.0  # Incorrect decision: Did not use tool when required
        elif extra_info["category"] == "search_free":
            if not has_tool_usage:
                tool_score = 1.0  # Correct decision: Did not use tool when not needed
            else:
                tool_score = 0.0  # Incorrect decision: Used tool when not needed
    else:
        flag = False

    if illegal_tool_used:
        is_format_error = True

    # 5. Format Score (Weight: 0.1)
    format_score = 0.0 if is_format_error else 1.0
    
    # Log debug information for problematic cases
    if is_format_error or not answer_text:
        logger.debug(
            f"Format issue detected:\n"
            f"Solution: {solution_str[:200]}...\n"
            f"Extracted answer: '{answer_text}'\n"
            f"Format error: {is_format_error}\n"
            f"Tool usage: {has_tool_usage}"
        )

    # Final weighted score calculation
    if flag == False:
        final_score = accuracy_score
    else:
        final_score = (0.7 * accuracy_score) + (0.2 * tool_score) + (0.1 * format_score)

    return final_score




if __name__ == "__main__":
    # Test case 1: Original test case
    predict_str = "The answer is 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {
        "answer": "The woman is to the left of the man who is holding the camera.",
        "id": 0,
        "image": "/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg",
        "pred_ans": "The woman is to the right of the man who is holding the camera.",
        "question": "Is the woman to the left or to the right of the man who is holding the camera?",
    }
    print("=== Test Case 1: Original test ===")
    import time

    time_start = time.time()
    score = compute_score("common_reasoning", predict_str, ground_truth, extra_info)
    print(f"Score: {score}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 2: Problematic case mentioned by user
    problematic_solution = """<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [226, 399, 265, 464], "label": "white van"}}
</tool_call>user
<tool_response>
Zoomed in on the image to the region [226, 399, 265, 464] with label white van.
</tool_response>
assistant
The white van is visible in the lower section of the image, near the diagonal road."""

    problematic_ground_truth = "Yes, the white van is indeed situated in the bottom part of the picture."
    problematic_extra_info = {
        "question": "Is the white van in the bottom part of the picture?",
    }

    print("\n=== Test Case 2: Problematic case (no answer tags) ===")
    print(f"Solution: {problematic_solution}")
    print(f"Ground truth: {problematic_ground_truth}")

    time_start = time.time()
    score2 = compute_score("common_reasoning", problematic_solution, problematic_ground_truth, problematic_extra_info)
    print(f"Score: {score2}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 3: Well-formatted case with tools
    well_formatted_solution = """<think>
I need to use the image zoom tool to get a better look at the specific area.
</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [226, 399, 265, 464], "label": "white van"}}
</tool_call>
<tool_response>
Zoomed in on the image to the region [226, 399, 265, 464] with label white van.
</tool_response>
<answer>Yes, the white van is indeed situated in the bottom part of the picture.</answer>"""

    print("\n=== Test Case 3: Well-formatted case ===")
    time_start = time.time()
    score3 = compute_score(
        "common_reasoning", well_formatted_solution, problematic_ground_truth, problematic_extra_info
    )
    print(f"Score: {score3}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")
 
