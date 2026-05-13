# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import threading
from openai import OpenAI
from typing import Any, Optional, Union
from PIL import Image

# 基础 URL 配置 (保持不变)
BASE_URL = "https://ai-notebook/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-f9befcd0-b0a2-4299-b8d6-974ad13595d9/user-fcaffd46-e912-43be-9cb5-1f326be31ef5/vscode/277b9574-f6dd-4783-a6f3-4fc8b7f11c1e/d5644e50-18b0-4970-8d62-53cf1c190afb/proxy/8000/v1"

# 线程本地存储，用于管理 OpenAI 客户端实例
thread_local = threading.local()


def get_client():
    """Fetches or creates a thread-local OpenAI client instance."""
    if not hasattr(thread_local, 'client'):
        # Assuming BASE_URL path is correct, and a placeholder api_key is needed
        thread_local.client = OpenAI(
            api_key='fsef',
            base_url=BASE_URL,
        )
    return thread_local.client


def generate_agent_summarize_prompt(message: str, main_query: str,
                                    image: Optional[Union[Image.Image, list[Image.Image]]] = None) -> str:
    """
    Generates a prompt for compressed summarization and calls the model to perform the summary.

    Args:
        message: The text output (i.e., the collected search results).
        main_query: The original query or question.
        image: Optional image object(s). (Parameter retained for function signature matching).

    Returns:
        The summarized text from the model.
    """
    client = get_client()

    # Clean up any image placeholders in the main_query
    if main_query:
        main_query = main_query.replace("<image>", "")

    # **New Compressed Summary Prompt (English Version)**
    summary_prompt = \
        f"""
**Role**: You are a professional and concise information synthesizer.

**Task**: Synthesize a **key findings summary** relevant to the "Question" based on the "Evidence/Reasoning" (i.e., the collection of search results).

**Constraints**: The summary must be **strictly limited to under 50 words**. Do not output any titles, roles, extra separators, or explanatory text. Begin the summary directly.

**Input**:
- Question: '{main_query}'
- Evidence/Reasoning: '{message}'

**Output Format**: Output only a single, concise summary text, not exceeding 50 words.
"""

    # Image handling (kept for structure, but not used by this prompt)
    # print('s1')
    # print('s2')

    # API call to the model
    # Note: Replace model="" with your actual model name, e.g., "qwen2-72b-instruct"
    response = client.chat.completions.create(
        model="",  # Please replace with your actual model name
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": summary_prompt},
                ]
            }
        ],
        max_tokens=8192,  # This is a general safety limit for the whole response
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content


import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
            self,
            messages: list[dict[str, Any]],
            image_data: Any,
            metrics: dict[str, Any],
            request_id: str,
            tools_kwargs: dict[str, Any],
            interaction: Optional[BaseInteraction] = None,
            interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
                agent_data.assistant_turns += 1
            elif state == AgentState.PROCESSING_TOOLS:
                # state = await self._handle_processing_tools_state(agent_data)
                state = await self._handle_processing_tools_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
                agent_data.user_turns += 1
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Check termination conditions
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    # async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
    async def _handle_processing_tools_state(self, agent_data: AgentData,
                                             sampling_params: dict[str, Any]) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        # 1. 确定本次实际执行的 tool_calls (受并行数量限制)
        active_tool_calls = agent_data.tool_calls[: self.max_parallel_calls]
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            # 寻找原始问题
            # print('寻找原始问题')
            # print(agent_data.messages[1])
            # tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
            tasks.append(
                self._call_tool(tool_call, agent_data.tools_kwargs, agent_data.messages, agent_data.image_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Handle responses for interaction if needed
        if self.interaction_config_file:
            for response in responses:
                if response.text:
                    agent_data.messages.append({"role": "tool", "content": response.text})

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        # 2. 使用 zip 同时遍历 "调用请求" 和 "调用结果"，确保一一对应
        for tool_call, tool_response in zip(active_tool_calls, responses):
            # for tool_response in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                content = []
                if tool_response.image:
                    images_to_process = tool_response.image if isinstance(tool_response.image, list) else [
                        tool_response.image]
                    for img in images_to_process:
                        if img is not None:
                            content.append({"type": "image"})

                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

                # --- 优化重点开始 ---
                tool_args = json.loads(tool_call.arguments)
                try:
                    query = tool_args.get("query")
                except Exception as e:
                    logger.error(f"Failed to generate internal summary: {e},{tool_args}")
                    query = ''
                # 只有当有 query 且 response 不为空时才总结
                if query and tool_response.text:
                    # 使用内部方法生成总结，不再请求外部 OpenAI 接口
                    # 这里的 await 是非阻塞 IO，直接由显卡/推理服务处理
                    try:
                        final_summary = await self._generate_internal_summary(
                            query=query,
                            content=tool_response.text,
                            sampling_params=sampling_params
                        )
                        message = {"role": "tool", "content": final_summary or ""}
                    except Exception as e:
                        logger.error(f"Failed to generate internal summary: {e}")

                        message = {"role": "tool", "content": tool_response.text or ""}
                # --- 优化重点结束 ---
                # 3
                tool_args = json.loads(tool_call.arguments)
                query = tool_args.get("query")
                if query:
                    final_summary = generate_agent_summarize_prompt(
                        message=tool_response.text,
                        main_query=query,
                        image=None
                    )
                message = {"role": "tool", "content": final_summary or ""}

            add_messages.append(message)
            agent_data.messages.extend(add_messages)

            # Handle image data
            if tool_response.image:
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]

                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            agent_data.image_data.append(img)
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        agent_data.image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt):]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt):]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _generate_internal_summary(self, query: str, content: str, sampling_params: dict[str, Any]) -> str:
        """
        Internal method to generate summary using the local server_manager.
        """
        # 1. 构建 Prompt (保持你原有的 Prompt 逻辑)
        if query:
            query = query.replace("<image>", "")

        summary_prompt = \
            f"""
**Role**: You are a professional and concise information synthesizer.

**Task**: Synthesize a **key findings summary** relevant to the "Question" based on the "Evidence/Reasoning" (i.e., the collection of search results).

**Constraints**: The summary must be **strictly limited to under 50 words**. Do not output any titles, roles, extra separators, or explanatory text. Begin the summary directly.

**Input**:
- Question: '{query}'
- Evidence/Reasoning: '{content}'

**Output Format**: Output only a single, concise summary text, not exceeding 50 words.
"""
        # 构造对话格式
        summary_messages = [{"role": "user", "content": summary_prompt}]

        # 2. Tokenize (复用类中已有的逻辑)
        if self.processor is not None:
            # 如果是多模态模型处理器
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    summary_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # 总结纯文本不需要传图片，传入 images=None 以节省资源
            model_inputs = self.processor(text=[raw_prompt], images=None, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            # 标准 Tokenizer
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    summary_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        # 3. 调整采样参数 (为了总结，通常希望确定性强一些，且长度受限)
        # 复制一份参数避免影响主流程
        summary_sampling_params = sampling_params.copy()
        # summary_sampling_params['max_tokens'] = 256 # 限制生成的最大 token 数，防止废话

        # 4. 调用 Server Manager 生成 (复用 generate_sequences 的逻辑)
        # 生成一个临时的 request_id
        temp_request_id = f"summary_{uuid4().hex}"

        output = await self.server_manager.generate(
            request_id=temp_request_id,
            prompt_ids=prompt_ids,
            sampling_params=summary_sampling_params,
            image_data=None,  # 总结任务不传图片
        )

        # 5. Decode 输出
        generated_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        print('输出解码')
        print(generated_text.strip())
        return generated_text.strip()

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], message, image) -> ToolResponse:
        # async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            # instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            # 注意其它工具create函数都要加
            instance_id, _ = await tool.create(message, image, create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(
                text=f"Error when executing tool: {e}",
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs)

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
