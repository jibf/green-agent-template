"""
ComplexFuncBench wrapper - Minimal adapter following tau2 integration pattern.

This module creates a wrapper that implements ComplexFuncBench's model interface
while delegating to A2A agents. This allows us to reuse ComplexFuncBench's
original evaluation logic completely.
"""
import sys
import os
import json
import copy
import asyncio
from typing import Any, Optional
import nest_asyncio

nest_asyncio.apply()

cfbench_path = os.path.join(os.path.dirname(__file__), '..', 'ComplexFuncBench')
sys.path.insert(0, cfbench_path)

from runner.base_runner import ModelRunner


class RemoteA2AModel:
    """
    A model wrapper that implements ComplexFuncBench's model interface
    while delegating to a remote A2A agent.

    This is analogous to tau2's RemoteA2AAgent pattern.
    """

    def __init__(self, messenger, agent_url: str, logger):
        self.messenger = messenger
        self.agent_url = agent_url
        self.logger = logger
        self.messages = []  # Track conversation history
        self._is_first_call = True

    def __call__(self, messages, tools=None, **kwargs):
        """
        Model call interface expected by ComplexFuncBench.

        Args:
            messages: Conversation history
            tools: Available tools in OpenAI format

        Returns:
            Mock response object with tool_calls or content
        """
        self.messages = copy.deepcopy(messages)

        # Format message for A2A agent
        if self._is_first_call:
            tools_desc = json.dumps(tools, ensure_ascii=False, indent=2)
            user_query = messages[0]['content'] if messages else ""

            formatted_message = f"""You are a helpful assistant with access to the following tools:

{tools_desc}

To use a tool, respond with a JSON object in this format:
{{
  "tool_calls": [
    {{
      "id": "call_xxx",
      "type": "function",
      "function": {{
        "name": "tool_name",
        "arguments": {{"param": "value"}}
      }}
    }}
  ]
}}

You can call multiple tools at once by including multiple objects in the tool_calls array.

When you're ready to provide a final answer, respond with plain text (not JSON).

User query: {user_query}"""

            self._is_first_call = False
        else:
            # Subsequent calls: send conversation history
            formatted_message = self._format_history(messages, tools)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            response_text = loop.run_until_complete(
                self._call_agent_with_retry(formatted_message)
            )
        except Exception as e:
            self.logger.error(f"Agent call failed: {e}")
            # Return error response
            return self._create_error_response(str(e))

        # Parse response
        return self._parse_response(response_text)

    def _format_history(self, messages: list, tools: list) -> str:
        """Format conversation history for agent."""
        history_str = json.dumps(messages, ensure_ascii=False, indent=2)
        tools_str = json.dumps(tools, ensure_ascii=False, indent=2)

        return f"""Continue the conversation. Available tools:

{tools_str}

Conversation history:
{history_str}

Provide your next action (tool calls or final response)."""

    async def _call_agent_with_retry(self, message: str, max_retries: int = 3) -> str:
        """Call A2A agent with retry logic."""
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.messenger.talk_to_agent(
                    message=message,
                    url=self.agent_url,
                    new_conversation=(attempt == 0 and self._is_first_call)
                )
                return response

            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

        raise last_error

    def _parse_response(self, response_text: str):
        """
        Parse A2A agent response into ComplexFuncBench's expected format.

        Returns a mock object with:
        - .tool_calls: list of tool calls (or None)
        - .content: text content (or None)
        """
        try:
            data = json.loads(response_text)

            # Check if it's tool calls
            if 'tool_calls' in data:
                return MockToolCallResponse(data['tool_calls'])

            # Otherwise treat as content
            return MockContentResponse(response_text)

        except json.JSONDecodeError:
            # Plain text response
            return MockContentResponse(response_text)

    def _create_error_response(self, error_msg: str):
        """Create error response."""
        if 'context length' in error_msg.lower() or 'token limit' in error_msg.lower():
            return {
                "error_type": "context_length_exceeded",
                "error_message": error_msg
            }
        return None


class MockToolCallResponse:
    """
    Mock response object for tool calls.

    Mimics OpenAI's ChatCompletionMessage structure.
    """

    def __init__(self, tool_calls_data: list):
        self.tool_calls = [
            MockToolCall(tc) for tc in tool_calls_data
        ]
        self.content = None


class MockToolCall:
    """Mock tool call object."""

    def __init__(self, data: dict):
        self.id = data.get('id', f"call_{hash(json.dumps(data))}")
        self.type = data.get('type', 'function')
        self.function = MockFunction(data.get('function', {}))


class MockFunction:
    """Mock function object."""

    def __init__(self, data: dict):
        self.name = data.get('name', '')
        self.arguments = json.dumps(data.get('arguments', {})) if isinstance(data.get('arguments'), dict) else data.get('arguments', '{}')


class MockContentResponse:
    """Mock response object for text content."""

    def __init__(self, content: str):
        self.tool_calls = None
        self.content = content


class ComplexFuncBenchRunner(ModelRunner):
    """
    Wrapper for ComplexFuncBench's base runner that uses A2A agent.

    This class extends ModelRunner and overrides the model to use RemoteA2AModel.
    """

    def __init__(self, args, logger, messenger, agent_url: str):
        super().__init__(args, logger)
        self.model = RemoteA2AModel(messenger, agent_url, logger)
        self.model_name = "remote_a2a_agent"

    def run(self, data):
        """
        Run evaluation using ComplexFuncBench's original logic.

        This method is adapted from GPTRunner.run() but uses RemoteA2AModel.
        """
        convs, functions = data['conversations'], data['functions']
        self.CompareClass.add_free_function(convs)

        gpt_functions = [{"type": "function", "function": copy.deepcopy(func)} for func in functions]

        messages = []
        query = convs[0]['content']
        messages.append({"role": "user", "content": query})

        self.init_golden(convs)

        while True:
            llm_response = self.model(messages, tools=gpt_functions)

            if llm_response is None:
                return self.return_result(messages, {"error_type": "unknown_error", "content": "llm_response is None"})

            # Check for context length error
            if isinstance(llm_response, dict) and llm_response.get("error_type") == "context_length_exceeded":
                return self.return_result(messages, {"error_type": "context_length_exceeded", "content": llm_response.get("error_message", "Context length exceeded")})

            if llm_response.tool_calls:
                if self.golden_fcs == []:
                    self.logger.error(f"Output FC:\n{llm_response.tool_calls}")
                    return self.return_result(messages, {"error_type": "func_hallucination", "content": "`self.golden_fcs == []`. Expected to stop. But Model continue to output function call."})

                # Add to messages
                if llm_response.content is not None:
                    self.model.messages.append({"role": "assistant", "content": llm_response.content, "tool_calls": llm_response.tool_calls})
                else:
                    self.model.messages.append({"role": "assistant", "tool_calls": llm_response.tool_calls})

                tool_calls = llm_response.tool_calls

                # Parse function calls
                function_calls = []
                for tool_call in tool_calls:
                    function_call = self._parse_tool_call(tool_call)
                    if function_call is None:
                        return self.return_result(messages, {"error_type": "decode_error", "content": f"{tool_call.function} is not Valid."})
                    function_calls.append(function_call)

                self.logger.info(f"Function Calls: \n{json.dumps(function_calls, ensure_ascii=False, indent=4)}\n")
                self.logger.info(f"Golden Function Call: \n{json.dumps(self.golden_fcs, ensure_ascii=False, indent=4)}\n")

                messages.append({"role": "assistant", "function_call": function_calls})

                self.error_message, success_map, success_matched, format_error = self.CompareClass.compare_turn_prediction(
                    functions, messages[:-1],
                    copy.deepcopy(function_calls), self.golden_fcs,
                    self.golden_obs
                )

                if len(success_map) == 0 and format_error == {}:
                    return self.return_result(messages, self.error_message)

                self.correct_count += len(success_map)

                # Build observations
                real_time_obs = []
                for t, function_call in enumerate(function_calls):
                    if t in success_map:
                        gold_idx = success_map[t]
                        real_time_obs.append(self.golden_obs[gold_idx])
                    else:
                        real_time_obs.append(self.unexpect_call_resp)

                messages.append({"role": "observation", "content": real_time_obs})

                self.process_matches(success_matched)

            else:
                # Final response
                if llm_response.content:
                    messages.append({"role": "assistant", "content": llm_response.content})

                return self.return_result(messages)

    def _parse_tool_call(self, tool_call):
        """Parse tool call from mock object."""
        function_call = {}
        function_call['name'] = tool_call.function.name

        if not function_call['name']:
            return None

        try:
            function_call['arguments'] = json.loads(tool_call.function.arguments)
        except:
            return None

        if function_call['arguments'] is None:
            function_call['arguments'] = {}

        return function_call
