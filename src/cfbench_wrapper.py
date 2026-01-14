"""
ComplexFuncBench wrapper - Minimal adapter following tau2 integration pattern.

This module creates a wrapper that implements ComplexFuncBench's model interface
while delegating to A2A agents. This allows us to reuse ComplexFuncBench's
original evaluation logic completely.
"""
import sys
import re
import os
import json
import copy
import asyncio
import nest_asyncio

nest_asyncio.apply()

cfbench_path = os.path.join(os.path.dirname(__file__), '..', 'ComplexFuncBench')
sys.path.insert(0, cfbench_path)

# Change working directory to ComplexFuncBench
# This is required because ComplexFuncBench uses relative paths like "utils/tool_info.json"
_original_cwd = os.getcwd()
os.chdir(cfbench_path)

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

    def __call__(self, messages, tools=None):
        """
        Model call interface expected by ComplexFuncBench.

        Args:
            messages: Conversation history
            tools: Available tools in OpenAI format

        Returns:
            Mock response object with tool_calls or content
        """
        self.messages = copy.deepcopy(messages)

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

            if 'tool_calls' in data:
                return MockToolCallResponse(data['tool_calls'])

            if 'response' in data and isinstance(data['response'], str):
                tool_calls = self._extract_tool_calls_from_text(data['response'])
                if tool_calls:
                    return MockToolCallResponse(tool_calls)
            else:
                tool_calls = self._extract_tool_calls_from_text(response_text)
                if tool_calls:
                    return MockToolCallResponse(tool_calls)

            return MockContentResponse(response_text)

        except json.JSONDecodeError:
            tool_calls = self._extract_tool_calls_from_text(response_text)
            if tool_calls:
                return MockToolCallResponse(tool_calls)
            return MockContentResponse(response_text)

    def _extract_tool_calls_from_text(self, text: str) -> list | None:
        """Extract tool calls from text containing ```json code blocks."""
        tool_calls = []

        json_blocks = re.findall(r'```json\s*\n(.*?)\n```', text, re.DOTALL)

        for block in json_blocks:
            try:
                data = json.loads(block)

                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if item.get('type') == 'function' and 'function' in item:
                                tool_calls.append(item)
                            elif 'name' in item and 'arguments' in item:
                                tool_calls.append({
                                    "type": "function",
                                    "function": {
                                        "name": item['name'],
                                        "arguments": item['arguments']
                                    }
                                })

                elif isinstance(data, dict):
                    if data.get('type') == 'function' and 'function' in data:
                        tool_calls.append(data)
                    elif 'name' in data and 'arguments' in data:
                        tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": data['name'],
                                "arguments": data['arguments']
                            }
                        })
            except json.JSONDecodeError:
                continue

        if not tool_calls:
            potential_jsons = re.findall(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}', text, re.DOTALL)
            for json_str in potential_jsons:
                try:
                    data = json.loads(json_str)
                    if data.get('type') == 'function' and 'function' in data:
                        tool_calls.append(data)
                    elif 'name' in data and 'arguments' in data:
                        tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": data['name'],
                                "arguments": data['arguments']
                            }
                        })
                except json.JSONDecodeError:
                    continue

        return tool_calls if tool_calls else None

    def _create_error_response(self, error_msg: str):
        """Create error response."""
        if 'context length' in error_msg.lower() or 'token limit' in error_msg.lower():
            return {
                "error_type": "context_length_exceeded",
                "error_message": error_msg
            }
        return None


class MockToolCallResponse:
    """Mock response object for tool calls."""

    def __init__(self, tool_calls_data: list):
        self.tool_calls = [MockToolCall(tc) for tc in tool_calls_data]
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
    """Wrapper for ComplexFuncBench's base runner that uses A2A agent."""

    def __init__(self, args, logger, messenger, agent_url: str):
        super().__init__(args, logger)
        self.model = RemoteA2AModel(messenger, agent_url, logger)
        self.model_name = "remote_a2a_agent"

    def run(self, data):
        """Run evaluation using ComplexFuncBench's original logic."""
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
                    return self.return_result(messages, {"error_type": "func_hallucination", "content": "Expected to stop but model continued outputting function calls."})

                if llm_response.content is not None:
                    self.model.messages.append({"role": "assistant", "content": llm_response.content, "tool_calls": llm_response.tool_calls})
                else:
                    self.model.messages.append({"role": "assistant", "tool_calls": llm_response.tool_calls})

                tool_calls = llm_response.tool_calls

                function_calls = []
                for tool_call in tool_calls:
                    function_call = self._parse_tool_call(tool_call)
                    if function_call is None:
                        return self.return_result(messages, {"error_type": "decode_error", "content": f"{tool_call.function} is not valid."})
                    function_calls.append(function_call)

                messages.append({"role": "assistant", "function_call": function_calls})

                self.error_message, success_map, success_matched, format_error = self.CompareClass.compare_turn_prediction(
                    functions, messages[:-1],
                    copy.deepcopy(function_calls), self.golden_fcs,
                    self.golden_obs
                )

                if len(success_map) == 0 and format_error == {}:
                    return self.return_result(messages, self.error_message)

                self.correct_count += len(success_map)

                real_time_obs = []
                for t, function_call in enumerate(function_calls):
                    if t in success_map:
                        real_time_obs.append(success_map[t])
                    elif t in format_error:
                        real_time_obs.append(format_error[t])
                    else:
                        real_time_obs.append(self.unexpect_call_resp)

                messages.append({"role": "observation", "content": real_time_obs})
                self.process_matches(success_matched)

            else:
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
        except json.JSONDecodeError:
            return None

        if function_call['arguments'] is None:
            function_call['arguments'] = {}

        return function_call
