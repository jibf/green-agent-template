"""BFCL wrapper - Adapter for BFCL benchmark to A2A protocol."""
import re
import sys
import os
import json
import copy
import asyncio
import nest_asyncio

nest_asyncio.apply()

bfcl_path = os.path.join(os.path.dirname(__file__), '..', 'BFCL', 'bfcl_eval')
sys.path.insert(0, bfcl_path)
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response, execute_multi_turn_func_call
from bfcl_eval.constants.enums import Language, ReturnFormat, ModelStyle
from bfcl_eval.utils import is_agentic, is_multi_turn, is_memory_prereq
from bfcl_eval.constants.model_config import ModelConfig, MODEL_CONFIG_MAPPING
from bfcl_eval.model_handler.api_inference.openai_response import OpenAIResponsesHandler
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI

model_config = ModelConfig(
    model_name="remote_a2a_agent",
    display_name="Remote A2A Agent",
    url="",
    org="AgentBeats",
    license="",
    model_handler=OpenAIResponsesHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True
)

if "remote_a2a_agent" not in MODEL_CONFIG_MAPPING:
    MODEL_CONFIG_MAPPING["remote_a2a_agent"] = model_config

if "remote/a2a/agent" not in MODEL_CONFIG_MAPPING:
    MODEL_CONFIG_MAPPING["remote/a2a/agent"] = model_config

BFCL_FC_BASE_INSTRUCTION = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task."""
class RemoteA2AModel:
    """Model wrapper that implements BFCL's interface for A2A agents."""

    def __init__(self, messenger, agent_url: str, logger):
        self.messenger = messenger
        self.agent_url = agent_url
        self.logger = logger
        self._is_first_call = True

    def __call__(self, messages, tools=None):
        if self._is_first_call:
            tools_desc = json.dumps(tools, ensure_ascii=False, indent=2)

            user_query_parts = []
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'system':
                    user_query_parts.append(f"System: {content}")
                elif role == 'user':
                    user_query_parts.append(content)

            user_query = "\n\n".join(user_query_parts) if user_query_parts else ""

            combined_query = f"""{BFCL_FC_BASE_INSTRUCTION}

{user_query}"""

            formatted_message = f"""Available tools:
{tools_desc}

User query: {combined_query}"""
        else:
            latest_msg = messages[-1]

            if latest_msg['role'] == 'observation':
                formatted_message = self._format_observations(messages)
            elif latest_msg['role'] == 'user':
                formatted_message = latest_msg['content']
            else:
                raise ValueError(f"Unexpected message role: {latest_msg['role']}")

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
        return self._parse_response(response_text)

    def _format_observations(self, messages: list) -> str:
        """Format tool observations (results) as simple text."""
        for msg in reversed(messages):
            if msg.get('role') == 'observation':
                observations = msg.get('content', [])
                if not observations:
                    return "No tool results available."

                # Multi-turn tests use string content directly
                if isinstance(observations, str):
                    return observations
                result_lines = []
                for obs in observations:
                    result_lines.append(f"Tool result: {json.dumps(obs, ensure_ascii=False)}")
                return "\n".join(result_lines)

        return "Continue with your next action."

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

                # Mark first call as done AFTER successful call
                if self._is_first_call:
                    self._is_first_call = False

                return response

            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

        raise last_error

    def _parse_response(self, response_text: str):
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
class BFCLRunner:
    """
    Runner for BFCL evaluation using A2A agent.

    This implements the interface expected by our Green Agent,
    similar to ComplexFuncBenchRunner.
    """

    def __init__(self, args, logger, messenger, agent_url: str, shared_state: dict = None, context_id: str = None):
        self.args = args
        self.logger = logger
        self.messenger = messenger
        self.agent_url = agent_url
        # Use context_id to create a unique model name for isolation

        self.context_id = context_id
        self.model_name = f"remote_a2a_agent_{context_id}" if context_id else "remote_a2a_agent"
        self.model = RemoteA2AModel(messenger, agent_url, logger)
        # Shared state for dependent tasks (e.g., MemoryAPI state across prerequisite conversations)
        self.shared_state = shared_state if shared_state is not None else {}

    def run(self, test_case):
        """
        Run a single BFCL test case.

        Following BFCL's logic:
        - Single-turn: Call model once, record response, evaluate (no execution)
        - Multi-turn: Loop with execution until termination condition

        Args:
            test_case: BFCL test case dict with keys:
                - id: test case ID
                - question: list of turns (for multi-turn) or single turn
                - function: list of available functions

        Returns:
            Tuple of (evaluation_messages, result_dict, success_info)
        """

        test_id = test_case['id']
        test_category = self._extract_test_category(test_id)
        # Multi-turn interaction is needed for:
        # 1. Explicit multi_turn tests (multi_turn_base, multi_turn_miss_func, etc.)
        # 2. Agentic tests (web_search, memory) - they need multiple rounds to collect info
        requires_multi_turn = is_multi_turn(test_category) or is_agentic(test_category)

        if requires_multi_turn:
            return self._run_multi_turn(test_case, test_category)
        else:
            return self._run_single_turn(test_case, test_category)

    def _run_single_turn(self, test_case, test_category):
        """
        Run single-turn test following BFCL's inference_single_turn_FC logic.

        Evidence: BFCL/bfcl_eval/model_handler/base_handler.py:685-719
        - Call model once
        - No function execution
        - Return model response for AST evaluation
        """
        evaluation_messages = []

        questions = test_case.get("question", [])
        turn_messages = questions[0] if isinstance(questions[0], list) else questions
        evaluation_messages.extend(turn_messages)

        functions = test_case.get("function", [])
        converted_tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_RESPONSES)
        gpt_functions = [{"type": "function", "function": tool} for tool in converted_tools]

        response = self.model(evaluation_messages, tools=gpt_functions)

        if isinstance(response, dict) and response.get("error_type"):
            return evaluation_messages, response, {"error": True}

        if response.tool_calls:
            if response.content is not None:
                evaluation_messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls
                })
            else:
                evaluation_messages.append({
                    "role": "assistant",
                    "tool_calls": response.tool_calls
                })
        else:
            if response.content:
                evaluation_messages.append({
                    "role": "assistant",
                    "content": response.content
                })

        eval_result = self.evaluate(test_case, evaluation_messages)

        if eval_result.get("valid"):
            result_dict = {
                "success": True,
                "error_message": None,
                "error_type": None,
                "eval_result": eval_result
            }
        else:
            result_dict = {
                "success": False,
                "error_message": eval_result.get("error", "Evaluation failed"),
                "error_type": eval_result.get("error_type", "evaluation_error"),
                "eval_result": eval_result
            }

        return evaluation_messages, result_dict, {"completed": True}

    def _run_multi_turn(self, test_case, test_category):
        MAXIMUM_STEP_LIMIT = 20

        evaluation_messages = []
        current_state = self.shared_state

        functions = test_case.get("function", [])
        converted_tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_RESPONSES)
        gpt_functions = [{"type": "function", "function": tool} for tool in converted_tools]
        questions = test_case.get("question", [])
        turns = questions  # Multi-turn format: list of lists
        for turn_idx, turn_messages in enumerate(turns):

            if not turn_messages:
                self.logger.warning(f"Turn {turn_idx} has empty messages (likely miss_func test)")

                turn_messages = [{
                    "role": "user",
                    "content": "Continue with the available functions."
                }]
            evaluation_messages.extend(turn_messages)

            # Step loop within this turn
            step_count = 0

            while step_count <= MAXIMUM_STEP_LIMIT:

                response = self.model(evaluation_messages, tools=gpt_functions)
                if isinstance(response, dict) and response.get("error_type"):
                    return evaluation_messages, response, {"error": True}
                if response.tool_calls:
                    if response.content is not None:
                        evaluation_messages.append({
                            "role": "assistant",
                            "content": response.content,
                            "tool_calls": response.tool_calls
                        })
                    else:
                        evaluation_messages.append({
                            "role": "assistant",
                            "tool_calls": response.tool_calls
                        })
                    function_calls = []
                    for tool_call in response.tool_calls:
                        function_call = self._parse_tool_call(tool_call)
                        if function_call is None:

                            self.logger.warning("Failed to decode function call, proceeding to next turn")
                            break
                        function_calls.append(function_call)

                    if not function_calls:

                        break

                    # Execute functions
                    execution_results = self._execute_functions(function_calls, test_case, current_state)

                    # execution_results is a list of dicts: [{"name": "find", "result": "..."}, ...]
                    formatted_results_list = []
                    for exec_result in execution_results:
                        func_name = exec_result['name']
                        result_str = exec_result['result']
                        formatted_results_list.append(f"Function: {func_name}\nResult: {result_str}")

                    formatted_results = "\n\n".join(formatted_results_list)
                    # Use "observation" role to match RemoteA2AModel's expected format
                    evaluation_messages.append({
                        "role": "observation",
                        "content": formatted_results
                    })

                    step_count += 1
                    if step_count > MAXIMUM_STEP_LIMIT:
                        self.logger.warning(f"Reached maximum step limit ({MAXIMUM_STEP_LIMIT})")
                        break

                else:
                    # No tool calls - model finished this turn
                    if response.content:
                        evaluation_messages.append({
                            "role": "assistant",
                            "content": response.content
                        })
                    break
        eval_result = self.evaluate(test_case, evaluation_messages)
        if eval_result.get("valid"):
            result_dict = {
                "success": True,
                "error_message": None,
                "error_type": None,
                "eval_result": eval_result  # Include full evaluation result
            }
        else:
            result_dict = {
                "success": False,
                "error_message": eval_result.get("error", "Evaluation failed"),
                "error_type": eval_result.get("error_type", "evaluation_error"),
                "eval_result": eval_result  # Include full evaluation result
            }

        return evaluation_messages, result_dict, {"completed": True}

    def evaluate(self, test_case, evaluation_messages):
        """
        Evaluate the test result using BFCL's original evaluation logic.

        This calls BFCL's ast_checker or multi_turn_checker depending on test type.

        Args:
            test_case: The original test case dict
            evaluation_messages: The conversation history with model responses

        Returns:
            Evaluation result dict with 'valid' key
        """

        test_category = self._extract_test_category(test_case['id'])
        if self._is_multi_turn_test(test_category):
            return self._evaluate_multi_turn(test_case, evaluation_messages)
        else:
            return self._evaluate_single_turn(test_case, evaluation_messages)

    def _evaluate_single_turn(self, test_case, evaluation_messages):
        """Evaluate single-turn test using ast_checker."""

        all_function_calls = []

        for msg in evaluation_messages:
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                for tool_call in msg['tool_calls']:
                    # tool_call might be dict or MockToolCall object
                    if isinstance(tool_call, dict):
                        # Direct dict format
                        func_call = {
                            'name': tool_call.get('function', {}).get('name', ''),
                            'arguments': tool_call.get('function', {}).get('arguments', {})
                        }
                        if isinstance(func_call['arguments'], str):
                            func_call['arguments'] = json.loads(func_call['arguments'])
                    else:
                        # MockToolCall object
                        func_call = self._parse_tool_call(tool_call)

                    if func_call and func_call.get('name'):
                        all_function_calls.append(func_call)

        # Convert to BFCL's expected format: list of dicts
        # Each dict is {"func_name": {args}}
        model_output = []
        for fc in all_function_calls:
            func_name = fc['name']
            func_args = fc['arguments']
            model_output.append({func_name: func_args})
        test_category = self._extract_test_category(test_case['id'])
        if not model_output:

            if "irrelevance" in test_category:
                return {"valid": True}

            elif "relevance" in test_category:
                return {
                    "valid": False,
                    "error": ["Invalid syntax. Failed to decode AST when it should have."],
                    "error_type": "relevance_error:decoder_failed"
                }
            else:
                return {
                    "valid": False,
                    "error": ["No function calls found in model output"],
                    "error_type": "no_function_calls"
                }
        if "irrelevance" in test_category:
            return {
                "valid": False,
                "error": ["Valid syntax. Successfully decode AST when it should not."],
                "error_type": "irrelevance_error:decoder_success"
            }

        # Full AST evaluation

        ground_truth = self._load_ground_truth(test_case)
        if is_memory_prereq(test_category):
            # Memory prerequisite tests don't have ground truth - they're just for state building
            # Mark as valid without evaluation since the goal is just to execute and save state
            self.logger.info(f"Skipping evaluation for memory prerequisite: {test_case['id']}")
            return {
                "valid": True,
                "note": "Memory prerequisite - no evaluation needed"
            }

        if ground_truth is None:
            # No ground truth available - this is an error condition

            if "relevance" in test_category or "irrelevance" in test_category:
                self.logger.warning(f"No ground truth for {test_case['id']} (might be expected for {test_category} tests, but cannot validate)")
            else:

                self.logger.error(f"No ground truth found for {test_case['id']}, cannot validate - marking as failed")
            return {
                "valid": False,
                "error_message": f"No ground truth found for test case {test_case['id']}",
                "error_type": "missing_ground_truth",
                "model_output": model_output
            }
        if is_agentic(test_category):

            # It checks if the expected answer appears in the last assistant message (non-function-call)
            # Find the last assistant message that doesn't contain tool_calls
            last_non_fc_message = None
            for msg in reversed(evaluation_messages):
                if msg.get('role') == 'assistant' and 'tool_calls' not in msg:
                    last_non_fc_message = msg.get('content', '')
                    break

            if last_non_fc_message is None:
                self.logger.error(f"Cannot find the last chat message that is not a function call for {test_case['id']}")
                return {
                    "valid": False,
                    "error_message": "Cannot find the last chat message that is not a function call.",
                    "error_type": "agentic:no_last_message"
                }
            result = agentic_checker(
                model_response=last_non_fc_message,
                possible_answer_list=ground_truth
            )
            return result
        else:

            language = self._extract_language(test_case['id'])
            # Note: Let exceptions propagate - they indicate bugs that should be fixed, not hidden
            result = ast_checker(
                func_description=test_case['function'],
                model_output=model_output,
                possible_answer=ground_truth,
                language=language,
                test_category=test_category,
                model_name="remote_a2a_agent"
            )

            return result

    def _evaluate_multi_turn(self, test_case, evaluation_messages):
        """Evaluate multi-turn test using multi_turn_checker."""
        try:

            # Multi-turn format: list of lists of lists

            model_output_per_turn = []
            current_turn_calls = []
            turn_started = False

            for msg in evaluation_messages:
                if msg.get('role') == 'user':
                    # New turn starts
                    # BFCL always records each turn, even if empty
                    if turn_started:

                        model_output_per_turn.append([current_turn_calls])
                        current_turn_calls = []
                    turn_started = True
                elif msg.get('role') == 'assistant' and 'tool_calls' in msg:

                    for tool_call in msg['tool_calls']:
                        if isinstance(tool_call, dict):
                            func_call = {
                                'name': tool_call.get('function', {}).get('name', ''),
                                'arguments': tool_call.get('function', {}).get('arguments', {})
                            }
                            if isinstance(func_call['arguments'], str):
                                func_call['arguments'] = json.loads(func_call['arguments'])
                        else:
                            func_call = self._parse_tool_call(tool_call)

                        if func_call and func_call.get('name'):

                            func_name = func_call['name']
                            func_args = func_call['arguments']
                            args_str = ", ".join(f"{k}={repr(v)}" for k, v in func_args.items())
                            func_call_str = f"{func_name}({args_str})"
                            current_turn_calls.append(func_call_str)
            if turn_started:
                model_output_per_turn.append([current_turn_calls])
            test_category = self._extract_test_category(test_case['id'])
            if is_memory_prereq(test_category):
                # Memory prerequisite tests don't have ground truth - they're just for state building
                # Mark as valid without evaluation since the goal is just to execute and save state
                self.logger.info(f"Skipping evaluation for memory prerequisite: {test_case['id']}")
                return {
                    "valid": True,
                    "note": "Memory prerequisite - no evaluation needed"
                }
            ground_truth = self._load_ground_truth(test_case)
            if ground_truth is None:
                self.logger.error(f"No ground truth found for {test_case['id']}, cannot validate - marking as failed")
                return {
                    "valid": False,
                    "error_message": f"No ground truth found for test case {test_case['id']}",
                    "error_type": "missing_ground_truth",
                    "model_output": model_output_per_turn
                }
            result = multi_turn_checker(
                multi_turn_model_result_list_decoded=model_output_per_turn,
                multi_turn_ground_truth_list=ground_truth,
                test_entry=test_case,
                test_category=test_category,
                model_name=self.model_name  # Use context-specific model name for isolation
            )

            # Clean up non-serializable objects from result
            # multi_turn_checker may include 'execution_result' which contains Directory objects
            result = self._sanitize_eval_result(result)

            return result

        except Exception as e:
            self.logger.error(f"Multi-turn evaluation failed: {e}", exc_info=True)
            return {
                "valid": False,
                "error": [str(e)],
                "error_type": "evaluation_error"
            }

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

    def _execute_functions(self, function_calls, test_case, current_state):
        """
        Execute function calls using BFCL's execution logic.

        For multi-turn tests only (which have involved_classes field):
            Use BFCL's execute_multi_turn_func_call for real execution

        Evidence: BFCL only executes functions in multi-turn inference
        - inference_single_turn_FC (line 685): No execution
        - inference_multi_turn_FC (line 296): Calls execute_multi_turn_func_call
        """

        has_executable_backend = 'involved_classes' in test_case

        if not has_executable_backend:
            # Single-turn test: Should not reach here
            # Single-turn tests don't execute functions
            self.logger.error("_execute_functions called for non-multi-turn test")
            return []

        # Multi-turn test: Use real execution
        return self._execute_multi_turn_functions(function_calls, test_case, current_state)

    def _is_multi_turn_test(self, test_category: str) -> bool:
        """Check if the test category requires multi-turn execution."""
        multi_turn_categories = [
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
            "multi_turn_composite"
        ]
        return any(category in test_category for category in multi_turn_categories)

    def _execute_multi_turn_functions(self, function_calls, test_case, current_state):
        """
        Execute functions using BFCL's real execution environment.

        This is used for multi-turn tests where execution state matters.
        """
        try:

            func_call_strings = []
            for fc in function_calls:
                func_name = fc['name']
                func_args = fc['arguments']

                # Convert arguments dict to function call string

                args_str = ", ".join(
                    f"{k}={repr(v)}" for k, v in func_args.items()
                )
                func_call_str = f"{func_name}({args_str})"
                func_call_strings.append(func_call_str)
            initial_config = test_case.get('initial_config', {})
            involved_classes = test_case.get('involved_classes', [])
            test_id = test_case['id']
            test_category = self._extract_test_category(test_id)

            # Execute using BFCL's multi-turn execution logic

            execution_results, updated_instances = execute_multi_turn_func_call(
                func_call_list=func_call_strings,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=self.model_name,  # Use context-specific model name for isolation
                test_entry_id=test_id,
                long_context=("long_context" in test_category or "composite" in test_category),
                is_evaL_run=False  # False during agent execution, True during evaluation
            )

            # Update current_state with new instances (for subsequent turns)
            current_state.update(updated_instances)
            # Convert execution results to strings for JSON serialization
            results = []
            for i, (fc, exec_result) in enumerate(zip(function_calls, execution_results)):
                results.append({
                    "name": fc['name'],
                    "result": str(exec_result)  # Convert to string for JSON serialization
                })

            return results

        except Exception as e:
            self.logger.error(f"Multi-turn execution failed: {e}", exc_info=True)
            # Fallback to mock execution
            results = []
            for fc in function_calls:
                results.append({
                    "name": fc['name'],
                    "result": f"Execution error: {str(e)}"
                })
            return results

    def _sanitize_eval_result(self, result):
        """
        Recursively sanitize evaluation result to remove non-serializable objects.

        BFCL's multi_turn_checker may include execution_result which contains
        Directory, File, and other custom objects that can't be JSON serialized.
        """
        if isinstance(result, dict):
            sanitized = {}
            for key, value in result.items():
                if key == 'execution_result':
                    # Convert execution_result to string representation

                    sanitized[key] = str(value)
                else:
                    sanitized[key] = self._sanitize_eval_result(value)
            return sanitized
        elif isinstance(result, list):
            return [self._sanitize_eval_result(item) for item in result]
        elif isinstance(result, (str, int, float, bool, type(None))):
            return result
        else:
            # Convert any other object to string (handles Directory, File, etc.)
            return str(result)

    def _load_ground_truth(self, test_case):
        """Load ground truth for the test case from possible_answer files."""
        test_id = test_case['id']
        test_category = self._extract_test_category(test_id)

        # Prerequisite tests don't have ground truth - they're just for building memory state
        if is_memory_prereq(test_category):
            return None
        bfcl_data_dir = os.path.join(os.path.dirname(__file__), '..', 'BFCL', 'bfcl_eval', 'data')
        possible_answer_dir = os.path.join(bfcl_data_dir, 'possible_answer')

        if "web_search" in test_category:
            # web_search_base, web_search_no_snippet -> web_search
            ground_truth_category = "web_search"
        elif "memory" in test_category:
            # memory_kv, memory_vector, memory_rec_sum -> memory
            ground_truth_category = "memory"
        else:
            ground_truth_category = test_category

        # Try to find the matching ground truth file
        # File name format: BFCL_v4_{category}.json
        ground_truth_file = os.path.join(possible_answer_dir, f"BFCL_v4_{ground_truth_category}.json")

        if not os.path.exists(ground_truth_file):
            self.logger.warning(f"Ground truth file not found: {ground_truth_file}")
            return None
        # but the test_id has been modified by process_web_search_test_case or process_memory_test_case.
        # We need to reverse this transformation.
        ground_truth_test_id = test_id
        if "web_search" in test_category:
            # "web_search_base_0" -> "web_search_0"
            ground_truth_test_id = test_id.replace(test_category, "web_search")
        elif "memory" in test_category:
            # "memory_kv_0-customer-0" -> "memory_0-customer-0"
            ground_truth_test_id = test_id.replace(test_category, "memory")
        try:
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry['id'] == ground_truth_test_id:
                            return entry['ground_truth']

            self.logger.warning(f"Ground truth not found for test ID: {ground_truth_test_id} (original: {test_id})")
            return None

        except Exception as e:
            self.logger.error(f"Failed to load ground truth: {e}")
            return None

    def _extract_test_category(self, test_id: str) -> str:
        """
        Extract test category from test ID.

        Example:
        - "simple_python_0" -> "simple_python"
        - "multiple_0" -> "multiple"
        - "parallel_multiple_5" -> "parallel_multiple"
        - "live_simple_0-0-0" -> "live_simple"
        - "multi_turn_base_0" -> "multi_turn_base"
        """
        if test_id.startswith('live_'):

            # "live_simple_0-0-0" -> "live_simple"
            match = re.match(r'^(live_[^_]+?)_\d', test_id)
            if match:
                return match.group(1)

        # Standard format: remove the numeric suffix (e.g., "_0", "_5")
        return test_id.rsplit('_', 1)[0]

    def _extract_language(self, test_id: str):
        """
        Extract language from test ID.

        Returns Language enum value (PYTHON, JAVA, JAVASCRIPT).
        Default to PYTHON if not specified.
        """
        test_id_lower = test_id.lower()

        if 'java' in test_id_lower and 'javascript' not in test_id_lower:
            return Language.JAVA
        elif 'javascript' in test_id_lower:
            return Language.JAVASCRIPT
        else:
            # Default to Python
            return Language.PYTHON
