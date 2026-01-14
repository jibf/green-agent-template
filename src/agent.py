from typing import Any
import json
import os
import random
import logging
from collections import defaultdict
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message
from dotenv import load_dotenv

load_dotenv()

from messenger import Messenger
from cfbench_wrapper import ComplexFuncBenchRunner
from response_evaluator import ResponseEvaluator


class MockArgs:
    """Mock args object required by ComplexFuncBench's ModelRunner."""
    def __init__(self, eval_model: str = "gpt-4o-2024-08-06"):
        self.eval_model = eval_model


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class Agent:
    required_roles: list[str] = ["agent"]
    required_config_keys: list[str] = []

    def __init__(self):
        self.messenger = Messenger()
        self.logger = None

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Run ComplexFuncBench evaluation on the purple agent.

        Args:
            message: The incoming message containing EvalRequest
            updater: Report progress and results
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        agent_url = str(request.participants["agent"])
        num_tasks = request.config.get("num_tasks", None)
        debug = request.config.get("debug", False)
        sample_ids = request.config.get("sample_ids", None)
        data_file = request.config.get("data_file", "data/ComplexFuncBench.jsonl")
        enable_response_eval = request.config.get("enable_response_eval", True)

        self.logger = logging.getLogger(f"cfbench_evaluator")
        self.logger.setLevel(logging.INFO)

        # Load test data
        try:
            test_data = self._load_test_data(data_file)
        except Exception as e:
            await updater.reject(new_agent_text_message(f"Failed to load data: {e}"))
            return

        # Filter test data
        if debug:
            test_data = random.sample(test_data, min(10, len(test_data)))
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Debug mode: testing {len(test_data)} samples")
            )
        elif sample_ids:
            test_data = [t for t in test_data if t['id'] in sample_ids]
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Testing {len(test_data)} specified samples")
            )
        elif num_tasks:
            test_data = test_data[:num_tasks]
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Testing first {len(test_data)} samples")
            )
        else:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Testing all {len(test_data)} samples")
            )

        results = []

        args = MockArgs()

        # Initialize response evaluator if enabled
        response_evaluator = None
        if enable_response_eval:
            try:
                response_evaluator = ResponseEvaluator(logger=self.logger)
                eval_model = os.getenv("EVAL_MODEL", "gpt-4o-2024-08-06")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Response evaluation enabled (using {eval_model})")
                )
            except ValueError as e:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Warning: Response evaluation disabled - {e}")
                )
                enable_response_eval = False

        for i, task in enumerate(test_data):
            task_id = task['id']

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Task {i+1}/{len(test_data)}: {task_id}")
            )

            try:
                # Create a new runner for each task to ensure clean state
                runner = ComplexFuncBenchRunner(args, self.logger, self.messenger, agent_url)

                # Run single task with ComplexFuncBench runner
                # ComplexFuncBench's run() returns: (messages, result_message, success_turn, correct_count)
                messages, result_message, success_turn, correct_count = runner.run(task)

                # Convert to expected format
                if result_message == "Success.":
                    success = True
                    message = result_message
                    error_type = None
                else:
                    success = False
                    message = result_message
                    error_type = result_message.get('error_type') if isinstance(result_message, dict) else 'unknown_error'

                # Calculate total turns and calls
                total_turn_num = len([m for m in messages if m.get('role') == 'assistant' and 'function_call' in m])
                total_call_num = sum(len(m.get('function_call', [])) for m in messages if m.get('role') == 'assistant')

                result = {
                    "id": task_id,
                    "success": success,
                    "message": message,
                    "count_dict": {
                        "success_turn_num": success_turn,
                        "total_turn_num": total_turn_num,
                        "correct_call_num": correct_count,
                        "total_call_num": total_call_num,
                        "real_turn_num": success_turn
                    },
                    "gen_convs": messages,
                    "final_response": None
                }

                # Extract final response
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant' and 'content' in msg:
                        result['final_response'] = msg['content']
                        break

                # Evaluate response quality if enabled
                if enable_response_eval and response_evaluator and result.get('final_response'):
                    resp_eval = await response_evaluator.evaluate_response(
                        task_data=task,
                        generated_convs=result.get('gen_convs', []),
                        final_response=result.get('final_response')
                    )
                    result['resp_eval'] = resp_eval
                else:
                    result['resp_eval'] = None

                results.append(result)

                # Log progress
                status = "✓" if result['success'] else "✗"
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"{status} Task {task_id}: {message if isinstance(message, str) else message.get('content', 'Error')}")
                )

            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                # Record failure
                results.append({
                    "id": task_id,
                    "success": False,
                    "message": {"error_type": "unknown_error", "content": str(e)},
                    "count_dict": {
                        "success_turn_num": 0,
                        "total_turn_num": 0,
                        "correct_call_num": 0,
                        "total_call_num": 0,
                        "real_turn_num": 0
                    },
                    "resp_eval": None
                })

        # Generate final report
        summary, result_data = self._calculate_metrics(results)

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary)),
                Part(root=DataPart(data=result_data))
            ],
            name="Result"
        )

        # Clean up
        self.messenger.reset()

    def _load_test_data(self, data_file: str) -> list:
        """Load test data from JSONL file."""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        test_data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))

        return test_data

    def _calculate_metrics(self, results: list) -> tuple[str, dict]:
        """Calculate evaluation metrics and generate summary."""
        total_samples = len(results)
        successful_samples = sum(1 for r in results if r['success'])

        # Calculate per-domain statistics
        domain_stats = defaultdict(lambda: {
            'success': 0,
            'total': 0,
            'correct_calls': 0,
            'total_calls': 0
        })

        for result in results:
            # Extract domain from task ID (e.g., "Car-Rental-0" -> "Car-Rental")
            domain = result['id'].rsplit('-', 1)[0]

            domain_stats[domain]['total'] += 1
            if result['success']:
                domain_stats[domain]['success'] += 1

            count_dict = result['count_dict']
            domain_stats[domain]['correct_calls'] += count_dict.get('correct_call_num', 0)
            domain_stats[domain]['total_calls'] += count_dict.get('total_call_num', 0)

        # Calculate overall metrics
        overall_success_rate = (successful_samples / total_samples * 100) if total_samples > 0 else 0

        total_correct_calls = sum(r['count_dict'].get('correct_call_num', 0) for r in results)
        total_all_calls = sum(r['count_dict'].get('total_call_num', 0) for r in results)
        overall_call_accuracy = (total_correct_calls / total_all_calls * 100) if total_all_calls > 0 else 0

        # Calculate response quality metrics (if available)
        completeness_scores = []
        correctness_scores = []
        for result in results:
            resp_eval = result.get('resp_eval')
            if resp_eval:
                complete_score = resp_eval.get('complete', {}).get('score')
                correct_score = resp_eval.get('correct', {}).get('score')

                if complete_score is not None and complete_score >= 0:
                    completeness_scores.append(complete_score)
                if correct_score is not None and correct_score >= 0:
                    correctness_scores.append(correct_score)

        avg_completeness = (sum(completeness_scores) / len(completeness_scores)) if completeness_scores else None
        avg_correctness = (sum(correctness_scores) / len(correctness_scores)) if correctness_scores else None

        # Format summary text
        domain_lines = []
        for domain, stats in sorted(domain_stats.items()):
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            call_acc = (stats['correct_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0
            domain_lines.append(
                f"  {domain}: {success_rate:.1f}% success ({stats['success']}/{stats['total']}), "
                f"{call_acc:.1f}% call accuracy"
            )

        response_quality_section = ""
        if avg_completeness is not None or avg_correctness is not None:
            response_quality_section = "\n\nResponse Quality:"
            if avg_completeness is not None:
                response_quality_section += f"\n  Completeness: {avg_completeness:.2f}/2.0"
            if avg_correctness is not None:
                response_quality_section += f"\n  Correctness: {avg_correctness:.2f}/2.0"

        summary = f"""ComplexFuncBench Evaluation Results

Overall Metrics:
  Success Rate: {overall_success_rate:.1f}% ({successful_samples}/{total_samples})
  Call Accuracy: {overall_call_accuracy:.1f}% ({total_correct_calls}/{total_all_calls}){response_quality_section}

Domain Breakdown:
{chr(10).join(domain_lines)}

Task Results:
{chr(10).join(f"  {r['id']}: {'✓' if r['success'] else '✗'}" for r in results)}
"""

        result_data = {
            "overall_success_rate": overall_success_rate,
            "overall_call_accuracy": overall_call_accuracy,
            "successful_samples": successful_samples,
            "total_samples": total_samples,
            "total_correct_calls": total_correct_calls,
            "total_all_calls": total_all_calls,
            "avg_completeness": avg_completeness,
            "avg_correctness": avg_correctness,
            "domain_stats": dict(domain_stats),
            "task_results": [
                {
                    "id": r['id'],
                    "success": r['success'],
                    "message": r['message'],
                    "count_dict": r['count_dict'],
                    "resp_eval": r.get('resp_eval')
                }
                for r in results
            ]
        }

        return summary, result_data
