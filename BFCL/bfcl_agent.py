from typing import Any
import json
import re
import os
import sys
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
from bfcl_wrapper import BFCLRunner
from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument


class MockArgs:
    """Mock args object required by BFCL's runner."""
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
        Run BFCL evaluation on the purple agent.

        Args:
            message: The incoming message containing EvalRequest
            updater: Report progress and results
        """
        input_text = get_message_text(message)
        from uuid import uuid4
        context_id = message.context_id or uuid4().hex

        self.logger = logging.getLogger("bfcl_evaluator")
        self.logger.setLevel(logging.INFO)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return
        except Exception as e:
            self.logger.error(f"Error parsing request: {e}", exc_info=True)
            await updater.reject(new_agent_text_message(f"Error parsing request: {e}"))
            return

        agent_url = str(request.participants["agent"])
        num_tasks = request.config.get("num_tasks", None)
        debug = request.config.get("debug", False)
        sample_ids = request.config.get("sample_ids", None)
        test_category = request.config.get("test_category", "v3_v4")  # Default to v3_v4 (multi-turn + agentic)
        data_file = request.config.get("data_file", None)
        invalid_tasks_file = request.config.get("invalid_tasks_file", "BFCL/invalid_tasks.txt")

        try:
            if data_file:
                test_data = self._load_test_data(data_file)
            elif sample_ids:
                # sample_ids takes precedence over test_category
                categories = set()
                import re
                for sample_id in sample_ids:
                    if sample_id.startswith('live_'):
                        match = re.match(r'^(live_[^_]+?)_\d', sample_id)
                        category = match.group(1) if match else sample_id
                    elif sample_id.startswith('memory_'):
                        match = re.match(r'^(memory_[^_]+?)(?:_prereq)?_\d', sample_id)
                        if match:
                            category = match.group(1)
                        else:
                            match = re.match(r'^(memory_\w+?)_', sample_id)
                            category = match.group(1) if match else sample_id
                    else:
                        category = sample_id.rsplit('_', 1)[0] if '_' in sample_id else sample_id
                    categories.add(category)

                test_data = []
                for category in categories:
                    category_data = self._load_test_data_by_category(category, context_id)
                    test_data.extend(category_data)
            else:
                # Default: use test_category (which defaults to "all")
                test_data = self._load_test_data_by_category(test_category, context_id)
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}", exc_info=True)
            await updater.reject(new_agent_text_message(f"Failed to load data: {e}"))
            return
        
        # filter out invalid tasks
        # Parse invalid tasks to extract (task_type, idx) pairs
        invalid_patterns = []
        if os.path.exists(invalid_tasks_file):
            with open(invalid_tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Split task_type and idx (e.g., "web_search_63" -> ("web_search", "63"))
                        parts = line.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            task_type, idx = parts
                            invalid_patterns.append((task_type, idx))
                        else:
                            # Fallback: treat as exact match
                            invalid_patterns.append((line, None))

        def is_invalid_task(task_id):
            """Check if task_id matches any invalid pattern."""
            for task_type, idx in invalid_patterns:
                if idx is None:
                    # Exact match
                    if task_id == task_type:
                        return True
                else:
                    task_idx = task_id.rsplit('_')[-1]
                    if task_type in task_id and idx == task_idx:
                        return True
            return False

        test_data = [t for t in test_data if not is_invalid_task(t['id'])]

        if debug:
            test_data = random.sample(test_data, min(10, len(test_data)))
        elif sample_ids:
            test_data = [t for t in test_data if t['id'] in sample_ids]
        elif num_tasks:
            test_data = test_data[:num_tasks]

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Evaluating {len(test_data)} test cases")
        )

        results = []

        args = MockArgs()
        dependencies, task_id_to_index = self._build_dependency_graph(test_data)
        shared_state = {}
        completed_tasks = set()

        for i, task in enumerate(test_data):
            task_id = task['id']

            task_deps = dependencies.get(task_id, set())
            if task_deps and not task_deps.issubset(completed_tasks):
                missing_deps = task_deps - completed_tasks
                self.logger.error(f"Task {task_id} has unmet dependencies: {missing_deps}")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"⚠️  {task_id}: Skipped (unmet dependencies)")
                )
                results.append({
                    "id": task_id,
                    "success": False,
                    "message": {"error_type": "dependency_error", "content": f"Unmet dependencies: {missing_deps}"},
                    "error_type": "dependency_error",
                    "eval_result": None
                })
                continue

            progress_pct = int((i / len(test_data)) * 100)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[{progress_pct}%] {i+1}/{len(test_data)}: {task_id}")
            )

            try:
                runner = BFCLRunner(args, self.logger, self.messenger, agent_url, shared_state=shared_state, context_id=context_id)
                messages, result_dict, success_info = runner.run(task)

                success = result_dict.get("success")
                error_type = result_dict.get("error_type")
                error_message = result_dict.get("error_message")

                result = {
                    "id": task_id,
                    "success": success,
                    "message": error_message if error_message else "Success",
                    "error_type": error_type,
                    "eval_result": result_dict.get("eval_result")
                }

                results.append(result)
                completed_tasks.add(task_id)

                status = "✓" if result['success'] else "✗"
                current_success = sum(1 for r in results if r['success'])
                current_accuracy = (current_success / len(results) * 100) if results else 0
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"{status} {task_id} | {current_accuracy:.1f}% ({current_success}/{len(results)})")
                )

            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                results.append({
                    "id": task_id,
                    "success": False,
                    "message": str(e),
                    "error_type": "runtime_error",
                    "eval_result": None
                })

        summary, result_data = self._calculate_metrics(results)

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary)),
                Part(root=DataPart(data=result_data))
            ],
            name="Result"
        )

        self.messenger.reset()

        try:
            from pathlib import Path
            import shutil
            bfcl_result_path = Path(__file__).parent.parent / "BFCL" / "result"
            context_snapshot_dir = bfcl_result_path / f"remote_a2a_agent_{context_id}"
            if context_snapshot_dir.exists():
                shutil.rmtree(context_snapshot_dir)
        except Exception as e:
            self.logger.warning(f"Failed to clean up snapshots: {e}")

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

    def _load_test_data_by_category(self, test_category: str, context_id: str) -> list:
        """Load test data by category using BFCL's utilities."""
        bfcl_path = os.path.join(os.path.dirname(__file__), '..', 'BFCL', 'bfcl_eval')
        if bfcl_path not in sys.path:
            sys.path.insert(0, bfcl_path)

        categories = parse_test_category_argument([test_category])

        all_test_data = []
        for category in categories:
            test_entries = load_dataset_entry(category, include_language_specific_hint=False)
            all_test_data.extend(test_entries)

        from bfcl_eval.utils import (
            populate_initial_settings_for_web_search_test_cases,
            populate_initial_settings_for_memory_test_cases,
        )
        from pathlib import Path

        all_test_data = populate_initial_settings_for_web_search_test_cases(all_test_data)

        if any("memory" in cat for cat in categories):
            bfcl_result_path = Path(__file__).parent.parent / "BFCL" / "result"
            bfcl_result_path.mkdir(parents=True, exist_ok=True)
            model_result_dir = bfcl_result_path / f"remote_a2a_agent_{context_id}"
            model_result_dir.mkdir(parents=True, exist_ok=True)

            all_test_data = populate_initial_settings_for_memory_test_cases(
                all_test_data, model_result_dir
            )

        return all_test_data

    def _build_dependency_graph(self, test_data: list) -> tuple[dict, dict]:
        """Build dependency graph from test data."""
        dependencies = {}
        task_id_to_index = {}

        for i, task in enumerate(test_data):
            task_id = task['id']
            task_id_to_index[task_id] = i
            dependencies[task_id] = set(task.get('depends_on', []))

        return dependencies, task_id_to_index

    def _calculate_metrics(self, results: list) -> tuple[str, dict]:
        """Calculate evaluation metrics."""
        total_samples = len(results)
        successful_samples = sum(1 for r in results if r['success'])

        category_stats = defaultdict(lambda: {'success': 0, 'total': 0})

        for result in results:
            task_id = result['id']
            category = task_id.rsplit('_', 1)[0] if '_' in task_id else task_id

            category_stats[category]['total'] += 1
            if result['success']:
                category_stats[category]['success'] += 1

        overall_accuracy = (successful_samples / total_samples * 100) if total_samples > 0 else 0

        category_lines = []
        for category, stats in sorted(category_stats.items()):
            accuracy = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            category_lines.append(
                f"  {category}: {accuracy:.1f}% ({stats['success']}/{stats['total']})"
            )

        detailed_task_lines = []
        for r in results:
            status_icon = '✓' if r['success'] else '✗'
            task_line = f"  [{status_icon}] {r['id']}"

            if not r['success']:
                error_type = r.get('error_type', 'unknown')
                if isinstance(r['message'], dict):
                    error_msg = r['message'].get('error', r['message'].get('content', 'Unknown error'))
                else:
                    error_msg = r['message']

                if isinstance(error_msg, list):
                    error_msg = '; '.join(str(e) for e in error_msg[:2])
                error_msg = str(error_msg)[:200]

                task_line += f"\n      Error ({error_type}): {error_msg}"

            detailed_task_lines.append(task_line)

        summary = f"""BFCL Evaluation Results

Overall Accuracy: {overall_accuracy:.1f}% ({successful_samples}/{total_samples})

Category Breakdown:
{chr(10).join(category_lines)}

Detailed Task Results:
{chr(10).join(detailed_task_lines)}
"""

        result_data = {
            "accuracy": overall_accuracy,
            "correct_count": successful_samples,
            "total_count": total_samples,
            "category_stats": dict(category_stats),
            "task_results": [
                {
                    "id": r['id'],
                    "valid": r['success'],
                    "error_type": r.get('error_type'),
                    "error": r['message'] if not r['success'] else None,
                    "eval_result": r.get('eval_result')
                }
                for r in results
            ]
        }

        return summary, result_data
