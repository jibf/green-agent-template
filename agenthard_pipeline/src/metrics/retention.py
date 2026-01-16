"""Retention ratio computation for tracking data preservation through pipeline stages."""

from collections import Counter
from typing import Dict, Set


def compute_retention_ratio(passed_ids: Set, baseline_ids: Set) -> Dict:
    """Compute retention ratio metrics.

    Args:
        passed_ids: Set of question IDs that passed current filter
        baseline_ids: Set of question IDs from initial baseline (for overall retention ratio)

    Returns:
        Dict: {
            "retention_ratio": float,  # overall retention ratio
            "subtask_size": Dict[str, float],  # retention ratio by task type
            "subtask_details": Dict[str, Dict]  # detailed stats for each subtask
        }
    """
    # Overall retention ratio
    overall_retention_ratio = len(passed_ids) / len(baseline_ids) if len(baseline_ids) > 0 else 0.0

    # Task-level retention ratio
    retention_task_types = [question_id.task_name for question_id in passed_ids]
    baseline_task_types = [question_id.task_name for question_id in baseline_ids]

    retention_stat = Counter(retention_task_types)
    baseline_stat = Counter(baseline_task_types)

    subtask_retention = {}
    subtask_details = {}

    for task_type in baseline_stat:
        base_num = baseline_stat[task_type]
        retention_num = retention_stat.get(task_type, 0)
        ratio = retention_num / base_num if base_num > 0 else 0.0

        subtask_retention[task_type] = ratio
        subtask_details[task_type] = {
            "base_num": base_num,
            "retention_num": retention_num,
            "ratio": ratio
        }

    return {
        "retention_ratio": overall_retention_ratio,
        "subtask_size": subtask_retention,
        "subtask_details": subtask_details
    }
