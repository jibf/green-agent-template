"""Utility functions for metrics computation."""

import random
import numpy as np
from typing import Dict, List, Tuple


def bootstrap_confidence_interval(
    scores: np.ndarray, n_bootstrap: int = 100, ci: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for scores.

    Args:
        scores: Array of scores
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    if len(scores) == 0:
        return (0, 0)
    if len(scores) == 1:
        return (scores[0], scores[0])

    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def create_task_wise_baseline_sample_set(
    responses_by_question: Dict[str, List[Dict]],
    target_task_count: int,
    logger=None
) -> Dict[str, List[Dict]]:
    """Create a baseline sample set by randomly sampling N tasks from original samples.

    Args:
        responses_by_question: Dict mapping question IDs to their responses
        target_task_count: Number of tasks to sample
        logger: Optional logger for logging info

    Returns:
        Dict mapping sampled task IDs to their responses
    """
    total_tasks = len(responses_by_question)
    total_samples = sum(
        len(responses) for responses in responses_by_question.values()
    )

    if logger:
        logger.info(
            f"Creating task-wise baseline: sampling {target_task_count} tasks from {total_samples} total samples ({total_tasks} unique tasks)"
        )
    print("Total tasks for BASELINE: ", target_task_count)

    if target_task_count >= total_tasks:
        if logger:
            logger.info("Target task count >= total tasks, returning all samples")
        return responses_by_question.copy()

    # Random sample with fixed seed for reproducibility
    random.seed(42)
    selected_task_ids = random.sample(
        list(responses_by_question.keys()), target_task_count
    )
    random.seed()  # Reset seed

    # Create baseline using dict comprehension
    baseline = {
        task_id: responses_by_question[task_id] for task_id in selected_task_ids
    }

    baseline_samples = sum(len(responses) for responses in baseline.values())
    if logger:
        logger.info(
            f"Task-wise baseline created: {baseline_samples} samples from {len(selected_task_ids)} tasks"
        )
    return baseline


def calculate_model_ranking(
    responses_by_question: Dict[str, List[Dict]]
) -> Tuple[Dict[str, List[str]], Dict]:
    """Calculate model ranking and detailed performance data.

    Args:
        responses_by_question: Dict mapping question IDs to their responses

    Returns:
        tuple: (model_ranking_dict, model_performance_dict)
            - model_ranking_dict: Dict[benchmark_name, List[model_name]] sorted by performance
            - model_performance_dict: Dict with overall and subtask performance data
    """
    benchmark_data = {}
    task_data = {}  # For subtask-level performance

    for responses in responses_by_question.values():
        for sample in responses:
            benchmark_name = str(sample["benchmark_name"])
            model_name = sample["model_path"]
            task_name = sample.get("task_name", "unknown")
            score = sample["eval_result"]["score"]

            # Benchmark-level data
            if benchmark_name not in benchmark_data:
                benchmark_data[benchmark_name] = {}
            if model_name not in benchmark_data[benchmark_name]:
                benchmark_data[benchmark_name][model_name] = []
            benchmark_data[benchmark_name][model_name].append(score)

            # Task-level data
            if benchmark_name not in task_data:
                task_data[benchmark_name] = {}
            if task_name not in task_data[benchmark_name]:
                task_data[benchmark_name][task_name] = {}
            if model_name not in task_data[benchmark_name][task_name]:
                task_data[benchmark_name][task_name][model_name] = []
            task_data[benchmark_name][task_name][model_name].append(score)

    # Calculate ranking for each benchmark
    model_ranking = {}
    model_performance = {}

    for benchmark_name, model_scores in benchmark_data.items():
        model_means = {}
        for model_name, scores in model_scores.items():
            model_means[model_name] = np.mean(scores)

        # Sort by performance (descending)
        model_ranking[benchmark_name] = sorted(
            model_means.keys(), key=lambda x: model_means[x], reverse=True
        )

        # Store performance data
        model_performance[benchmark_name] = {}
        for model_name, avg_score in model_means.items():
            model_performance[benchmark_name][model_name] = {
                "overall_score": float(avg_score),
                "subtask_scores": {}
            }

            # Add subtask scores
            if benchmark_name in task_data:
                for task_name, task_models in task_data[benchmark_name].items():
                    if model_name in task_models:
                        task_avg = np.mean(task_models[model_name])
                        model_performance[benchmark_name][model_name]["subtask_scores"][task_name] = float(task_avg)

    return model_ranking, model_performance
