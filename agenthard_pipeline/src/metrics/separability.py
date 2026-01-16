"""Separability metric computation using bootstrap confidence intervals."""

import numpy as np
from typing import Dict, List, Callable
from scipy.special import comb

from .utils import bootstrap_confidence_interval


def compute_separability(
    responses_by_question: Dict[str, List[Dict]],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    bootstrap_ci_func: Callable = None
) -> Dict[str, float]:
    """Compute separability metric for each benchmark.

    Separability measures how well the benchmark can distinguish between different models
    based on their performance confidence intervals.

    Args:
        responses_by_question: Dict mapping question IDs to their responses
        n_bootstrap: Number of bootstrap samples for CI computation
        ci: Confidence interval level (e.g., 0.95 for 95% CI)
        bootstrap_ci_func: Optional custom bootstrap CI function (for testing with self._bootstrap_confidence_interval)

    Returns:
        Dict mapping benchmark names to their separability scores [0, 1]
    """
    if bootstrap_ci_func is None:
        bootstrap_ci_func = bootstrap_confidence_interval

    score_dict = {}
    separability_dict = {}

    # Collect scores by benchmark and model
    for responses in responses_by_question.values():
        for sample in responses:
            model_name = sample["model_path"]
            benchmark_name = str(sample["benchmark_name"])
            score = sample["eval_result"]["score"]

            if benchmark_name not in score_dict:
                score_dict[benchmark_name] = {}
            if model_name not in score_dict[benchmark_name]:
                score_dict[benchmark_name][model_name] = []
            score_dict[benchmark_name][model_name].append(score)

    # Calculate separability for each benchmark
    for benchmark in score_dict:
        # Get models with scores
        models_with_scores = sorted(
            [model for model, scores in score_dict[benchmark].items() if scores]
        )

        if len(models_with_scores) < 2:
            separability_dict[benchmark] = (
                1.0 if len(models_with_scores) < 2 else 0.0
            )
            continue

        score_matrix = [
            score_dict[benchmark][model] for model in models_with_scores
        ]
        num_models = len(score_matrix)
        intervals = []

        # Compute confidence intervals for each model
        for i in range(num_models):
            i_ci = bootstrap_ci_func(
                np.array(score_matrix[i]), n_bootstrap=n_bootstrap, ci=ci
            )
            intervals.append(i_ci)
        intervals.sort(key=lambda x: x[0])

        # Count overlapping pairs
        overlapping_pairs = []
        total_pairs = comb(num_models, 2)

        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                # If the start of the second interval is less than the end of the first, they overlap
                if intervals[j][0] < intervals[i][1]:
                    # Check if the pair is already in the list
                    if (intervals[i], intervals[j]) not in overlapping_pairs and (
                        intervals[j],
                        intervals[i],
                    ) not in overlapping_pairs:
                        overlapping_pairs.append((intervals[i], intervals[j]))
                else:
                    break

        # Separability is 1 minus the fraction of overlapping pairs
        separability = (
            1 - len(overlapping_pairs) / total_pairs if total_pairs > 0 else 0
        )
        separability_dict[benchmark] = separability

    return separability_dict
