"""Visualization functions for model performance and agreement metrics."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Callable

from .utils import bootstrap_confidence_interval


def visualize_model_performance(
    responses_by_question: Dict[str, List[Dict]],
    title: str,
    filename: str,
    n_bootstrap: int = 100,
    ci: float = 0.95,
    model_ranking: Dict[str, List[str]] = None,
    bootstrap_ci_func: Callable = None,
    logger=None,
    agreement_heatmap_callback: Callable = None
):
    """Create bar graph visualization of model-wise performance with confidence intervals.

    Args:
        responses_by_question: Dict mapping question IDs to their responses
        title: Title for the plot
        filename: Base filename for saving plots
        n_bootstrap: Number of bootstrap samples for CI computation
        ci: Confidence interval level (e.g., 0.95 for 95% CI)
        model_ranking: Optional pre-computed model ranking dict
        bootstrap_ci_func: Optional custom bootstrap CI function (for compatibility with self._bootstrap_confidence_interval)
        logger: Optional logger for logging info
        agreement_heatmap_callback: Optional callback function to create agreement heatmap

    Returns:
        Dict containing agreement statistics by benchmark
    """
    if bootstrap_ci_func is None:
        bootstrap_ci_func = bootstrap_confidence_interval

    if logger:
        logger.info(f"Creating visualization: {title}")

    # Create output directory for plots
    plots_dir = "pipeline_results/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Group samples by benchmark
    benchmark_data = {}
    for responses in responses_by_question.values():
        for sample in responses:
            benchmark_name = str(sample["benchmark_name"])
            model_name = sample["model_path"]
            score = sample["eval_result"]["score"]

            if benchmark_name not in benchmark_data:
                benchmark_data[benchmark_name] = {}
            if model_name not in benchmark_data[benchmark_name]:
                benchmark_data[benchmark_name][model_name] = []
            benchmark_data[benchmark_name][model_name].append(score)

    # Store agreement stats for return
    agreement_stats = {}

    # Create separate plots for each benchmark
    for benchmark_name, model_scores in benchmark_data.items():
        # Determine model ordering
        if model_ranking and benchmark_name in model_ranking:
            # Use provided ranking, but only include models that exist in current data
            available_models = set(model_scores.keys())
            ordered_models = [
                model
                for model in model_ranking[benchmark_name]
                if model in available_models
            ]
            # Add any models not in ranking at the end
            remaining_models = sorted(available_models - set(ordered_models))
            model_order = ordered_models + remaining_models
        else:
            # Calculate ranking based on current data (for original dataset)
            model_means = {}
            for model_name, scores in model_scores.items():
                model_means[model_name] = np.mean(scores)
            model_order = sorted(
                model_means.keys(), key=lambda x: model_means[x], reverse=True
            )

        # Calculate means and confidence intervals for each model in the determined order
        model_names = []
        means = []
        ci_lowers = []
        ci_uppers = []

        for model_name in model_order:
            scores = np.array(model_scores[model_name])
            mean_score = np.mean(scores)
            ci_lower, ci_upper = bootstrap_ci_func(
                scores, n_bootstrap, ci
            )

            model_names.append(model_name)
            means.append(mean_score)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)

        # Create the plot with smaller figure size
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(model_names))

        # Create bars with error bars
        bars = plt.bar(
            x_pos,
            means,
            yerr=[
                np.array(means) - np.array(ci_lowers),
                np.array(ci_uppers) - np.array(means),
            ],
            capsize=3,
            alpha=0.7,
            color="skyblue",
            edgecolor="navy",
            linewidth=0.8,
        )

        # Customize the plot
        plt.xlabel("Model", fontsize=10, fontweight="bold")
        plt.ylabel("Performance Score", fontsize=10, fontweight="bold")
        plt.title(
            f"{title} - {benchmark_name}\nModel Performance with {int(ci * 100)}% Confidence Intervals",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )

        # Set x-axis labels
        plt.xticks(x_pos, model_names, rotation=45, ha="right", fontsize=9)

        # Add value labels on top of bars (smaller font)
        for bar, mean, ci_low, ci_high in zip(bars, means, ci_lowers, ci_uppers):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ci_high - mean) + 0.005,
                f"{mean:.3f}\n[{ci_low:.3f}, {ci_high:.3f}]",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis="y")

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the plot with reduced DPI
        safe_benchmark_name = benchmark_name.replace("/", "_").replace(" ", "_")
        plot_filename = os.path.join(
            plots_dir, f"{filename}_{safe_benchmark_name}.png"
        )
        plt.savefig(
            plot_filename,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            format="png",
        )
        plt.close()

        if logger:
            logger.info(f"Saved plot: {plot_filename}")

        # Create model agreement heatmap using callback if provided
        if agreement_heatmap_callback:
            heatmap_stats = agreement_heatmap_callback(
                responses_by_question, benchmark_name, title, filename, model_order
            )
            if heatmap_stats:
                agreement_stats[benchmark_name] = heatmap_stats

    return agreement_stats


def create_model_agreement_heatmap(
    responses_by_question: Dict[str, List[Dict]],
    benchmark_name: str,
    title: str,
    filename: str,
    model_order: List[str],
    logger=None
) -> Dict:
    """Create a heatmap showing model agreement based on task correctness.

    Args:
        responses_by_question: Dict mapping question IDs to their responses
        benchmark_name: Name of the benchmark to create heatmap for
        title: Title for the heatmap
        filename: Base filename for saving the heatmap
        model_order: Order of models for the heatmap axes
        logger: Optional logger for logging info

    Returns:
        Dict containing agreement statistics (avg, min, max, num_models)
    """
    if logger:
        logger.info(f"Creating model agreement heatmap for {benchmark_name}")

    # Create output directory for plots
    plots_dir = "pipeline_results/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Group responses by question ID and model
    question_model_scores = {}
    for question_id, responses in responses_by_question.items():
        for response in responses:
            if str(response["benchmark_name"]) == benchmark_name:
                model_name = response["model_path"]
                score = response["eval_result"]["score"]

                if question_id not in question_model_scores:
                    question_model_scores[question_id] = {}
                question_model_scores[question_id][model_name] = score

    # Filter to only include models that are in model_order and have data
    available_models = set()
    for question_id, model_scores in question_model_scores.items():
        available_models.update(model_scores.keys())

    # Use only models that exist in both model_order and available_models
    filtered_model_order = [
        model for model in model_order if model in available_models
    ]

    if len(filtered_model_order) < 2:
        if logger:
            logger.warning(
                f"Not enough models ({len(filtered_model_order)}) to create agreement heatmap for {benchmark_name}"
            )
        return None

    # Calculate agreement matrix
    n_models = len(filtered_model_order)
    agreement_matrix = np.zeros((n_models, n_models))

    # For each pair of models, calculate agreement
    for i, model1 in enumerate(filtered_model_order):
        for j, model2 in enumerate(filtered_model_order):
            if i == j:
                agreement_matrix[i, j] = 1.0  # Perfect agreement with self
            else:
                agreements = 0
                total_comparisons = 0

                # Compare on each question where both models have responses
                for question_id, model_scores in question_model_scores.items():
                    if model1 in model_scores and model2 in model_scores:
                        score1 = model_scores[model1]
                        score2 = model_scores[model2]

                        # Convert scores to binary correctness (assuming score > 0.5 means correct)
                        correct1 = 1 if score1 > 0.5 else 0
                        correct2 = 1 if score2 > 0.5 else 0

                        # Agreement if both correct or both incorrect
                        if correct1 == correct2:
                            agreements += 1
                        total_comparisons += 1

                if total_comparisons > 0:
                    agreement_matrix[i, j] = agreements / total_comparisons
                else:
                    agreement_matrix[i, j] = 0.0

    # Create the heatmap
    plt.figure(figsize=(max(8, n_models * 0.8), max(6, n_models * 0.6)))

    # Create heatmap with custom colormap
    sns.heatmap(
        agreement_matrix,
        xticklabels=filtered_model_order,
        yticklabels=filtered_model_order,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Agreement Rate"},
        square=True,
        linewidths=0.5,
        annot_kws={"fontsize": 8},
    )

    plt.title(
        f"{title} - {benchmark_name}\nModel Agreement Heatmap",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Model", fontsize=10, fontweight="bold")
    plt.ylabel("Model", fontsize=10, fontweight="bold")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()

    # Save the heatmap
    safe_benchmark_name = benchmark_name.replace("/", "_").replace(" ", "_")
    heatmap_filename = os.path.join(
        plots_dir, f"{filename}_{safe_benchmark_name}_agreement_heatmap.png"
    )
    plt.savefig(
        heatmap_filename,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )
    plt.close()

    if logger:
        logger.info(f"Saved agreement heatmap: {heatmap_filename}")

    # Calculate and log summary statistics
    # Calculate average agreement (excluding diagonal)
    mask = ~np.eye(n_models, dtype=bool)
    avg_agreement = np.mean(agreement_matrix[mask])
    min_agreement = np.min(agreement_matrix[mask])
    max_agreement = np.max(agreement_matrix[mask])

    if logger:
        logger.info(f"Agreement statistics for {benchmark_name}:")
        logger.info(f"  Average agreement: {avg_agreement:.3f}")
        logger.info(f"  Min agreement: {min_agreement:.3f}")
        logger.info(f"  Max agreement: {max_agreement:.3f}")

    return {
        "avg_agreement": avg_agreement,
        "min_agreement": min_agreement,
        "max_agreement": max_agreement,
        "num_models": n_models,
    }
