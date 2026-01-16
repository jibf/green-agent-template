"""Reporting utilities for writing metrics summaries to CSV files."""

import os
import csv
from typing import Dict


def write_metrics_summary_to_csv(
    metrics_summary: Dict,
    output_path: str,
    command_args: Dict = None,
    logger=None
) -> None:
    """Write metrics_summary to CSV file with benchmark comparisons and model performance.

    Args:
        metrics_summary: Dictionary containing all metrics data
        output_path: Path to output CSV file
        command_args: Optional command-line arguments to include in report
        logger: Optional logger for logging info
    """
    if logger:
        logger.info(f"Writing metrics summary to {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []

    # Helper function to handle None values and format numbers
    def format_value(value):
        if value is None:
            return "-"
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return str(value)

    # Header row for metrics comparison
    rows.append(["Metrics Comparison", "Baseline", "Step1", "Step2", "Step4"])

    # Get baseline data and benchmark name (from "original" step)
    baseline_data = metrics_summary.get("original", {})
    steps = ["step1", "step2", "step4"]

    # Get benchmark name from the first available data (assuming single benchmark)
    benchmark_name = list(baseline_data["diversity"].keys())[0]

    # Add benchmark name row
    rows.append([f"Benchmark: {benchmark_name or 'Unknown'}", "", "", "", "", ""])
    rows.append([])  # Empty row for separation

    # Add command-line arguments row
    if command_args:
        rows.append(["Command Arguments", "", "", "", "", ""])
        for arg_name, arg_value in command_args.items():
            if arg_value is not None:
                rows.append([f"  {arg_name}: {arg_value}", "", "", "", "", ""])
        rows.append([])  # Empty row for separation

    # Agreement in format (avg/min/max) - extract using benchmark_name
    agreement_row = ["Agreement (sampled_baseline/after_step)"]
    baseline_agreement_data = baseline_data.get("agreement", {})
    baseline_bench_data = baseline_agreement_data[benchmark_name]
    baseline_avg = baseline_bench_data.get("avg")
    baseline_min = baseline_bench_data.get("min")
    baseline_max = baseline_bench_data.get("max")
    baseline_formatted = f"{format_value(baseline_avg)}/{format_value(baseline_min)}/{format_value(baseline_max)}"
    agreement_row.append(baseline_formatted)

    for step in steps:
        step_data = metrics_summary.get(step, {})
        # Current step agreement
        current_agreement_data = step_data.get("agreement", {})
        current_bench_data = current_agreement_data.get(benchmark_name, {})
        current_avg = current_bench_data.get("avg")
        current_min = current_bench_data.get("min")
        current_max = current_bench_data.get("max")
        current_formatted = f"{format_value(current_avg)}/{format_value(current_min)}/{format_value(current_max)}"

        # Step-specific baseline agreement
        step_baseline_agreement = step_data.get("agreement_baseline", {})
        step_baseline_bench_data = step_baseline_agreement.get(benchmark_name, {})
        step_baseline_avg = step_baseline_bench_data.get("avg")
        step_baseline_min = step_baseline_bench_data.get("min")
        step_baseline_max = step_baseline_bench_data.get("max")
        step_baseline_formatted = f"{format_value(step_baseline_avg)}/{format_value(step_baseline_min)}/{format_value(step_baseline_max)}"

        agreement_row.append(f"{step_baseline_formatted}/{current_formatted}")
    rows.append(agreement_row)

    # CI Overlap (separability with step-specific baselines)
    ci_overlap_row = ["CI Overlap (sampled_baseline/after_step)"]
    # For baseline column, use separability from original step
    baseline_separability = baseline_data.get("separability", [])
    baseline_sep_value = baseline_separability[0][benchmark_name]
    ci_overlap_row.append(format_value(baseline_sep_value))

    for step in steps:
        step_data = metrics_summary.get(step, {})
        # Current step separability
        current_separability = step_data.get("separability", [])
        current_sep_value = current_separability[0].get(benchmark_name) if len(current_separability) > 0 else None

        # Step-specific baseline separability
        step_baseline_separability = step_data.get("separability_baseline", [])
        step_baseline_value = step_baseline_separability[0].get(benchmark_name) if len(step_baseline_separability) > 0 else None

        baseline_val = format_value(step_baseline_value)
        current_val = format_value(current_sep_value)
        ci_overlap_row.append(f"{baseline_val}/{current_val}")
    rows.append(ci_overlap_row)

    # Diversity (extract value using benchmark_name)
    diversity_row = ["Diversity (sampled_baseline/after_step)"]
    baseline_div_value = baseline_data["diversity"][benchmark_name]
    diversity_row.append(format_value(baseline_div_value))

    for step in steps:
        step_data = metrics_summary.get(step, {})
        # Current step diversity
        current_diversity = step_data.get("diversity", {})
        current_div_value = current_diversity.get(benchmark_name) if current_diversity is not None else None

        # Step-specific baseline diversity
        step_baseline_diversity = step_data.get("diversity_baseline", {})
        step_baseline_value = step_baseline_diversity.get(benchmark_name) if step_baseline_diversity is not None else None

        baseline_val = format_value(step_baseline_value)
        current_val = format_value(current_div_value)
        diversity_row.append(f"{baseline_val}/{current_val}")
    rows.append(diversity_row)

    # Precision
    precision_row = ["Precision"]
    precision_row.append("-")  # No baseline precision
    for step in steps:
        step_data = metrics_summary.get(step, {})
        human_alignment = step_data.get("human_alignment", {})
        precision_row.append(format_value(human_alignment.get("precision")))
    rows.append(precision_row)

    # Recall
    recall_row = ["Recall"]
    recall_row.append("-")  # No baseline recall
    for step in steps:
        step_data = metrics_summary.get(step, {})
        human_alignment = step_data.get("human_alignment", {})
        recall_row.append(format_value(human_alignment.get("recall")))
    rows.append(recall_row)

    # Question Num (TP + FP + TN + FN from human alignment)
    question_num_row = ["Question Num"]
    question_num_row.append(metrics_summary["original"]["question_num"])
    for step in steps:
        step_data = metrics_summary.get(step, {})
        human_alignment = step_data.get("human_alignment", {})
        tn = human_alignment.get("tn", 0) or 0
        fn = human_alignment.get("fn", 0) or 0
        question_remain = tn + fn
        question_num_row.append(str(question_remain) if question_remain > 0 else "-")
    rows.append(question_num_row)

    # Total Num (from metrics_summary.total_num)
    total_num_row = ["Total Num"]
    baseline_total_num = baseline_data.get("total_num")
    total_num_row.append(str(baseline_total_num))
    for step in steps:
        step_data = metrics_summary.get(step, {})
        current_total_num = step_data.get("total_num")
        total_num_row.append(str(current_total_num))
    rows.append(total_num_row)

    # Retention Ratio
    retention_row = ["Retention Ratio"]
    retention_row.append("-")  # No baseline retention ratio
    for step in steps:
        step_data = metrics_summary.get(step, {})
        retention_ratio = step_data.get("retention_ratio")
        retention_row.append(format_value(retention_ratio))
    rows.append(retention_row)

    # Add empty rows for separation
    rows.append([])
    rows.append([])

    # Model Performance section
    rows.append(["Model Performance"])
    rows.append([])

    # Get all models and subtasks from the data
    all_models = set()
    all_subtasks = set()

    # Include baseline data
    baseline_model_performance = baseline_data.get("model_performance", {})
    for benchmark_data in baseline_model_performance.values():
        for model_name, model_data in benchmark_data.items():
            all_models.add(model_name)
            subtask_scores = model_data.get("subtask_scores", {})
            all_subtasks.update(subtask_scores.keys())

    # Include step data
    for step in steps:
        step_data = metrics_summary.get(step, {})
        model_performance = step_data.get("model_performance", {})

        # Model performance is nested: {benchmark: {model: {scores...}}}
        for benchmark_data in model_performance.values():
            for model_name, model_data in benchmark_data.items():
                all_models.add(model_name)
                subtask_scores = model_data.get("subtask_scores", {})
                all_subtasks.update(subtask_scores.keys())

    all_models = sorted(list(all_models))
    all_subtasks = sorted(list(all_subtasks))

    # Create baseline model performance table
    baseline_model_performance = baseline_data.get("model_performance", {})
    if baseline_model_performance:
        rows.append(["BASELINE - Model Performance"])

        # Headers: Model, Overall, subtask1, subtask2, ...
        header = ["Model", "Overall"] + all_subtasks
        rows.append(header)

        # Get benchmark data (using benchmark_name)
        benchmark_data = baseline_model_performance.get(benchmark_name, {}) if benchmark_name else {}

        # Model data rows (sorted by overall score descending)
        model_rows = []
        for model_name in all_models:
            if model_name in benchmark_data:
                model_data = benchmark_data[model_name]
                overall_score_raw = model_data.get("overall_score")
                overall_score = format_value(overall_score_raw)
                subtask_scores = model_data.get("subtask_scores", {})

                row = [model_name, overall_score]
                for subtask in all_subtasks:
                    row.append(format_value(subtask_scores.get(subtask)))
                model_rows.append((overall_score_raw or 0, row))

        # Sort by overall score (descending) and add to rows
        model_rows.sort(key=lambda x: x[0], reverse=True)
        for _, row in model_rows:
            rows.append(row)

        rows.append([])  # Empty row after baseline

    # Create model performance table for each step
    for step in steps:
        step_data = metrics_summary.get(step, {})
        model_performance = step_data.get("model_performance", {})

        if not model_performance:
            continue

        # Step header
        rows.append([f"{step.upper()} - Model Performance"])

        # Headers: Model, Overall, subtask1, subtask2, ...
        header = ["Model", "Overall"] + all_subtasks
        rows.append(header)

        # Get benchmark data (using benchmark_name)
        benchmark_data = model_performance.get(benchmark_name, {}) if benchmark_name else {}

        # Model data rows (sorted by overall score descending)
        model_rows = []
        for model_name in all_models:
            if model_name in benchmark_data:
                model_data = benchmark_data[model_name]
                overall_score_raw = model_data.get("overall_score")
                overall_score = format_value(overall_score_raw)
                subtask_scores = model_data.get("subtask_scores", {})

                row = [model_name, overall_score]
                for subtask in all_subtasks:
                    row.append(format_value(subtask_scores.get(subtask)))
                model_rows.append((overall_score_raw or 0, row))

        # Sort by overall score (descending) and add to rows
        model_rows.sort(key=lambda x: x[0], reverse=True)
        for _, row in model_rows:
            rows.append(row)

        # Add subtask statistics table after model performance
        rows.append([])  # Empty row before subtask stats
        rows.append([f"{step.upper()} - Subtask Statistics"])

        # Get subtask details from retention ratio computation
        retention_metrics = step_data.get("retention_metrics")
        if isinstance(retention_metrics, dict) and "subtask_details" in retention_metrics:
            subtask_details = retention_metrics["subtask_details"]

            # Headers: Subtask, Base Num, Retention Num, Ratio
            subtask_header = ["Subtask", "Base Num", "Retention Num", "Ratio"]
            rows.append(subtask_header)

            # Data rows
            for subtask_name, details in subtask_details.items():
                base_num = details.get("base_num", 0)
                retention_num = details.get("retention_num", 0)
                ratio = details.get("ratio", 0.0)

                subtask_row = [
                    subtask_name,
                    str(base_num),
                    str(retention_num),
                    format_value(ratio)
                ]
                rows.append(subtask_row)
        else:
            # If no subtask details available, show placeholder
            rows.append(["No subtask statistics available"])

        rows.append([])  # Empty row between steps

    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)

    if logger:
        logger.info(f"Metrics summary CSV written to {output_path}")
