#!/usr/bin/env python3
"""
Main pipeline for benchmark filtering.
Orchestrates the complete filtering process with rule-based and LLM-as-Judge stages.
"""

import sys
import os
import json
import logging
import argparse
import csv
import numpy as np
import pickle
from copy import deepcopy
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime

from sentence_transformers import SentenceTransformer
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.difficulty_based_filtering import DifficultyBasedFilter
from src.rule_filtering_orchestrator import RuleFilteringOrchestrator
from src.llm_judge_filtering import LLMJudge, LLMJudgeConfig, LLMJudgeStep
from src.data_loader import BenchmarkDataLoader
from src.utils.types import (
    Benchmark,
    UniqueQuestionID,
    LLMJudgeOutput,
    PipelineOutput,
    RuleBasedOutput,
)
from src.utils import group_responses_by_question, log_confusion_matrix_human_labelled, write_metrics_summary_to_csv
from src.bench_loaders import get_bench_loader
from src.metrics import (
    bootstrap_confidence_interval,
    create_task_wise_baseline_sample_set,
    calculate_model_ranking,
    compute_separability,
    compute_retention_ratio,
    visualize_model_performance,
    create_model_agreement_heatmap,
)

# Logger will be configured in main() function
logger = logging.getLogger(__name__)

COMMON_MODEL_SET = {
    'Kimi-K2-Instruct',
    'gpt-4.1-mini',
    'Qwen3-235B-A22B-Thinking-2507-FP8',
    'gpt-4.1-nano',
    'DeepSeek-V3.1-thinking-off',
    'Qwen3-235B-A22B-FP8',
    'o4-mini-high',
    'o3-high',
    'gpt-4.1',
    'gpt-4o-mini',
    'gpt-4o-20240806',
    'claude-4-sonnet-thinking-on-10k',
    'claude-4-sonnet-thinking-off',
    'Qwen3-235B-A22B-Instruct-2507-FP8',
    'DeepSeek-V3-0324',
    'claude-4-opus-thinking-off',
}


class BenchmarkFilteringPipeline:
    """Complete benchmark filtering pipeline."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_loader = BenchmarkDataLoader()
        self.orchestrator = RuleFilteringOrchestrator()
        self.use_specific_filters = self.config.get("use_specific_filters", False)

        # Store metrics for final summary
        results_template = {
            "separability": [],
            "separability_baseline": [],
            "diversity": None,
            "diversity_baseline": None,
            "agreement": {},
            "agreement_baseline": {},
            "human_alignment": {
                "precision": None,
                "recall": None,
                "f1": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "tn": None
            },
            "retention_ratio": None,
            "total_num": None,
            "subtask_size": {},
            "model_performance": {},
        }
        self.metrics_summary = {
            "original": deepcopy(results_template),
            "step1": deepcopy(results_template),
            "step2": deepcopy(results_template),
            "step3": deepcopy(results_template),
        }

        # Determine which LLM judge steps to run based on filtering scheme
        filter_mode = self.config.get("llm_filter_mode", "both")

        if filter_mode == "common":
            llm_steps = [LLMJudgeStep.UNIVERSAL_FILTER]
        elif filter_mode == "specific":
            llm_steps = [LLMJudgeStep.SPECIFIC_FILTER]
        elif filter_mode == "both":
            llm_steps = [LLMJudgeStep.UNIVERSAL_FILTER, LLMJudgeStep.SPECIFIC_FILTER]
        else:
            raise ValueError(
                f"Invalid llm_filter_mode: {filter_mode}. Must be 'common', 'specific', or 'both'"
            )

        self.llm_config = LLMJudgeConfig(
            model=self.config.get("llm_model", "openai/gpt-4.1"),
            max_samples=self.config.get("llm_max_samples", None),
            max_retries=self.config.get("llm_max_retries", 3),
            retry_delay=self.config.get("llm_retry_delay", 1.0),
            num_proc=self.config.get("num_proc", 1),
            steps=llm_steps,
            enable_rebuttal=False,  # Fixed: always disable rebuttal
        )

        # ----- Embedding model for semantic diversity -----
        model_name = self.config.get("embedding_model", "Qwen/Qwen3-Embedding-8B")
        self.embedder = SentenceTransformer(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"torch_dtype": torch.float16},
        )
        self.embedding_batch_size = self.config.get("embedding_batch_size", 8)

        self.fitering_template = {
            "Benchmark": None,
            "task_type": None,
            "task_id": None,
            "specific_rule_passed": None,
            "specific_llm_passed": None,
            "topk_selection_passed": None,
            "comp_passed": None,
        }
        self.filtering_summary = {}

    def _make_json_serializable(self, obj):
        """Make objects JSON serializable by converting enums and other non-serializable types."""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "value"):  # Handle enums like BenchmarkType
            return obj.value
        else:
            return obj

    def _load_precomputed_results(self, csv_path: str) -> Dict[UniqueQuestionID, Dict]:
        """Load precomputed filtering results from CSV and convert to dict keyed by UniqueQuestionID."""
        precomputed_results = {}

        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Create UniqueQuestionID from CSV columns
                unique_question_id = UniqueQuestionID(
                    benchmark=row["Benchmark"],
                    task_name=row["task_type"] if row["task_type"] else None,
                    question_id=row["task_id"],
                )

                # Convert string values to appropriate types, empty strings to None
                results = {}
                for key, value in row.items():
                    if key in ["Benchmark", "task_type", "task_id"]:
                        continue  # Skip keys used for UniqueQuestionID

                    if value == "" or value is None:
                        results[key] = None
                    elif value.lower() in ["true", "false"]:
                        results[key] = value.lower() == "true"
                    else:
                        results[key] = value

                precomputed_results[unique_question_id] = results

        return precomputed_results

    def _convert_response_list_to_qid_list(
        self, response_list: List[Dict]
    ) -> List[UniqueQuestionID]:
        """Convert response list to unique question ID list, removing duplicates."""
        questions = set()
        for response in response_list:
            unique_question_id = UniqueQuestionID(
                benchmark=response["benchmark_name"],  # Already enum after data loading
                task_name=response.get("task_name", None),
                question_id=response["meta"]["id"],
            )
            questions.add(unique_question_id)

        return list(questions)

    def filter_illegal_data(
        self, response_dict: Dict[UniqueQuestionID, List[Dict]]
    ) -> Dict[UniqueQuestionID, List[Dict]]:
        """Filter out illegal data entries from the response dictionary.
        1. duplicate questions from same model
        3. questions do not have all model's responses
        4. models that do not have responses for all questions
        """
        filtered_dict = {}

        # filter out duplicate questions from same model
        for qid, responses in response_dict.items():
            # hardcode for BFCL, only loading multi_turn data
            if qid.benchmark.value == "BFCL" and "multi_turn" not in qid.task_name:
                continue
            model_names = []
            valid_responses = []
            for response in responses:
                model_name = response["model_name"]
                if model_name not in model_names:
                    model_names.append(model_name)
                    valid_responses.append(response)
            filtered_dict[qid] = valid_responses
        response_count = sum([len(responses) for responses in response_dict.values()])
        valid_count = sum([len(responses) for responses in filtered_dict.values()])
        logging.info(
            f"Filtered {response_count - valid_count}/{response_count} duplicated responses."
        )

        # filter out models which do not have enough responses in each benchmark
        non_standard_prefixes = [
            "anthropic-",
            "openai-",
        ]  # TODO: hack for NexusBench, need to fix ASAP
        benchmark_question_statistics = {}
        for qid, responses in filtered_dict.items():
            for response in responses:
                for prefix in non_standard_prefixes:
                    if response["model_name"].startswith(prefix):
                        response["model_name"] = response["model_name"].replace(
                            prefix, ""
                        )

                benchmark_name = response["benchmark_name"]
                model_name = response["model_name"]
                if benchmark_name not in benchmark_question_statistics:
                    benchmark_question_statistics[benchmark_name] = {}
                if model_name not in benchmark_question_statistics[benchmark_name]:
                    benchmark_question_statistics[benchmark_name][model_name] = 0
                benchmark_question_statistics[benchmark_name][model_name] += 1

        benchmark_reamined_model = {}
        for benchmark_name, statics in benchmark_question_statistics.items():
            benchmark_reamined_model[benchmark_name] = set()
            avg_question_count = sum(statics.values()) / len(statics)
            for model_name, v in statics.items():
                if v > 0.8 * avg_question_count:
                    benchmark_reamined_model[benchmark_name].add(model_name)
                else:
                    logging.info(
                        f"Filtered model {model_name} for benchmark {benchmark_name} due to lacking valid responses."
                    )

        # filter out models do not have all results for all benchmarks
        for idx, benchmark_name in enumerate(benchmark_reamined_model):
            if idx == 0:
                remained_model_cross_all_benchmark = benchmark_reamined_model[
                    benchmark_name
                ]
                continue
            remained_model_cross_all_benchmark = (
                remained_model_cross_all_benchmark.intersection(
                    benchmark_reamined_model[benchmark_name]
                )
            )
        remained_model_cross_all_benchmark = (
            remained_model_cross_all_benchmark.intersection(COMMON_MODEL_SET)
        )
        for idx, benchmark_name in enumerate(benchmark_reamined_model):
            filterd_model = (
                benchmark_reamined_model[benchmark_name]
                - remained_model_cross_all_benchmark
            )
            logging.info(
                f"Filtered model {filterd_model} for benchmark {benchmark_name} to maintain consistency cross all benchmarks."
            )

        final_dict = {}
        for qid, responses in filtered_dict.items():
            filterd_response_list = []
            unique_model_set = set()
            for response in responses:
                model_name = response["model_name"]
                if model_name in remained_model_cross_all_benchmark:
                    filterd_response_list.append(response)
                    unique_model_set.add(model_name)

            if unique_model_set == remained_model_cross_all_benchmark:
                final_dict[qid] = filterd_response_list

        response_count = sum([len(responses) for responses in filtered_dict.values()])
        valid_count = sum([len(responses) for responses in final_dict.values()])

        logging.info(
            f"Filtered {response_count - valid_count}/{response_count} responses to maintain consistency cross all benchmarks."
        )

        return final_dict

    def _is_question_flawed(self, llm_output):
        """Check if question is flawed based on any available filter results"""
        if not llm_output:
            return False
        # Check specific filter first, then universal filter
        if llm_output.specific_filter:
            return llm_output.specific_filter.is_flawed
        elif llm_output.universal_filter:
            return llm_output.universal_filter.is_flawed
        return False

    def _get_all_questions_from_loader(self):
        benchmark_names = self.config.get("target_benchmark", None)
        all_questions = {}
        benchmark_names = benchmark_names if benchmark_names is not None else [i.value for i in Benchmark]

        for benchmark_name in benchmark_names:
            benchmark = Benchmark(benchmark_name)
            loader_class = get_bench_loader(benchmark)
            loader = loader_class()
            benchmark_questions = loader.load_questions()
            all_questions[benchmark_name] = benchmark_questions

        return all_questions


    def run_pipeline(self) -> Dict[UniqueQuestionID, PipelineOutput]:
        """Run the complete filtering pipeline."""
        logger.info("Starting benchmark filtering pipeline")

        # Load precomputed results if provided
        precomputed_results = None
        if self.config.get("precomputed_results"):
            precomputed_results = self._load_precomputed_results(
                self.config["precomputed_results"]
            )
            logger.info(f"Loaded {len(precomputed_results)} precomputed results")

        all_responses = self._load_benchmark_data()
        self.all_questions = self._get_all_questions_from_loader()
        self.human_labelled_details = self.data_loader.load_human_labelled_ground_truth(self.config.get("target_benchmark"))
        current_human_labelled = set(self.human_labelled_details.keys())

        responses_by_question = group_responses_by_question(all_responses)
        responses_by_question = self.filter_illegal_data(responses_by_question)

        pipeline_outputs = {k: PipelineOutput() for k in responses_by_question.keys()}
        self.initial_baseline_ids = set(responses_by_question.keys())

        # Compute and log metrics for original/initial dataset
        self.compute_and_log_metrics(
            passed_responses=responses_by_question,
            input_ids=self.initial_baseline_ids,
            phase="initial",
            human_labelled_questions=current_human_labelled,
            baseline_source=responses_by_question,
            no_system_prompt=True,  # Fixed: always use no system prompt
        )

        current_responses = responses_by_question

        # Step 1: Apply benchmark-specific filtering for benchmarks that have specific filters
        logger.info("Step 1: Benchmark-specific filtering")
        if precomputed_results:
            # Use precomputed results instead of running actual filtering
            step1_passed = {}
            for question_id, responses in responses_by_question.items():
                if question_id in precomputed_results and precomputed_results[question_id].get('specific_rule_passed') is True:
                    step1_passed[question_id] = responses
        else:
            step1_passed, step1_dropped = self._run_benchmark_specific_filtering(
                responses_by_question
            )
        # Compute and log all metrics for step1 (including baseline)
        self.compute_and_log_metrics(
            passed_responses=step1_passed,
            input_ids=set(responses_by_question.keys()),
            phase="step1",
            human_labelled_questions=current_human_labelled,
            baseline_source=responses_by_question,
        )
        current_responses = step1_passed
        # Update human labelled questions to only include those that passed step1
        current_human_labelled = current_human_labelled & set(step1_passed.keys())

        # Update pipeline_outputs with step1 results
        step1_passed_questions = set(step1_passed.keys())
        for question_id in pipeline_outputs.keys():
            passed = question_id in step1_passed_questions
            pipeline_outputs[question_id].rule_based_output = RuleBasedOutput(
                passed=passed, reason=None
            )

        # Count unique tasks and samples in step1_passed
        step1_unique_tasks = len(step1_passed)
        step1_sample_count = sum(
            len(responses) for responses in step1_passed.values()
        )
        logger.info(
            f"Step 1 passed: {step1_sample_count} samples from {step1_unique_tasks} unique tasks"
        )

        # Step 2: LLM-as-Judge filtering
        logger.info("Step 2: LLM-as-Judge filtering")

        step1_passed_qids = [qid for qid in current_responses]
        if precomputed_results:
            step2_passed = {}
            for qid in step1_passed_qids:
                if (
                    qid in precomputed_results
                    and precomputed_results[qid].get("specific_llm_passed") is True
                ):
                    step2_passed[qid] = responses_by_question[qid]

            # Create dummy LLM outputs for pipeline_outputs consistency
            for qid in step1_passed_qids:
                if qid in precomputed_results:
                    passed = qid in step2_passed
                    from src.utils.types import LLMJudgeOutput

                    pipeline_outputs[qid].llm_judge_output = LLMJudgeOutput(
                        passed=passed,
                        reason="Precomputed result",
                        model_response={},
                    )
        else:
            step2_result = self._run_llm_judge(current_responses)
            # Update pipeline_outputs with step2 results
            for question_id, llm_output in step2_result.items():
                pipeline_outputs[question_id].llm_judge_output = llm_output

            step2_passed = {
                qid: responses_by_question[qid]
                for qid in step1_passed_qids
                if not self._is_question_flawed(
                    pipeline_outputs[qid].llm_judge_output
                )
            }
        # Compute and log all metrics for step2 (including baseline)
        self.compute_and_log_metrics(
            passed_responses=step2_passed,
            input_ids=set(step1_passed_qids),
            phase="step2",
            human_labelled_questions=current_human_labelled,
            baseline_source=responses_by_question,
        )
        current_responses = step2_passed
        # Update human labelled questions to only include those that passed step2
        current_human_labelled = current_human_labelled & set(step2_passed.keys())
        # Count unique tasks and samples in step2_passed
        step2_unique_tasks = len(step2_passed)
        step2_sample_count = sum(
            len(responses) for responses in step2_passed.values()
        )
        logger.info(
            f"Step 2 passed: {step2_sample_count} samples from {step2_unique_tasks} unique tasks"
        )

        # Step 3: Apply difficulty-based filtering
        logger.info("Step 3: Difficulty-based filtering (final stage)")
        step3_passed = self._run_comprehensive_filtering(
            current_responses, responses_by_question
        )
        # Compute and log all metrics for step3 (including baseline)
        self.compute_and_log_metrics(
            passed_responses=step3_passed,
            input_ids=set(current_responses.keys()),
            phase="step3",
            human_labelled_questions=current_human_labelled,
            baseline_source=responses_by_question,
        )

        # Count unique tasks and samples in step3_passed
        step3_unique_tasks = len(step3_passed)
        step3_sample_count = sum(
            len(responses) for responses in step3_passed.values()
        )
        logger.info(
            f"Step 3 passed: {step3_sample_count} samples from {step3_unique_tasks} unique tasks"
        )
        # Update human labelled questions to only include those that passed step3
        current_human_labelled = current_human_labelled & set(step3_passed.keys())

        self._print_final_summary(pipeline_outputs)
        return pipeline_outputs

    def _load_benchmark_data(self) -> List[Dict]:
        """Load benchmark data based on configuration."""
        logger.info("Loading benchmark data...")

        target_benchmark = self.config.get("target_benchmark")
        if target_benchmark:
            logger.info(f"Loading only target benchmark: {target_benchmark}")

        all_samples = self.data_loader.load_benchmark_data(
            "benchmark", target_benchmark
        )
        logger.info(f"Loaded {len(all_samples):,} total samples")

        return all_samples

    def _run_comprehensive_filtering(
        self, responses_by_question: Dict[str, List[Dict]], all_responses_by_question: Dict[str, List[Dict]]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """Run Step 3: Difficulty-based filtering."""

        logger.info("Applying difficulty-based filtering...")
        rule_filter = DifficultyBasedFilter()
        passed_responses_by_question = (
            rule_filter.filter_samples(responses_by_question, all_responses_by_question)
        )
        logger.info(f"Step 3 completed: {len(passed_responses_by_question):,} samples passed")
        return passed_responses_by_question

    def _run_benchmark_specific_filtering(
        self, responses_by_question: Dict[str, List[Dict]]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """Run Step 1: Benchmark-specific filtering."""
        logger.info(
            "Step 1: Applying benchmark-specific filtering for benchmarks with specific filters..."
        )

        # Group responses by benchmark, but flatten to samples for compatibility with existing filters
        benchmark_groups = {}
        for responses in responses_by_question.values():
            benchmark_name = responses[0].get("benchmark_name", "unknown")
            if benchmark_name not in benchmark_groups:
                benchmark_groups[benchmark_name] = []
            benchmark_groups[benchmark_name].extend(responses)

        all_passed_samples = []
        all_dropped_samples = []

        # Process each benchmark
        for benchmark_name, benchmark_samples in benchmark_groups.items():
            if benchmark_name in self.orchestrator.benchmark_filters:
                logger.info(
                    f"Applying {str(benchmark_name)}-specific filtering to {len(benchmark_samples)} samples"
                )
                passed_samples, dropped_samples = self.orchestrator.filter_samples(
                    benchmark_samples,
                    use_specific_filters=True,
                    target_benchmark=benchmark_name,
                )
                all_passed_samples.extend(passed_samples)
                all_dropped_samples.extend(dropped_samples)
                logger.info(
                    f"{benchmark_name}: {len(passed_samples)} passed, {len(dropped_samples)} dropped"
                )
            else:
                # No specific filter available, keep all samples from this benchmark
                logger.info(
                    f"No specific filter for {benchmark_name}, keeping all {len(benchmark_samples)} samples"
                )
                all_passed_samples.extend(benchmark_samples)

        # Convert back to responses_by_question format
        from src.utils import group_responses_by_question

        passed_responses_by_question = group_responses_by_question(all_passed_samples)
        dropped_responses_by_question = group_responses_by_question(all_dropped_samples)

        passed_count = len(all_passed_samples)
        dropped_count = len(all_dropped_samples)
        logger.info(
            f"Step 1 completed: {passed_count:,} samples passed, {dropped_count:,} samples dropped"
        )
        return passed_responses_by_question, dropped_responses_by_question

    def compute_and_log_metrics(
        self,
        passed_responses: Dict,
        input_ids: set,
        phase: str,
        human_labelled_questions: set,
        baseline_source: Dict = None,
        no_system_prompt: bool = False, # Use system prompt for embeddings  
    ) -> None:
        """Compute and log all metrics for a given filtering phase.

        Args:
            passed_responses: Dict of responses that passed the filter (contains question IDs as keys)
            input_ids: Set of question IDs that were input to the filter
            phase: Phase name (e.g., 'initial', 'step1', 'step2', 'step3')
            human_labelled_questions: Set of all human labelled questions for current filtering stage
            baseline_source: Source data for creating baseline (original responses_by_question)
        """
        benchmark_name = self.config.get("target_benchmark", None)[0]
        logger.info(f"\n=== Computing Metrics for {phase.upper()} ===")
        summary_key = "original" if phase == "initial" else phase

        # Extract passed_ids from passed_responses
        passed_ids = set(passed_responses.keys()) if passed_responses else set()
        baseline_task_count = len(passed_ids)
        self.metrics_summary[summary_key]["total_num"] = baseline_task_count

        # ============================== save filtered csv ==============================
        self._write_filter_summary(
            passed_ids=passed_ids,
            input_ids=input_ids,
            phase=phase,
        )

        # ============================== model ranking ==============================
        model_ranking, model_performance = calculate_model_ranking(passed_responses)
        if phase == "initial":
            self.model_ranking = model_ranking

        # Store model performance data (assuming single benchmark)
        self.metrics_summary[summary_key]["model_performance"] = model_performance

        # ============================== compute embeddings (initial phase only) ==============================
        if phase == "initial":
            # Use system prompt for embeddings
            if not no_system_prompt:
                embed_file = f"./{benchmark_name}_embed_dict.pkl"
            else:
                embed_file = f"./{benchmark_name}_embed_dict_only_instruction.pkl"
            if os.path.exists(embed_file):
                with open(embed_file, "rb") as f:
                    self.embeddings_dict = pickle.load(f)
            else:
                self._compute_embeddings_dict(passed_responses, no_system_prompt)
                with open(embed_file, "wb") as f:
                    pickle.dump(self.embeddings_dict, f)

        # ============================== compute Separability ==============================
        sep_list = []
        for i in range(3):
            separability_dict = compute_separability(passed_responses, n_bootstrap=10000, ci=0.95, bootstrap_ci_func=bootstrap_confidence_interval)
            phase_label = "before filtering" if phase == "initial" else f"after {phase}"
            logger.info(f"Benchmark separability {phase_label}: {json.dumps(separability_dict, indent=2)}")
            sep_list.append(separability_dict)
        self.metrics_summary[summary_key]["separability"] = sep_list


        # ============================== compute diversity metrics ==============================
        diversity_dict = self._compute_diversity(passed_ids)
        self.metrics_summary[summary_key]["diversity"] = diversity_dict
        phase_label = "before filtering" if phase == "initial" else f"for {phase}"
        logger.info(f"Benchmark semantic diversity {phase_label}: {json.dumps(diversity_dict, indent=2)}")

        # ============================== visualize agreement & diversity metrics ==============================
        phase_titles = {
            "initial": "Original Dataset",
            "step1": "After Step 1 (Rule-based Filtering)",
            "step2": "After Step 2 (LLM-as-Judge Filtering)",
            "step3": "After Step 3 (Difficulty-based Filtering)",
        }
        title = phase_titles.get(phase, f"After {phase}")
        filename = f"{phase}_filtered_performance" if phase != "initial" else "original_performance"

        self._visualize_model_performance(passed_responses, title, filename, model_ranking=model_ranking)

        # ============================== compute agreement ==============================
        # Store agreement stats by benchmark (computed in `_visualize_model_performance`)
        agreement_by_benchmark = {}
        for benchmark_name, benchmark_stats in self._current_agreement_stats.items():
            agreement_by_benchmark[benchmark_name] = {
                "min": float(benchmark_stats["min_agreement"]),
                "max": float(benchmark_stats["max_agreement"]),
                "avg": float(benchmark_stats["avg_agreement"])
            }

        self.metrics_summary[summary_key]["agreement"] = agreement_by_benchmark

        # ============================== compute retention ratio ==============================
        retention_metrics = compute_retention_ratio(passed_ids, self.initial_baseline_ids)
        self.metrics_summary[summary_key]["retention_ratio"] = retention_metrics["retention_ratio"]
        self.metrics_summary[summary_key]["subtask_size"] = retention_metrics["subtask_size"]
        # Store complete retention metrics including subtask_details
        self.metrics_summary[summary_key]["retention_metrics"] = retention_metrics
        if phase == "initial":
            self.metrics_summary[summary_key]["question_num"] = len(human_labelled_questions)

        if phase != "initial":
            # Confusion matrix calculation using current human labelled set
            human_alignment_metrics = log_confusion_matrix_human_labelled(
                human_labelled_questions=human_labelled_questions,
                human_labelled_details=self.human_labelled_details,
                passed_ids=passed_ids,
                input_ids=input_ids,
            )
            self.metrics_summary[summary_key]["human_alignment"] = human_alignment_metrics
            # =====================================================================
            # Compute baseline metrics based on same number of samples
            baseline_responses = create_task_wise_baseline_sample_set(baseline_source, baseline_task_count, logger=logger)
            baseline_sample_count = sum(len(responses) for responses in baseline_responses.values())
            logger.info(f"Created task-wise baseline sample set with {baseline_sample_count} samples from {baseline_task_count} tasks")

            # Compute baseline separability (run 3 times for stability)
            sep_list = []
            for i in range(3):
                baseline_separability = compute_separability(baseline_responses, n_bootstrap=10000, ci=0.95, bootstrap_ci_func=bootstrap_confidence_interval)
                logger.info(f"Benchmark separability baseline (vs. {phase}): {json.dumps(baseline_separability, indent=2)}")
                sep_list.append(baseline_separability)
            self.metrics_summary[summary_key]["separability_baseline"] = sep_list

            # Compute baseline diversity
            baseline_diversity = self._compute_diversity(set(baseline_responses.keys()))
            self.metrics_summary[summary_key]["diversity_baseline"] = baseline_diversity
            logger.info(f"Benchmark diversity baseline (vs. {phase}): {json.dumps(baseline_diversity, indent=2)}")

            # Store filtered agreement stats before computing baseline (to avoid overwriting)
            filtered_agreement_stats = self._current_agreement_stats.copy() if hasattr(self, '_current_agreement_stats') else {}

            stored_model_ranking = getattr(self, 'model_ranking', None)
            self._visualize_model_performance(
                baseline_responses,
                "Baseline (Random Sampling)",
                f"{phase}_baseline_performance",
                model_ranking=stored_model_ranking
            )

            # Compute baseline agreement (now that _visualize_model_performance has updated _current_agreement_stats)
            baseline_agreement_by_benchmark = {}
            for benchmark_name, benchmark_stats in self._current_agreement_stats.items():
                baseline_agreement_by_benchmark[benchmark_name] = {
                    "min": float(benchmark_stats["min_agreement"]),
                    "max": float(benchmark_stats["max_agreement"]),
                    "avg": float(benchmark_stats["avg_agreement"])
                }
            self.metrics_summary[summary_key]["agreement_baseline"] = baseline_agreement_by_benchmark
            logger.info(f"Benchmark agreement baseline (vs. {phase}): {json.dumps(baseline_agreement_by_benchmark, indent=2)}")

            # Restore filtered agreement stats for future use
            self._current_agreement_stats = filtered_agreement_stats

        output_path = self.config.get("report_filename")
        write_metrics_summary_to_csv(
            self.metrics_summary,
            output_path,
            self.config.get("command_args"),
            logger
        )

    # Semantic diversity metric
    def _compute_diversity(
        self, question_ids: set
    ) -> Dict[str, float]:
        """Compute semantic diversity for each benchmark using average pairwise cosine distance.

        Uses pre-computed embeddings for efficiency.
        The metric is bounded in [0, 1] per the expression: (2 / (N * (N - 1))) * sum{i<j} [1 - cos(e_i, e_j)]
        where N is the number of samples and cos(·,·) is cosine similarity.

        Args:
            question_ids: Set of question IDs to compute diversity for
        """
        diversity_dict: Dict[str, float] = {}

        for benchmark_name in self.all_questions:
            # Get pre-computed embeddings for these questions
            embeddings = np.array([self.embeddings_dict[qid] for qid in question_ids if qid.benchmark.value == benchmark_name])

            # Cosine similarity matrix via dot product because embeddings are normalised
            sim_matrix = np.matmul(embeddings, embeddings.T)

            # Extract upper triangle (i < j)
            triu_indices = np.triu_indices(len(embeddings), k=1)
            cosine_sims = sim_matrix[triu_indices]
            avg_distance = np.mean(1 - cosine_sims)

            diversity_dict[benchmark_name] = float(avg_distance)

        return diversity_dict


    def _compute_embeddings_dict(self, responses_by_question: Dict[str, List[Dict]], no_system_prompt: bool = False) -> None: # Use system prompt for embeddings
        """Compute embeddings for all questions and store them for reuse.

        This function should be called once at the beginning to compute embeddings
        for all questions in the dataset.

        Args:
            responses_by_question: Dict mapping question IDs to their responses
        """
        logger.info("Computing embeddings for all questions (one-time calculation)...")
        self.embeddings_dict = {}

        # Process each benchmark
        for benchmark_name, question_list in self.all_questions.items():
            texts = []
            question_ids = []

            for question in question_list:
                if question not in responses_by_question:
                    continue
                # Extract text for embedding

                instruction = question.instruction
                
                if not no_system_prompt:
                    system_prompt = getattr(question, "agent_system_prompt",
                                          getattr(question, "system_prompt", "")
                                        )
                    text = system_prompt + "\n" + instruction
                else:
                    text = instruction

                texts.append(text)
                question_ids.append(question)

            if len(texts) == 0:
                continue

            # Compute embeddings for this benchmark
            embeddings = self.embedder.encode(
                texts,
                batch_size=self.embedding_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Store embeddings by question ID (same type as passed_ids)
            for i, question_id in enumerate(question_ids):
                self.embeddings_dict[question_id] = embeddings[i]

    def _visualize_model_performance(
        self,
        responses_by_question: Dict[str, List[Dict]],
        title: str,
        filename: str,
        n_bootstrap: int = 100,
        ci: float = 0.95,
        model_ranking: Dict[str, List[str]] = None,
    ):
        """Wrapper for visualize_model_performance from metrics module."""
        # Use visualize_model_performance with a callback for heatmap creation
        # Create wrapper callback that passes logger to the heatmap function
        def heatmap_callback(responses, benchmark_name, title, filename, model_order):
            return create_model_agreement_heatmap(
                responses, benchmark_name, title, filename, model_order, logger=logger
            )

        agreement_stats = visualize_model_performance(
            responses_by_question,
            title,
            filename,
            n_bootstrap,
            ci,
            model_ranking,
            bootstrap_ci_func=bootstrap_confidence_interval,
            logger=logger,
            agreement_heatmap_callback=heatmap_callback
        )

        # Store agreement stats in self for compatibility with existing code
        if not hasattr(self, "_current_agreement_stats"):
            self._current_agreement_stats = {}
        self._current_agreement_stats.update(agreement_stats)

    def _write_filter_summary(
        self, passed_ids: set, input_ids: set, phase: str
    ):
        """Record filtering summary for each phase.

        Args:
            passed_ids: Set of question IDs that passed current phase
            input_ids: Set of question IDs that input to current phase
            phase: Phase name ("initial", "step0", "step1", "step2", "final")
        """
        logger.info(f"Recording filter summary for phase: {phase}")

        if phase == "initial":
            # Initialize all questions with basic info
            for question_id in passed_ids:
                if question_id not in self.filtering_summary:
                    # Create new entry from template
                    entry = self.fitering_template.copy()

                    # Fill basic info from question_id
                    entry["Benchmark"] = (
                        question_id.benchmark.value
                        if hasattr(question_id.benchmark, "value")
                        else str(question_id.benchmark)
                    )
                    entry["task_type"] = (
                        question_id.task_name if question_id.task_name else ""
                    )
                    entry["task_id"] = question_id.question_id

                    self.filtering_summary[question_id] = entry

        else:
            # Mark questions that were NOT in passed_ids as failed for this phase
            field_map = {
                "step1": "specific_rule_passed",
                "step2": "specific_llm_passed",
                "step3": "comp_passed",
            }

            if phase in field_map:
                field_name = field_map[phase]
                for question_id in input_ids:
                    entry = self.filtering_summary[question_id]

                    # If this question is not in passed_ids and hasn't been marked as failed yet
                    if question_id not in passed_ids:
                        entry[field_name] = False
                    else:
                        entry[field_name] = True

            # Use the new compute_retention_ratio function
            retention_metrics = compute_retention_ratio(passed_ids, self.initial_baseline_ids)

            logging.info(f"==================== Retention ratio after {phase} ====================")
            logging.info(f"Overall retention: {len(passed_ids)}/{len(self.initial_baseline_ids)} = {retention_metrics['retention_ratio']*100:.2f}%")

            for task_type, ratio in retention_metrics['subtask_size'].items():
                logging.info(f"{task_type}: {ratio*100:.2f}%")

        self._save_filter_summary_csv()

    def _save_filter_summary_csv(self):
        """Save filtering summary to CSV file, ordered by filtering stage (latest filtered first)."""
        if not self.filtering_summary:
            logger.info("No filtering summary to save")
            return

        # Sort questions by when they were filtered (reverse order - latest filtered first)
        def get_filter_stage(entry):
            # Return stage number where question was filtered (higher = later)
            order = 0
            for stage in ["topk_selection_passed", "specific_llm_passed", "specific_rule_passed", "comp_passed"]:
                if entry[stage] == True:
                    order += 1
            if entry.get("is_issue") == True:
                order += 0.5
            return order

        # Sort by filter stage (descending) then by question_id for consistency
        sorted_items = sorted(
            self.filtering_summary.items(),
            key=lambda x: (get_filter_stage(x[1]), str(x[0])),
            reverse=True,
        )

        csv_path = self.config.get("csv_filename")

        # Write CSV
        fieldnames = list(self.fitering_template.keys())

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for question_id, entry in sorted_items:
                row = entry.copy()

                # Replace None values with empty strings for CSV
                for key, value in row.items():
                    if value is None:
                        row[key] = ""

                writer.writerow(row)

        logger.info(f"Filtering summary saved to: {csv_path}")

    def _print_final_summary(
        self, pipeline_outputs: Dict[UniqueQuestionID, PipelineOutput]
    ):
        logger.info("Pipeline completed - Final summary")

        step1_passed = [
            qid
            for qid, output in pipeline_outputs.items()
            if not output.rule_based_output or output.rule_based_output.passed
        ]

        step2_passed = [
            qid
            for qid in step1_passed
            if not self._is_question_flawed(pipeline_outputs[qid].llm_judge_output)
        ]

        step1_benchmarks = Counter(qid.benchmark.value for qid in step1_passed)
        step2_benchmarks = Counter(qid.benchmark.value for qid in step2_passed)

        logger.info(f"\nStep 1 (Rule-based) results:")
        for benchmark, count in sorted(step1_benchmarks.items()):
            logger.info(f"  {benchmark}: {count:,}")

        logger.info(f"\nStep 2 (LLM-as-Judge) results:")
        for benchmark, count in sorted(step2_benchmarks.items()):
            logger.info(f"  {benchmark}: {count:,}")

        logger.info(f"\nUnique questions:")
        logger.info(f"  After Step 1: {len(step1_passed):,}")
        logger.info(f"  After Step 2: {len(step2_passed):,}")

        # Print compiled metrics summary
        self._print_compiled_metrics_summary()

        logger.info(f"\nPipeline complete!")

    def _print_compiled_metrics_summary(self):
        """Print a compiled summary of all metrics collected during the pipeline."""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"COMPILED METRICS SUMMARY")
        logger.info(f"{'=' * 80}")

        step_names = {
            "original": "Original Dataset",
            "step1": "After Step 1 (Rule-based)",
            "step2": "After Step 2 (LLM-as-Judge)",
            "step3": "After Step 3 (Difficulty-based)",
            "step1_baseline": "Step 1 Baseline",
            "step2_baseline": "Step 2 Baseline",
            "step3_baseline": "Step 3 Baseline",
        }

        for step_key, step_name in step_names.items():
            if step_key not in self.metrics_summary:
                continue

            step_metrics = self.metrics_summary[step_key]
            if not any(step_metrics.values()):  # Skip if no metrics available
                continue

            logger.info(f"\n{step_name}:")
            logger.info(f"{'-' * 50}")

            available_keys = set(step_metrics.keys())

            # Separability
            if "separability" in available_keys and step_metrics["separability"]:
                logger.info(f"Separability by benchmark:")
                # Handle both dict and list of dicts (for repeated runs)
                if isinstance(step_metrics["separability"], dict):
                    sep_dicts = [step_metrics["separability"]]
                else:
                    sep_dicts = step_metrics["separability"]
                # Print each run
                for run_idx, sep_dict in enumerate(sep_dicts):
                    if len(sep_dicts) > 1:
                        logger.info(f"  Run {run_idx + 1}:")
                    for benchmark, separability in sep_dict.items():
                        logger.info(f"    {benchmark}: {separability:.3f}")
                    # avg_separability = np.mean(list(sep_dict.values()))
                    # logger.info(f"    Average: {avg_separability:.3f}")
            else:
                logger.info(f"Separability: Not computed")

            if "diversity" in available_keys and step_metrics["diversity"]:
                logger.info(f"Semantic diversity by benchmark:")
                for benchmark, diversity in step_metrics["diversity"].items():
                    logger.info(f"  {benchmark}: {diversity:.3f}")
                # avg_diversity = np.mean(list(step_metrics["diversity"].values()))
                # logger.info(f"  Average: {avg_diversity:.3f}")
            elif "diversity" in available_keys:
                logger.info(f"Semantic diversity: Not computed")

            if "agreement" in available_keys and step_metrics["agreement"]:
                logger.info(f"Model agreement statistics:")
                for benchmark, stats in step_metrics["agreement"].items():
                    logger.info(f"  {benchmark}:")
                    logger.info(f"    Average agreement: {stats['avg']:.3f}")
                    logger.info(f"    Min agreement: {stats['min']:.3f}")
                    logger.info(f"    Max agreement: {stats['max']:.3f}")
            elif "agreement" in available_keys:
                logger.info(f"Model agreement: Not computed")

    def _run_llm_judge(
        self, responses_by_question: Dict[UniqueQuestionID, List[Dict]]
    ) -> Dict[UniqueQuestionID, LLMJudgeOutput]:
        """Run LLM judge independently on questions from benchmark datasets."""
        # Determine which benchmarks to process based on target_benchmark config
        judge = LLMJudge(self.llm_config)
        return judge.judge_questions(responses_by_question)


def main():
    """Main entry point."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Benchmark Filtering Pipeline")
    parser.add_argument(
        "--llm-model",
        default="google/gemini-2.5-pro-thinking-on",
        help="LLM model to use for Step 2 (default: google/gemini-2.5-pro-thinking-on)",
    )
    parser.add_argument(
        "--llm-max-samples",
        type=int,
        help="Maximum samples to process in Step 2 (default: all)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes for multiprocessing (default: 1)",
    )
    parser.add_argument(
        "--target-benchmark",
        nargs="+",
        choices=[benchmark.value for benchmark in Benchmark],
        help="Target benchmark(s) to process (default: all available benchmarks)",
    )
    parser.add_argument(
        "--llm-filter-mode",
        choices=["common", "specific", "both"],
        default="specific",
        help="LLM filtering scheme: 'common' (universal filter only), 'specific' (benchmark-specific filter only), 'both' (universal + specific filters)",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
        help="SentenceTransformer model for semantic diversity computation (default: Qwen/Qwen3-Embedding-8B)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=4,
        help="Batch size when encoding texts for diversity (default: 8)",
    )
    parser.add_argument(
        "--precomputed-results",
        type=str,
        help="Path to CSV file with precomputed filtering results to skip actual filtering steps",
    )
    args = parser.parse_args()

    output_dir = "pipeline_results"
    os.makedirs(output_dir, exist_ok=True)

    # Generate file prefix based on target_benchmark
    if args.target_benchmark:
        benchmark_prefix = "_".join(args.target_benchmark)
    else:
        benchmark_prefix = "all"

    # Generate filenames with unified naming convention
    log_filename = f"{output_dir}/{benchmark_prefix}_{timestamp}_pipeline.log"
    csv_filename = f"{output_dir}/{benchmark_prefix}_{timestamp}_filtering_summary.csv"
    report_filename = f"{output_dir}/{benchmark_prefix}_{timestamp}_report.csv"

    # Store command-line arguments for logging
    command_args = {
        "llm_model": args.llm_model,
        "llm_max_samples": args.llm_max_samples,
        "num_proc": args.num_proc,
        "target_benchmark": args.target_benchmark,
        "llm_filter_mode": args.llm_filter_mode,
        "embedding_model": args.embedding_model,
        "embedding_batch_size": args.embedding_batch_size,
        "precomputed_results": args.precomputed_results,
    }

    # Configuration
    config = {
        "llm_model": args.llm_model,
        "llm_max_samples": args.llm_max_samples,
        "num_proc": args.num_proc,
        "target_benchmark": args.target_benchmark,
        "llm_filter_mode": args.llm_filter_mode,
        "embedding_model": args.embedding_model,
        "embedding_batch_size": args.embedding_batch_size,
        "csv_filename": csv_filename,
        "report_filename": report_filename,
        "precomputed_results": args.precomputed_results,
        "command_args": command_args,
    }

    # Set up logging with dynamic filename
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
        force=True  # This allows reconfiguring if already configured
    )

    # Run pipeline
    pipeline = BenchmarkFilteringPipeline(config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
