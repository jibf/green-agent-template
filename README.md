# AgentHard Pipeline

A comprehensive benchmark filtering pipeline for agent evaluation datasets. This pipeline applies multi-stage filtering (rule-based, LLM-as-Judge, and difficulty-based filtering) to ensure high-quality benchmark data.

## Overview

The pipeline processes agent benchmark datasets through multiple filtering stages:

- **Step 1**: Benchmark-specific rule-based filtering
- **Step 2**: LLM-as-Judge quality assessment
- **Step 3**: Difficulty-based filtering

Throughout the process, the pipeline computes and tracks various quality metrics including:
- Model performance separability
- Semantic diversity
- Model agreement statistics
- Precision/Recall against human-labeled ground truth
- Retention ratios across filtering stages

## Features

- 🔍 Multi-stage filtering with rule-based and LLM-based approaches
- 📊 Comprehensive metrics computation and visualization
- 🎯 Support for multiple benchmark formats
- 🔄 Parallel processing support for LLM-based filtering
- 💾 Precomputed results caching
- 📈 Automatic report generation (CSV format)
- 🔧 Configurable filtering modes and parameters

## Project Structure

```
agenthard_pipeline_anonymous/
├── main.py                          # Main pipeline entry point
├── src/
│   ├── benchmark_specific_filters/  # Benchmark-specific filtering rules
│   ├── bench_loaders/               # Data loaders for different benchmarks
│   ├── prompts/                     # LLM-as-Judge prompts
│   ├── metrics/                     # Metrics computation modules
│   ├── utils/                       # Utility functions and types
│   ├── difficulty_based_filtering.py
│   ├── data_loader.py
│   ├── llm_judge_filtering.py
│   └── rule_filtering_orchestrator.py
├── benchmark/                       # Benchmark evaluation results (user-provided)
├── problematic_questions/           # Known problematic samples
├── human_labelled_ground_truth/     # Human-labeled validation data
├── cached_results/                  # Precomputed LLM-as-Judge results (optional)
├── pipeline_results/                # Output directory
├── requirements.txt                 # Python dependencies
├── setup_env.sh                    # Environment setup script
└── .env.example                    # API configuration template
```

## Prerequisites

- Python 3.11+
- Conda or Miniconda
- CUDA-compatible GPU (recommended for embedding models)
- **Benchmark evaluation results** (see [Benchmark Data Setup](#benchmark-data-setup) below)
- OpenAI-compatible API endpoint for LLM-as-Judge filtering (optional if using precomputed results)

## Installation

### Quick Setup (Recommended)

Run the automated setup script:

```bash
bash setup_env.sh
```

This will:
1. Create a conda environment named `agenthard_pipeline` with Python 3.11
2. Install all required dependencies
3. Verify the installation

### Manual Setup

```bash
# Create and activate conda environment
conda create -n agenthard_pipeline python=3.11 -y
conda activate agenthard_pipeline

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### API Configuration

The pipeline requires API credentials for LLM-as-Judge filtering:

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and configure your API credentials:
   ```bash
   API_KEY=your_api_key_here
   BASE_URL=https://api.openai.com/v1
   ```

   **Note**: Currently only OpenAI-compatible API endpoints are supported (e.g., OpenAI, Azure OpenAI, vLLM, LiteLLM).

   **Optional**: If you have precomputed LLM-as-Judge results in the `cached_results/` directory, you can skip API configuration and use `--precomputed-results` flag (see [Usage Examples](#examples) below).

### Benchmark Data Setup

**IMPORTANT**: This repository contains only the filtering pipeline. You must separately run benchmark evaluations and provide the results.

#### Directory Structure

Place your benchmark evaluation results in the `benchmark/` directory with the following structure:

```
benchmark/
├── <benchmark_name>-evaluation/
│   ├── <model_1>.jsonl
│   ├── <model_2>.jsonl
│   ├── <model_3>.jsonl
│   └── ...
```

Example:
```
benchmark/
├── tau-bench-evaluation/
│   ├── gpt-4o.jsonl
│   ├── claude-4-sonnet.jsonl
│   └── gemini-2.0-pro.jsonl
├── DrafterBench-evaluation/
│   ├── gpt-4o.jsonl
│   └── ...
```

#### JSONL Format

Each line in the JSONL file should be a JSON object with the following required fields:

```json
{
  "model_path": "anthropic/claude-4.5-sonnet",
  "benchmark_name": "tau-bench",
  "task_name": "airline",
  "messages": [
    {"role": "system", "content": "...", "turn_idx": 0},
    {"role": "user", "content": "...", "turn_idx": 1},
    {"role": "assistant", "content": "...", "turn_idx": 2, "tool_calls": [...]},
    {"role": "tool", "content": "...", "turn_idx": 3, "tool_call_id": "...", "name": "..."}
  ],
  "meta": {
    "id": "airline_1",
    "is_correct": true
  },
  "eval_result": {
    "score": 1.0,
    "passed": true
  }
}
```

**Required fields:**
- `model_path` or `model_name`: Model identifier
- `benchmark_name`: Name of the benchmark
- `task_name`: Task/subtask identifier (optional for some benchmarks)
- `messages`: Full conversation trace including tool calls
- `meta`: Metadata object with at least an `id` field
- `eval_result`: Evaluation results with success indicators (`score`, `passed`, `is_correct`, etc.)

**Note**: This pipeline does NOT run benchmark evaluations. You need to:
1. Run your models on the target benchmarks separately
2. Format the results according to the above specification
3. Place the JSONL files in the appropriate `benchmark/<benchmark_name>-evaluation/` directory

## Quick Start Workflow

### Typical Usage Flow

1. **Prepare benchmark data**: Place evaluation results in `benchmark/<benchmark>-evaluation/`
2. **Option A - With API access**: Configure `.env` with API credentials
3. **Option B - Without API access**: Use precomputed results from `cached_results/`
4. **Run the pipeline**: Execute `python main.py` with appropriate flags
5. **Review results**: Check output files in `pipeline_results/`

### Running Without API Access

If you don't have API credentials but want to test the pipeline, use precomputed results:

```bash
# Example: Run pipeline with cached LLM-as-Judge results
python main.py \
  --target-benchmark tau-bench \
  --precomputed-results cached_results/tau-bench_20250924_011852_filtering_summary.csv
```

This will:
- ✅ Skip Step 2 (LLM-as-Judge filtering)
- ✅ Use cached filtering decisions
- ✅ Still compute all metrics and generate reports
- ❌ Cannot modify LLM filtering decisions

## Usage

### Basic Usage

```bash
conda activate agenthard_pipeline
python main.py --target-benchmark <benchmark_name>
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--target-benchmark` | str[] | all | Target benchmark(s) to process |
| `--llm-model` | str | google/gemini-2.5-pro-thinking-on | LLM model for Step 2 filtering |
| `--llm-filter-mode` | str | specific | Filtering mode: `common`, `specific`, or `both` |
| `--num-proc` | int | 1 | Number of parallel processes for LLM filtering |
| `--llm-max-samples` | int | None | Maximum samples to process (for testing) |
| `--embedding-model` | str | Qwen/Qwen3-Embedding-8B | Model for semantic diversity computation |
| `--embedding-batch-size` | int | 4 | Batch size for embedding computation |
| `--precomputed-results` | str | None | Path to CSV with precomputed filtering results |

### Examples

**Process a single benchmark:**
```bash
python main.py --target-benchmark tau-bench
```

**Process multiple benchmarks with parallel LLM filtering:**
```bash
python main.py --target-benchmark tau-bench DrafterBench --num-proc 8
```

**Use a specific LLM model:**
```bash
python main.py \
  --target-benchmark tau-bench \
  --llm-model openai/gpt-4o \
  --llm-filter-mode both
```

**Test with limited samples:**
```bash
python main.py \
  --target-benchmark tau-bench \
  --llm-max-samples 100
```

**Use precomputed LLM-as-Judge results (skip API calls):**
```bash
python main.py \
  --target-benchmark tau-bench \
  --precomputed-results cached_results/tau-bench_20250924_011852_filtering_summary.csv
```

**Note**: The `cached_results/` directory contains precomputed LLM-as-Judge filtering results. Using this option allows you to run the pipeline without API credentials, as it will skip Step 2 (LLM-as-Judge) and use the cached results instead. This is useful for:
- Testing the pipeline without API access
- Reproducing previous filtering runs
- Avoiding redundant API calls

## Output

The pipeline generates several output files in the `pipeline_results/` directory:

1. **Filtering Summary CSV**: `<benchmark>_<timestamp>_filtering_summary.csv`
   - Detailed filtering results for each question
   - Shows which filters passed/failed for each sample

2. **Metrics Report CSV**: `<benchmark>_<timestamp>_report.csv`
   - Comprehensive metrics comparison across filtering stages
   - Model performance statistics
   - Retention ratios and human alignment metrics

3. **Log File**: `<benchmark>_<timestamp>_pipeline.log`
   - Detailed execution logs
   - Metrics computation results
   - Error messages and warnings

4. **Visualizations**: `<phase>_filtered_performance.png`
   - Model performance charts with confidence intervals
   - Model agreement heatmaps

## Metrics

The pipeline computes the following metrics at each filtering stage:

- **Separability**: Measures how well the benchmark distinguishes between model performance levels
- **Diversity**: Semantic diversity of questions using embedding-based similarity
- **Agreement**: Model agreement statistics (min/max/average across questions)
- **Precision/Recall**: Alignment with human-labeled ground truth data
- **Retention Ratio**: Proportion of samples retained after filtering
- **Model Performance**: Per-model and per-subtask performance scores

## Supported Benchmarks

The pipeline currently supports:
- tau-bench
- tau2-bench
- DrafterBench
- ACEBench
- BFCL (v3)
- ComplexFuncBench

## Troubleshooting

### Module Not Found Errors

Ensure the conda environment is activated:
```bash
conda activate agenthard_pipeline
```

### API Rate Limiting

If you encounter rate limits during LLM-as-Judge filtering:
- Reduce `--num-proc` to decrease parallel requests
- Use `--llm-max-samples` to process in batches
- Consider using cached results with `--precomputed-results`

### GPU Memory Issues

For embedding computation on GPU:
- Reduce `--embedding-batch-size` (default: 4)
- Ensure CUDA is properly configured
- The pipeline will automatically fall back to CPU if GPU is unavailable

### Missing Benchmark Data

Ensure the `benchmark/` directory contains the evaluation data:
```bash
ls benchmark/<benchmark_name>-evaluation/
```

## Development

### Adding a New Benchmark

1. Create a loader in `src/bench_loaders/<benchmark>_loader.py`
2. Create a filter in `src/benchmark_specific_filters/<benchmark>_filter.py`
3. Create prompts in `src/prompts/<benchmark>_prompt.py`
4. Add the benchmark to `src/utils/types.py` in the `Benchmark` enum

### Running Tests

```bash
# Test with a small sample
python main.py --target-benchmark tau-bench --llm-max-samples 10
```

## Contributing

When contributing:
1. Ensure all code passes syntax checks
2. Follow existing code style and structure
3. Update documentation for new features
4. Test with multiple benchmarks

