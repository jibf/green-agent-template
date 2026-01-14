# ComplexFuncBench Green Agent

A green agent for evaluating [ComplexFuncBench](https://github.com/MaggiR/MAFC-Benchmark) - a comprehensive function calling benchmark. Built with [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) protocol for [AgentBeats](https://agentbeats.dev) platform.

## Quick Links

- 📖 **[Integration Guide](CFBENCH_INTEGRATION.md)** - How to configure and use this evaluator
- 🧪 **[Testing Guide](TESTING_GUIDE.md)** - 3-level testing strategy
- 🛠️ **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Technical details

## What is ComplexFuncBench?

ComplexFuncBench evaluates LLMs on 5 complex function calling scenarios:
- 🔗 Multi-step function calling in a single turn
- 📋 User-provided constraints
- 🧠 Parameter value reasoning from implicit information
- 📝 Long parameter values (>500 tokens)
- 📚 Long context (128k tokens)

Uses real Booking.com API for 1,000 travel planning tasks.

## Project Structure

```
src/
├─ server.py              # A2A server and agent card
├─ executor.py            # A2A request handling
├─ agent.py               # Green agent (evaluation orchestrator)
├─ messenger.py           # A2A messaging utilities
├─ cfbench_wrapper.py     # ComplexFuncBench integration (tau2 pattern)
└─ response_evaluator.py  # GPT-4o response quality evaluation
tests/
└─ test_agent.py          # A2A protocol tests
ComplexFuncBench/         # Original benchmark code (submodule)
mock_purple_agent.py      # Mock agent for testing
Dockerfile                # Docker configuration
```

## Getting Started

### 1. Setup Environment

```bash
# Install dependencies
uv sync

# Create .env file
cp .env.example .env

# Edit .env and configure:
# Required:
#   RAPID_API_KEY      - Get from https://rapidapi.com/
#   OPENAI_API_KEY     - API key for evaluation model (GPT-4o)
# Optional:
#   OPENAI_BASE_URL    - Custom API endpoint (leave blank for OpenAI default)
#   EVAL_MODEL         - Model name (default: gpt-4o-2024-08-06)
```

The evaluation model is used for:
- LLM-based function call comparison (when rule-based and response-based matching fail)
- Response quality evaluation (completeness and correctness scoring)
```

### 2. Quick Test (3 Levels)

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed instructions.

**Level 1 - A2A Protocol Tests:**
```bash
uv run src/server.py --port 9009
pytest tests/test_agent.py
```

**Level 2 - Integration Flow Test:**
```bash
# Terminal 1
uv run src/server.py --port 9009

# Terminal 2
python mock_purple_agent.py --port 9019

# Terminal 3
curl -X POST http://localhost:9009/tasks \
  -H "Content-Type: application/json" \
  -d '{"message": {"role": "user", "parts": [{"text": "{\"participants\": {\"agent\": \"http://localhost:9019\"}, \"config\": {\"sample_ids\": [\"Car-Rental-0\"]}}"}]}}'
```

**Level 3 - Real Evaluation:**
Replace `mock_purple_agent.py` with your real LLM-based agent.

## Running with scenario.toml

For local testing or AgentBeats platform deployment, use `scenario.toml`:

```bash
# Edit scenario.toml to configure:
# - num_tasks: number of tasks to run
# - sample_ids: specific samples to test
# - enable_response_eval: enable/disable response evaluation
# - Your purple agent command

# Run locally with agentbeats-run
uv run agentbeats-run scenario.toml

# Or run with custom purple agent
# 1. Start your purple agent on port 8000
# 2. Edit scenario.toml participants.cmd
# 3. Run: uv run agentbeats-run scenario.toml
```

The `[config]` section in `scenario.toml` defines evaluation parameters:
- `num_tasks` - Number of tasks to run (default: all)
- `sample_ids` - Specific samples to test (overrides num_tasks)
- `enable_response_eval` - Enable response quality evaluation (default: true)
- `debug` - Debug mode for quick testing (default: false)

## Running Locally (Manual)

```bash
# Install dependencies
uv sync

# Run the server
uv run src/server.py
```

## Running with Docker

```bash
# Build the image
docker build -t my-agent .

# Run the container
docker run -p 9009:9009 my-agent
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).
