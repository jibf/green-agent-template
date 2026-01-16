# Multi-Benchmark Green Agent

A unified green agent for evaluating multiple function calling benchmarks on the [AgentBeats](https://agentbeats.dev) platform. Built with [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) protocol.

## 🌟 Key Features

### Production-Ready Quality Control Pipeline

**All three benchmarks include curated `invalid_tasks.txt` files** that filter out problematic test cases identified during porting:
- ❌ **Ground truth errors** - Incorrect expected outputs in original benchmarks
- ❌ **Ambiguous prompts** - Unclear or underspecified task instructions
- ❌ **Evaluation issues** - Tests with broken or unreliable evaluation logic
- ✅ **Result**: More robust and reliable evaluation than original benchmarks

**Impact**:
- **BFCL**: 139 invalid tasks filtered from 4,706 total → **4,567 high-quality tests**
- **ComplexFuncBench**: 161 invalid tasks filtered from 1,000 total → **839 high-quality tests**
- **Tau2**: 96 invalid tasks filtered → **More reliable customer service evaluations**

This quality control pipeline ensures your agent evaluations are meaningful and reproducible, eliminating frustration from benchmark bugs.

---

## Supported Benchmarks

This green agent supports **three major function calling benchmarks**:

### 🔵 BFCL (Berkeley Function Calling Leaderboard)
- **Total test cases**: 4,706 across all categories
- **Recommended subset**: v3_v4 (~400 tests: multi-turn + agentic)
- **Filtered**: 139 invalid tasks removed for quality
- **Categories**: Single-turn, multi-turn, live, agentic (memory & web search)
- **Focus**: Comprehensive function calling evaluation across diverse scenarios

### 🟢 ComplexFuncBench
- **Total test cases**: 1,000 real-world travel planning scenarios
- **Filtered**: 161 invalid tasks removed for quality → **839 valid tests**
- **Focus**: Complex function calling with real Booking.com API
- **Features**: Multi-step calls, constraints, reasoning, long parameters/context (128k)

### 🟣 Tau2
- **Test cases**: Domain-specific customer service scenarios
- **Filtered**: 96 invalid tasks removed for quality
- **Domains**: Airline, retail, telecom
- **Focus**: Realistic multi-turn conversations with tool usage

## Quick Start

### Local Development

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run the agent
uv run python docker-entrypoint.py --host 127.0.0.1 --port 8001
```

### Docker Usage

```bash
# Build the image
docker build -t green-agent:latest .

# Run the container
docker run -p 8001:8001 \
  -e OPENAI_API_KEY=your-key \
  -e OPENAI_BASE_URL=your-base-url \
  green-agent:latest
```

## Running Evaluations

The agent automatically selects the appropriate benchmark based on the evaluation request's `config.benchmark` field.

### Using scenario.toml Files

We provide 6 pre-configured scenario files:

**BFCL:**
- `scenario.bfcl.toml` - Full v3_v4 evaluation (~400 tasks)
- `scenario.bfcl-debug.toml` - Quick test (10 tasks)

**ComplexFuncBench:**
- `scenario.cfb.toml` - Full evaluation (~200 tasks)
- `scenario.cfb-debug.toml` - Quick test (10 tasks)

**Tau2:**
- `scenario.tau2.toml` - Full airline domain evaluation
- `scenario.tau2-debug.toml` - Quick test (5 tasks)

**Run locally:**
```bash
# Test BFCL (debug mode)
uv run agentbeats-run scenario.bfcl-debug.toml

# Test ComplexFuncBench (full)
uv run agentbeats-run scenario.cfb.toml

# Test Tau2 (debug)
uv run agentbeats-run scenario.tau2-debug.toml
```

### Manual API Request

You can also send evaluation requests directly to the A2A endpoint:

```bash
curl -X POST http://localhost:8001/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "role": "user",
      "parts": [{
        "text": "{\"participants\": {\"agent\": \"http://purple-agent:8000\"}, \"config\": {\"benchmark\": \"bfcl\", \"test_category\": \"v3_v4\", \"num_tasks\": 10}}"
      }]
    }
  }'
```

## Configuration

Each benchmark supports different configuration options in the `[config]` section of scenario.toml:

### BFCL Config Options
```toml
[config]
benchmark = "bfcl"
test_category = "v3_v4"  # Options: v3_v4, all, multi_turn, agentic, memory, web_search, etc.
num_tasks = 10           # Optional: limit number of tasks
# sample_ids = ["simple_python_0", "simple_python_1"]  # Optional: specific tasks
```

**Available categories**:
- `v3_v4` - Multi-turn + agentic (recommended, default)
- `all` - All test categories
- `multi_turn` - Multi-turn conversations
- `agentic` - Memory + web search tests
- `memory` - Memory tests (kv, vector, rec_sum)
- `web_search` - Web search tests
- `simple_python` - Single-turn Python tests
- And more...

### ComplexFuncBench Config Options
```toml
[config]
benchmark = "cfb"
num_tasks = 10           # Optional: limit number of tasks
# sample_ids = ["Car-Rental-0", "Hotel-0"]  # Optional: specific tasks
```

### Tau2 Config Options
```toml
[config]
benchmark = "tau2"
domain = "airline"       # Options: airline, retail, telecom, mock
num_tasks = 5            # Optional: limit number of tasks
user_llm = "openai/gpt-4o"  # LLM for user simulator
```

## Environment Variables

Required/optional environment variables:

```bash
# OpenAI (or compatible endpoint)
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://your-custom-endpoint/v1  # Optional

# For BFCL web search tests
SERPAPI_API_KEY=your-serpapi-key  # Optional

# For ComplexFuncBench
RAPID_API_KEY=your-rapidapi-key  # Optional, for real Booking.com API
```

## Project Structure

```
green-agent-template/
├── BFCL/                      # BFCL benchmark module
│   ├── bfcl_agent.py          # BFCL green agent
│   ├── server.py              # BFCL standalone server
│   └── bfcl_eval/             # BFCL evaluation code & data
├── ComplexFuncBench/          # ComplexFuncBench module
│   ├── cfbench_agent.py       # CFB green agent
│   ├── server.py              # CFB standalone server
│   └── ComplexFuncBench/      # CFB evaluation code & data
├── Tau2/                      # Tau2 benchmark module
│   ├── tau2_evaluator.py      # Tau2 green agent
│   └── tau2-bench/            # Tau2 evaluation code & data
├── docker-entrypoint.py       # Unified Docker entrypoint
├── router_executor.py         # Routes requests to benchmarks
├── Dockerfile                 # Multi-benchmark Docker image
├── scenario.*.toml            # 6 scenario configurations
└── .github/workflows/         # CI/CD automation
    └── publish.yml            # Auto-publish Docker images
```

## Publishing to GitHub Container Registry

The repository includes a GitHub Actions workflow that automatically builds and publishes Docker images.

### Automatic Publishing

**On push to `main` branch:**
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

**On version tag (e.g., `v1.0.0`):**
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

### Manual Publishing

```bash
# Build for linux/amd64 (required by AgentBeats)
docker build --platform linux/amd64 -t ghcr.io/your-username/your-repo:v1.0 .

# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u your-username --password-stdin

# Push the image
docker push ghcr.io/your-username/your-repo:v1.0
```

### Configure GitHub Secrets

Add your API keys in: **Settings → Secrets and variables → Actions**

Required secrets:
- `OPENAI_API_KEY` - For LLM evaluation
- `OPENAI_BASE_URL` - (Optional) Custom endpoint

Optional secrets:
- `SERPAPI_API_KEY` - For BFCL web search tests
- `RAPID_API_KEY` - For ComplexFuncBench Booking.com API

## AgentBeats Platform Integration

### Workflow Trigger

The repository includes `.github/workflows/publish.yml` that:
1. Builds the Docker image on push/tag
2. Publishes to GitHub Container Registry
3. Makes it available on AgentBeats platform

### Running on AgentBeats

Users can select which scenario to run:
1. Go to your repository on GitHub
2. Click "Actions" tab
3. Select "Run Assessment" workflow
4. Choose a scenario file from dropdown:
   - `scenario.bfcl.toml`
   - `scenario.bfcl-debug.toml`
   - `scenario.cfb.toml`
   - `scenario.cfb-debug.toml`
   - `scenario.tau2.toml`
   - `scenario.tau2-debug.toml`
5. Click "Run workflow"

The platform will:
- Pull your Docker image from GHCR
- Start the container with the selected configuration
- Run the evaluation against the purple agent
- Display results

## Testing

### Unit Tests
```bash
# Run A2A protocol tests
uv run pytest tests/
```

### Integration Tests
```bash
# Test BFCL
uv run python test_bfcl_e2e.py --test-category v3_v4 --num-tasks 10

# Test ComplexFuncBench
# (start green and purple agents first)
uv run python test_cfb_e2e.py

# Test Tau2
uv run python test_tau2_e2e.py --domain airline --num-tasks 5
```

### State Isolation Test
```bash
# Verify tasks don't leak state
uv run python test_state_isolation.py
```

## Documentation

- 📖 **[TESTING.md](TESTING.md)** - Complete testing guide for all benchmarks
- 🐳 **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** - Docker deployment details
- 📝 **[CFBENCH_INTEGRATION.md](CFBENCH_INTEGRATION.md)** - ComplexFuncBench integration details

## Quality Control Pipeline

### Invalid Task Filtering

During the benchmark porting process, we identified and documented problematic test cases in `invalid_tasks.txt` files for each benchmark. These are **automatically filtered out** during evaluation to ensure reliable results.

**Why filter tasks?**
- **Ground truth errors**: Some original benchmarks have incorrect expected outputs
- **Ambiguous prompts**: Tasks with unclear or underspecified instructions lead to inconsistent results
- **Evaluation issues**: Tests with broken logic, missing dependencies, or unreliable scoring

**Implementation**:
```python
# Example from BFCL/bfcl_agent.py
invalid_patterns = []
with open('BFCL/invalid_tasks.txt') as f:
    for line in f:
        task_type, idx = line.strip().rsplit('_', 1)
        invalid_patterns.append((task_type, idx))

# Smart filtering with regex boundary matching
def is_invalid_task(task_id):
    for task_type, idx in invalid_patterns:
        # "multi_turn_base_4" matches ("multi_turn_base", "4") ✓
        # "multi_turn_base_42" does NOT match ("multi_turn_base", "4") ✗
        if re.search(rf'{task_type}_{idx}(?:$|[^0-9])', task_id):
            return True
    return False

test_data = [t for t in test_data if not is_invalid_task(t['id'])]
```

**Files**:
- [`BFCL/invalid_tasks.txt`](BFCL/invalid_tasks.txt) - 139 filtered tasks
- [`ComplexFuncBench/invalid_tasks.txt`](ComplexFuncBench/invalid_tasks.txt) - 161 filtered tasks
- [`Tau2/invalid_tasks.txt`](Tau2/invalid_tasks.txt) - 96 filtered tasks

**Benefit**: You get higher-quality evaluation results without manual debugging of benchmark issues.

## Architecture

The router-based architecture allows a single Docker image to serve multiple benchmarks:

```
User Request (scenario.toml)
       ↓
   [config.benchmark]
       ↓
   router_executor.py
       ↓
   ┌───┴───┬───────┬────────┐
   │       │       │        │
 BFCL    CFB    Tau2    (Future)
```

Each benchmark module:
- Has its own `*_agent.py` with evaluation logic
- Can run standalone via `server.py`
- Shares common infrastructure (messenger, A2A protocol)
- Maintains independent state and configuration

## Contributing

When adding a new benchmark:
1. Create a new module directory (e.g., `NewBench/`)
2. Implement agent with `async def run(message, updater)` method
3. Add benchmark to `router_executor.py`
4. Create `scenario.newbench.toml` and `scenario.newbench-debug.toml`
5. Update this README

## License

[Your License Here]

## Support

- 🐛 Report issues: [GitHub Issues](https://github.com/your-org/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-org/your-repo/discussions)
- 📧 Contact: [your-email@example.com](mailto:your-email@example.com)
