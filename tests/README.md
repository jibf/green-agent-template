# Tests

End-to-end tests for the multi-benchmark green agent.

## Test Files

| File | Benchmark | Description |
|------|-----------|-------------|
| `test_bfcl_e2e.py` | BFCL | Berkeley Function Calling Leaderboard tests with JSON validation |
| `test_cfb_e2e.py` | ComplexFuncBench | Travel planning with Booking.com API tests with JSON validation |
| `test_tau2_e2e.py` | Tau2 | Customer service conversation tests with JSON validation |
| `test_state_isolation.py` | All | Verify state isolation between tasks |

**Note**: All E2E tests automatically validate that the result JSON structure matches the requirements in `leaderboard-queries.json` and save the full result to a file for inspection.

## Prerequisites

Start both agents before running tests:

**Terminal 1 - Purple Agent:**
```bash
cd ../agent-template
uv run agentbeats serve --port 8000
```

**Terminal 2 - Green Agent:**
```bash
cd green-agent-template
uv run python docker-entrypoint.py --port 8001
```

## Running Tests

### Quick Test (Recommended)

From the project root:
```bash
./quick-test.sh bfcl 2    # Test BFCL with 2 tasks
./quick-test.sh cfb 2     # Test ComplexFuncBench with 2 tasks
./quick-test.sh tau2 2    # Test Tau2 with 2 tasks
```

### Direct Test Execution

**BFCL:**
```bash
# Test with number of tasks
python tests/test_bfcl_e2e.py --num-tasks 5

# Test specific samples
python tests/test_bfcl_e2e.py --sample-ids simple_python_0 simple_python_1

# Test specific category
python tests/test_bfcl_e2e.py --test-category multi_turn_base --num-tasks 3
```

**ComplexFuncBench:**
```bash
# Test with number of tasks
python tests/test_cfb_e2e.py --num-tasks 5

# Test specific samples
python tests/test_cfb_e2e.py --sample-ids Car-Rental-0 Hotel-0
```

**Tau2:**
```bash
# Test with number of tasks
python tests/test_tau2_e2e.py --num-tasks 5

# Test specific domain
python tests/test_tau2_e2e.py --domain airline --num-tasks 3
```

**State Isolation:**
```bash
python tests/test_state_isolation.py
```

## Test Parameters

### Common Parameters

- `--green-agent URL`: Green agent URL (default: http://localhost:8001)
- `--purple-agent URL`: Purple agent URL (default: http://localhost:8000)
- `--num-tasks N`: Number of tasks to run

### BFCL Specific

- `--test-category CATEGORY`: Test category (e.g., simple_python, multi_turn_base)
- `--sample-ids ID [ID ...]`: Specific sample IDs to test

### ComplexFuncBench Specific

- `--sample-ids ID [ID ...]`: Specific sample IDs to test

### Tau2 Specific

- `--domain DOMAIN`: Domain to test (airline, retail, telecom, mock, all)

## Troubleshooting

**Connection Failed:**
- Verify both green and purple agents are running
- Check ports 8000 and 8001 are not in use by other services

**Import Errors:**
- Run tests with `uv run python` to use the correct virtual environment
- Ensure all dependencies are installed: `uv sync`

**Test Failures:**
- Check agent logs for detailed error messages
- Verify API keys are set in `.env` file (for BFCL web search, CFB real API)
