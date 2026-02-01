#!/bin/bash
# Quick local test script
# Usage: ./quick-test.sh [benchmark] [num_tasks]
# Example: ./quick-test.sh bfcl 2

set -e

BENCHMARK=${1:-"cfb"}
NUM_TASKS=${2:-2}
GREEN_PORT=8001
PURPLE_PORT=8000

echo "🧪 Quick Local Test"
echo "===================="
echo "Benchmark: $BENCHMARK"
echo "Num tasks: $NUM_TASKS"
echo ""

# Check if green-agent is running
echo "📡 Checking Green Agent at http://localhost:$GREEN_PORT ..."
if ! curl -s http://localhost:$GREEN_PORT/.well-known/agent-card.json > /dev/null; then
    echo "❌ Green Agent is not running!"
    echo ""
    echo "Please start it in another terminal:"
    echo "  cd $(pwd)"
    echo "  uv run python docker-entrypoint.py --host 127.0.0.1 --port $GREEN_PORT"
    echo ""
    exit 1
fi
echo "✅ Green Agent is running"

# Check if purple-agent is running
echo "📡 Checking Purple Agent at http://localhost:$PURPLE_PORT ..."
if ! curl -s http://localhost:$PURPLE_PORT/.well-known/agent-card.json > /dev/null; then
    echo "❌ Purple Agent is not running!"
    echo ""
    echo "Please start it in another terminal:"
    echo "  cd ../agent-template"
    echo "  uv run agentbeats serve --port $PURPLE_PORT"
    echo ""
    exit 1
fi
echo "✅ Purple Agent is running"
echo ""

# Run test based on benchmark
case $BENCHMARK in
    bfcl)
        echo "🚀 Running BFCL test..."
        uv run python tests/test_bfcl_e2e.py --num-tasks $NUM_TASKS
        ;;
    tau2)
        echo "🚀 Running Tau2 test..."
        uv run python tests/test_tau2_e2e.py --num-tasks $NUM_TASKS
        ;;
    cfb)
        echo "🚀 Running CFB test..."
        uv run python tests/test_cfb_e2e.py --num-tasks $NUM_TASKS
        ;;
    *)
        echo "❌ Unknown benchmark: $BENCHMARK"
        echo "   Available: bfcl, tau2, cfb"
        exit 1
        ;;
esac

echo ""
echo "✅ Test completed!"
