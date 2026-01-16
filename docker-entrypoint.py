#!/usr/bin/env python3
"""
Docker Entrypoint for Multi-Benchmark Green Agent

This script serves as the unified entrypoint for the Docker container.
It starts an A2A server with the RouterExecutor that dynamically routes
evaluation requests to the appropriate benchmark agent.

Usage:
    python docker-entrypoint.py --host 0.0.0.0 --port 8001

Arguments:
    --host: Host address to bind to (default: 0.0.0.0)
    --port: Port to listen on (default: 8001)
    --card-url: URL to advertise in the agent card (optional)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entrypoint function"""
    parser = argparse.ArgumentParser(
        description="Multi-Benchmark Green Agent Server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to listen on (default: 8001)"
    )
    parser.add_argument(
        "--card-url",
        default=None,
        help="URL to advertise in the agent card (optional)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Multi-Benchmark Green Agent")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    if args.card_url:
        logger.info(f"Card URL: {args.card_url}")
    logger.info("Supported benchmarks: BFCL, ComplexFuncBench, Tau2")
    logger.info("=" * 60)

    # Import A2A server components
    import uvicorn
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentSkill,
    )

    # Import the router executor
    from router_executor import RouterExecutor

    # Create executor instance
    executor = RouterExecutor()
    logger.info(f"Initialized {executor.name}")
    logger.info(f"Description: {executor.description}")

    # Create agent skill
    skill = AgentSkill(
        id="multi_benchmark_evaluation",
        name="Multi-Benchmark Evaluation",
        description="Evaluates agents on BFCL, ComplexFuncBench, and Tau2 benchmarks. Specify 'benchmark' in config.",
        tags=["benchmark", "evaluation", "BFCL", "ComplexFuncBench", "Tau2", "function-calling"],
        examples=[
            '{"participants": {"agent": "http://localhost:8000"}, "config": {"benchmark": "bfcl", "test_category": "v3_v4", "num_tasks": 10}}',
            '{"participants": {"agent": "http://localhost:8000"}, "config": {"benchmark": "cfb", "num_tasks": 10}}',
            '{"participants": {"agent": "http://localhost:8000"}, "config": {"benchmark": "tau2", "domain": "airline", "num_tasks": 5}}'
        ]
    )

    # Create agent card
    agent_card = AgentCard(
        name="MultiBenchmarkGreenAgent",
        description="Unified green agent supporting BFCL, ComplexFuncBench, and Tau2 benchmarks with quality control pipeline.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    # Create request handler and app
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore()
    )

    app = A2AStarletteApplication(
        request_handler=request_handler,
        agent_card=agent_card
    )

    # Start the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Agent card available at: http://{args.host}:{args.port}/.well-known/agent-card.json")
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
