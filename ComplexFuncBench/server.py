import argparse
import sys
from pathlib import Path
import uvicorn
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env from project root
load_dotenv(project_root / ".env")

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

# Import executor from same directory
from ComplexFuncBench.executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Fill in your agent card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    
    skill = AgentSkill(
        id="cfbench_evaluation",
        name="CFBench Evaluation",
        description="Evaluates agents on CFBench tasks.",
        tags=["benchmark", "evaluation", "CFBench"],
        examples=[
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"num_tasks": 10, "debug": false}}',
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"debug": true, "enable_response_eval": true}}',
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"sample_ids": ["Car-Rental-0", "Hotel-0"]}}'
        ]
    )

    agent_card = AgentCard(
        name="CFBenchEvaluator",
        description="Complex Function Calling Benchmark evaluator - tests agents on complex function calling tasks.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()