"""
Router Executor for Multi-Benchmark Green Agent

This executor dynamically routes evaluation requests to the appropriate benchmark agent
based on the 'benchmark' field in the request config.

Supported benchmarks:
- bfcl: Berkeley Function Calling Leaderboard
- cfb/complexfuncbench: ComplexFuncBench
- tau2: Tau2 customer service benchmark
"""

import json
import logging
import sys
from pathlib import Path
from typing import AsyncIterator

from a2a.server.tasks import TaskUpdater
from a2a.types import Artifact, Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class RouterExecutor:
    """Routes evaluation requests to the appropriate benchmark agent"""

    def __init__(self):
        self.name = "Multi-Benchmark Green Agent"
        self.description = (
            "Unified green agent supporting BFCL, ComplexFuncBench, and Tau2 benchmarks. "
            "Specify the benchmark in the config field of the evaluation request."
        )

    async def run(
        self, message: Message, updater: TaskUpdater
    ) -> AsyncIterator[Artifact]:
        """
        Route to the appropriate benchmark agent based on config.benchmark

        Expected request format:
        {
            "participants": {
                "agent": "http://purple-agent:8000"
            },
            "config": {
                "benchmark": "bfcl",  // or "cfb" or "tau2"
                // ... other benchmark-specific config
            }
        }
        """
        try:
            input_text = get_message_text(message)
            request = json.loads(input_text)

            # Extract benchmark type from config
            config = request.get("config", {})
            benchmark = config.get("benchmark", "bfcl")  # Default to BFCL

            logger.info(f"Routing to benchmark: {benchmark}")
            logger.info(f"Config: {config}")

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Loading {benchmark.upper()} benchmark...")
            )

            # Dynamically import and instantiate the appropriate agent
            agent_instance = self._load_agent(benchmark)

            # Delegate to the agent's run method
            async for artifact in agent_instance.run(message, updater):
                yield artifact

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in evaluation request: {e}"
            logger.error(error_msg)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_msg)
            )
            raise

        except Exception as e:
            error_msg = f"Error in router: {e}"
            logger.error(error_msg, exc_info=True)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_msg)
            )
            raise

    def _load_agent(self, benchmark: str):
        """
        Dynamically load the appropriate agent based on benchmark type

        Args:
            benchmark: One of "bfcl", "cfb", "complexfuncbench", or "tau2"

        Returns:
            Agent instance with a run() method

        Raises:
            ValueError: If benchmark is not supported
        """
        benchmark_lower = benchmark.lower()

        if benchmark_lower == "bfcl":
            from BFCL.bfcl_agent import Agent
            logger.info("Loaded BFCL agent")
            return Agent()

        elif benchmark_lower in ["cfb", "complexfuncbench"]:
            from ComplexFuncBench.cfbench_agent import Agent
            logger.info("Loaded ComplexFuncBench agent")
            return Agent()

        elif benchmark_lower == "tau2":
            from Tau2.tau2_evaluator import Tau2Evaluator
            logger.info("Loaded Tau2 evaluator")
            return Tau2Evaluator()

        else:
            supported = ["bfcl", "cfb", "complexfuncbench", "tau2"]
            raise ValueError(
                f"Unknown benchmark: '{benchmark}'. "
                f"Supported benchmarks: {', '.join(supported)}"
            )
