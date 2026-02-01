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

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    InvalidRequestError,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from a2a.utils import get_message_text, new_agent_text_message, new_task

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Terminal states for task
TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class RouterExecutor(AgentExecutor):
    """Routes evaluation requests to the appropriate benchmark agent"""

    def __init__(self):
        self.name = "Multi-Benchmark Green Agent"
        self.description = (
            "Unified green agent supporting BFCL, ComplexFuncBench, and Tau2 benchmarks. "
            "Specify the benchmark in the config field of the evaluation request."
        )
        # Optional: cache agent instances per context to reduce instantiation overhead
        self.agents = {}  # context_id -> {benchmark -> agent instance}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        Execute method required by A2A framework.

        This method:
        1. Extracts the message and task from the context
        2. Creates or retrieves the task
        3. Parses the benchmark type from the request config
        4. Dynamically loads the appropriate benchmark agent
        5. Delegates to the agent's run() method

        Args:
            context: Request context containing message and task info
            event_queue: Event queue for publishing status updates and artifacts
        """
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        # Create task if it doesn't exist
        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id

        # Create TaskUpdater for compatibility with existing agent.run() interface
        updater = TaskUpdater(event_queue, task.id, context_id)

        try:
            # Parse request to get benchmark type
            input_text = get_message_text(msg)
            request = json.loads(input_text)

            # Extract benchmark type from config
            config = request.get("config", {})
            benchmark = config.get("benchmark", "bfcl")  # Default to BFCL

            logger.info(f"Routing to benchmark: {benchmark}")
            logger.info(f"Config: {config}")

            # Start work
            await updater.start_work()
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Loading {benchmark.upper()} benchmark...")
            )

            # Dynamically load and cache the appropriate agent
            agent_instance = self._load_agent(benchmark, context_id)

            # Delegate to the agent's run method
            await agent_instance.run(msg, updater)

            # Mark as complete if agent didn't reach a terminal state
            if not updater._terminal_state_reached:
                await updater.complete()

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in evaluation request: {e}"
            logger.error(error_msg)
            await updater.failed(new_agent_text_message(error_msg, context_id=context_id, task_id=task.id))

        except Exception as e:
            error_msg = f"Error in router: {e}"
            logger.error(error_msg, exc_info=True)
            await updater.failed(new_agent_text_message(error_msg, context_id=context_id, task_id=task.id))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        Request the agent to cancel an ongoing task.

        Currently not implemented - raises UnsupportedOperationError.
        """
        raise ServerError(error=UnsupportedOperationError())

    def _load_agent(self, benchmark: str, context_id: str):
        """
        Dynamically load the appropriate agent based on benchmark type.

        Agents are cached per context to reduce instantiation overhead while
        maintaining isolation between different evaluation contexts.

        Args:
            benchmark: One of "bfcl", "cfb", "complexfuncbench", or "tau2"
            context_id: Context ID for agent caching

        Returns:
            Agent instance with a run() method

        Raises:
            ValueError: If benchmark is not supported
        """
        benchmark_lower = benchmark.lower()

        # Initialize context cache if needed
        if context_id not in self.agents:
            self.agents[context_id] = {}

        # Return cached agent if available
        if benchmark_lower in self.agents[context_id]:
            logger.info(f"Using cached {benchmark_lower.upper()} agent for context {context_id}")
            return self.agents[context_id][benchmark_lower]

        # Load and cache new agent instance
        if benchmark_lower == "bfcl":
            # Add BFCL to sys.path for relative imports (messenger, bfcl_wrapper, etc.)
            bfcl_path = str(Path(__file__).parent / "BFCL")
            if bfcl_path not in sys.path:
                sys.path.insert(0, bfcl_path)

            from BFCL.bfcl_agent import Agent
            logger.info(f"Loading BFCL agent for context {context_id}")
            agent = Agent()
            self.agents[context_id][benchmark_lower] = agent
            return agent

        elif benchmark_lower in ["cfb", "complexfuncbench"]:
            # Add ComplexFuncBench to sys.path for relative imports
            cfb_path = str(Path(__file__).parent / "ComplexFuncBench")
            if cfb_path not in sys.path:
                sys.path.insert(0, cfb_path)

            from ComplexFuncBench.cfbench_agent import Agent
            logger.info(f"Loading ComplexFuncBench agent for context {context_id}")
            agent = Agent()
            self.agents[context_id][benchmark_lower] = agent
            return agent

        elif benchmark_lower == "tau2":
            # Add Tau2 to sys.path for relative imports
            tau2_path = str(Path(__file__).parent / "Tau2")
            if tau2_path not in sys.path:
                sys.path.insert(0, tau2_path)

            from Tau2.tau2_evaluator import Tau2Evaluator
            logger.info(f"Loading Tau2 evaluator for context {context_id}")
            evaluator = Tau2Evaluator()
            self.agents[context_id][benchmark_lower] = evaluator
            return evaluator

        else:
            supported = ["bfcl", "cfb", "complexfuncbench", "tau2"]
            raise ValueError(
                f"Unknown benchmark: '{benchmark}'. "
                f"Supported benchmarks: {', '.join(supported)}"
            )
