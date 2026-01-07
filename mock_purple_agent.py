"""
Mock Purple Agent for ComplexFuncBench Testing

This is a simple mock agent that returns predetermined responses to test
the green agent integration without requiring a real LLM.

Level 2 Testing: Tests that the overall flow works correctly.
"""
import argparse
import json
import re
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from loguru import logger


def prepare_agent_card(url: str) -> AgentCard:
    """Create the agent card for the mock purple agent."""
    skill = AgentSkill(
        id="function_calling",
        name="Function Calling",
        description="Mock agent for ComplexFuncBench testing",
        tags=["benchmark", "cfbench", "mock"],
        examples=[],
    )
    return AgentCard(
        name="mock_cfbench_agent",
        description="Mock agent for ComplexFuncBench evaluation testing",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


class MockCFBenchExecutor(AgentExecutor):
    """
    Mock executor that returns simple predetermined responses.

    Strategy:
    - First call: Return a function call (Search_Car_Location)
    - Second call: Return a final response

    This tests the basic flow without requiring real function calling logic.
    """

    def __init__(self):
        self.ctx_id_to_call_count: dict[str, int] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        logger.info(f"Received input (first 200 chars): {user_input[:200]}...")

        # Track call count per conversation
        if context.context_id not in self.ctx_id_to_call_count:
            self.ctx_id_to_call_count[context.context_id] = 0

        call_count = self.ctx_id_to_call_count[context.context_id]
        self.ctx_id_to_call_count[context.context_id] += 1

        # Parse tools from the message to get available function names
        available_tools = self._parse_tools(user_input)

        # Generate response based on call count
        if call_count == 0:
            # First call: return a function call
            # Use the first available tool if any, otherwise use a generic one
            if available_tools:
                tool_name = available_tools[0]
                # Try to extract a reasonable parameter from the query
                query = self._extract_query(user_input)

                response = {
                    "function_calls": [
                        {
                            "name": tool_name,
                            "arguments": {"query": query}
                        }
                    ]
                }
            else:
                # Fallback if no tools found
                response = {
                    "function_calls": [
                        {
                            "name": "Search_Car_Location",
                            "arguments": {"query": "San Diego"}
                        }
                    ]
                }
            response_text = json.dumps(response, ensure_ascii=False)
            logger.info(f"Returning function call: {response_text}")
        else:
            # Second+ call: return final response
            response = {
                "response": "Based on the search results, I have found the requested information."
            }
            response_text = json.dumps(response, ensure_ascii=False)
            logger.info(f"Returning final response: {response_text}")

        # Send response back via A2A
        await event_queue.enqueue_event(
            new_agent_text_message(response_text, context_id=context.context_id)
        )

    def _parse_tools(self, message: str) -> list[str]:
        """Extract tool names from the message."""
        try:
            # Look for tool definitions in JSON format
            # Pattern: "name": "ToolName"
            tool_names = re.findall(r'"name":\s*"([^"]+)"', message)
            # Filter out common non-tool names
            tool_names = [t for t in tool_names if not t.startswith('call_')]
            return tool_names[:5]  # Return first 5 tools
        except Exception as e:
            logger.warning(f"Failed to parse tools: {e}")
            return []

    def _extract_query(self, message: str) -> str:
        """Extract a query string from the user message."""
        try:
            # Look for "User query:" pattern
            match = re.search(r'User query:\s*(.+?)(?:\n|$)', message)
            if match:
                query = match.group(1).strip()
                # Take first few words
                words = query.split()[:3]
                return ' '.join(words)
            return "test query"
        except Exception as e:
            logger.warning(f"Failed to extract query: {e}")
            return "test query"

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(
        description="Run the mock purple agent for ComplexFuncBench testing."
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9019, help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url", type=str, help="External URL for the agent card"
    )
    args = parser.parse_args()

    logger.info("Starting mock CFBench purple agent...")
    logger.info("This is a MOCK agent for testing - it returns predetermined responses")

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=MockCFBenchExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
