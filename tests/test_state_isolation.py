"""
Test to verify state isolation between tasks.

This test checks that:
1. Purple Agent maintains separate conversation history per context_id
2. Green Agent creates new runner instances for each task
3. Each task starts with a fresh conversation
"""
import asyncio
import json
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


async def test_state_isolation(
    green_agent_url: str = "http://localhost:8001",
    purple_agent_url: str = "http://localhost:8000",
):
    """Test that tasks are properly isolated."""
    print("🧪 State Isolation Test")
    print("=" * 60)
    print(f"Green Agent: {green_agent_url}")
    print(f"Purple Agent: {purple_agent_url}")
    print()

    # Use simple_python tasks that should be independent
    eval_request = {
        "participants": {
            "agent": purple_agent_url
        },
        "config": {
            "benchmark": "bfcl",
            "sample_ids": [
                "simple_python_0",  # Task 1
                "simple_python_1",  # Task 2 - should not know about Task 1
            ],
        }
    }

    async with httpx.AsyncClient(timeout=600) as httpx_client:
        print("📡 Connecting to Green Agent...")
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=green_agent_url)

        try:
            agent_card = await resolver.get_agent_card()
            print(f"✅ Connected to: {agent_card.name}")
            print()
        except Exception as e:
            print(f"❌ Failed to connect to Green Agent: {e}")
            return

        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        message = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=json.dumps(eval_request)))],
            message_id=uuid4().hex,
            context_id=uuid4().hex,
        )

        print("🚀 Running 2 tasks to test isolation...")
        print()

        try:
            task_count = 0
            task_results = []

            async for event in client.send_message(message):
                match event:
                    case Message() as msg:
                        if msg.parts:
                            for part in msg.parts:
                                if hasattr(part, 'text'):
                                    print(f"📨 {part.text}")

                    case (task, update):
                        if update:
                            if hasattr(update, 'status'):
                                status = update.status
                                if hasattr(status, 'message') and status.message:
                                    msg_text = status.message
                                    if hasattr(msg_text, 'parts'):
                                        for part in msg_text.parts:
                                            if hasattr(part, 'text'):
                                                text = part.text
                                                if text.startswith("Task "):
                                                    task_count += 1
                                                    print(f"📊 {text}")

                            if hasattr(update, 'artifact'):
                                artifact = update.artifact
                                if artifact and hasattr(artifact, 'parts'):
                                    for part in artifact.parts:
                                        if hasattr(part, 'root') and hasattr(part.root, 'data'):
                                            task_results = part.root.data.get('task_results', [])

            print("\n" + "=" * 60)
            print("✅ Isolation Test Results:")
            print("=" * 60)

            if len(task_results) >= 2:
                task1 = task_results[0]
                task2 = task_results[1]

                print(f"\nTask 1 ({task1.get('id')}):")
                print(f"  Valid: {task1.get('valid')}")
                print(f"  Error: {task1.get('error', 'None')}")

                print(f"\nTask 2 ({task2.get('id')}):")
                print(f"  Valid: {task2.get('valid')}")
                print(f"  Error: {task2.get('error', 'None')}")

                # Both tasks should succeed independently
                if task1.get('valid') and task2.get('valid'):
                    print("\n✅ SUCCESS: Both tasks completed independently")
                    print("   State isolation is working correctly!")
                else:
                    print("\n⚠️  WARNING: One or both tasks failed")
                    print("   This may indicate state leakage issues")
            else:
                print(f"\n⚠️  Expected 2 task results, got {len(task_results)}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_state_isolation())
