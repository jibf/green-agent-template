"""
End-to-end test for Tau2 integration.

Usage:
    # Start Purple Agent in one terminal:
    cd ../agent-template && uv run src/server.py --port 8000

    # Start Green Agent in another terminal:
    cd green-agent-template && uv run Tau2/tau2_evaluator.py --host 127.0.0.1 --port 9009

    # Run this test:
    python test_tau2_e2e.py --num-tasks 2
"""
import asyncio
import json
from uuid import uuid4
import argparse

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


def validate_tau2_result(result_data: dict) -> bool:
    """Validate Tau2 result structure matches leaderboard-queries.json requirements."""
    required_fields = {
        "domain": str,
        "pass_rate": (int, float),
        "score": (int, float),
        "max_score": int,
        "time_used": (int, float),
    }

    print("\n🔍 Validating JSON structure...")

    for field, expected_types in required_fields.items():
        if field not in result_data:
            print(f"   ❌ Missing field: {field}")
            return False
        if not isinstance(result_data[field], expected_types):
            print(f"   ❌ Wrong type for {field}: expected {expected_types}, got {type(result_data[field])}")
            return False

    # If domain is 'all', check for domain-specific stats
    if result_data["domain"] == "all":
        for domain_name in ["airline_stats", "retail_stats", "telecom_stats"]:
            if domain_name in result_data:
                stats = result_data[domain_name]
                if not isinstance(stats, dict) or "pass_rate" not in stats:
                    print(f"   ❌ {domain_name} should have 'pass_rate' field")
                    return False

    print("   ✅ JSON structure is valid")
    return True


async def test_tau2_e2e(
    green_agent_url: str = "http://localhost:9009",
    purple_agent_url: str = "http://localhost:8000",
    domain: str = "all",
    num_tasks: int = 2,
):
    """Test Tau2 integration end-to-end."""
    print("🧪 Tau2 End-to-End Test")
    print("=" * 60)
    print(f"Green Agent: {green_agent_url}")
    print(f"Purple Agent: {purple_agent_url}")
    print(f"Domain: {domain}")
    print(f"Tasks: {num_tasks}")
    print()

    # Prepare eval request
    eval_request = {
        "participants": {
            "agent": purple_agent_url
        },
        "config": {
            "benchmark": "tau2",
            "domain": domain,
            "num_tasks": num_tasks,
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

        # Create client
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        # Create message
        message = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=json.dumps(eval_request)))],
            message_id=uuid4().hex,
            context_id=uuid4().hex,
        )

        print("🚀 Sending evaluation request...")
        print()

        try:
            event_count = 0
            last_status = None
            final_result = None

            async for event in client.send_message(message):
                event_count += 1

                match event:
                    case Message() as msg:
                        print(f"📨 Message from Green Agent")
                        if msg.parts:
                            for part in msg.parts:
                                if hasattr(part, 'text'):
                                    print(f"   {part.text}")

                    case (task, update):
                        if update:
                            if hasattr(update, 'status'):
                                status = update.status
                                last_status = status.state
                                if hasattr(status, 'message') and status.message:
                                    msg_text = status.message
                                    if hasattr(msg_text, 'parts'):
                                        for part in msg_text.parts:
                                            if hasattr(part, 'text'):
                                                print(f"📊 {part.text}")
                                    else:
                                        print(f"📊 Status: {msg_text}")

                            if hasattr(update, 'artifact'):
                                artifact = update.artifact
                                if artifact:
                                    print(f"\n📦 Artifact received: {artifact.name}")
                                    if hasattr(artifact, 'parts'):
                                        for part in artifact.parts:
                                            if hasattr(part, 'root'):
                                                root = part.root
                                                if hasattr(root, 'text'):
                                                    print("\n" + "=" * 60)
                                                    print("SUMMARY:")
                                                    print("=" * 60)
                                                    print(root.text)
                                                if hasattr(root, 'data'):
                                                    final_result = root.data
                                                    print("\n" + "=" * 60)
                                                    print("RESULT DATA:")
                                                    print("=" * 60)
                                                    print(json.dumps(root.data, indent=2))

                    case _:
                        print(f"⚠️  Unknown event: {type(event)}")

            print("\n" + "=" * 60)
            print(f"✅ Test completed!")
            print(f"   Total events: {event_count}")
            print(f"   Final status: {last_status}")
            print("=" * 60)

            # Print summary if we got results
            if final_result:
                # Validate JSON structure
                is_valid = validate_tau2_result(final_result)

                print("\n📊 Quick Summary:")
                if 'pass_rate' in final_result:
                    print(f"   Pass Rate: {final_result['pass_rate']:.1f}%")
                if 'score' in final_result:
                    print(f"   Score: {final_result['score']}/{final_result.get('max_score', '?')}")
                if 'time_used' in final_result:
                    print(f"   Time Used: {final_result['time_used']:.2f}s")
                if 'average_reward' in final_result:
                    print(f"   Average Reward: {final_result['average_reward']:.2f}")
                if 'average_cost' in final_result:
                    print(f"   Average Cost: ${final_result['average_cost']:.2f}")
                if 'success_rate' in final_result:
                    print(f"   Success Rate: {final_result['success_rate']:.1f}%")

                if 'task_results' in final_result:
                    print(f"\n   Task Results:")
                    for task_result in final_result['task_results']:
                        reward = task_result.get('reward', 0)
                        cost = task_result.get('cost', 0)
                        status = "✓" if reward > 0 else "✗"
                        print(f"     {status} Task {task_result.get('task_index', '?')}: "
                              f"Reward={reward:.2f}, Cost=${cost:.2f}")

                # Save result to file for inspection
                result_file = "test_tau2_result.json"
                with open(result_file, 'w') as f:
                    json.dump(final_result, f, indent=2)
                print(f"\n💾 Full result saved to: {result_file}")

                if not is_valid:
                    print("\n⚠️  Warning: Result structure does not match leaderboard requirements!")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    parser = argparse.ArgumentParser(description="Tau2 end-to-end test")
    parser.add_argument("--green-agent", default="http://localhost:8001", help="Green Agent URL")
    parser.add_argument("--purple-agent", default="http://localhost:8000", help="Purple Agent URL")
    parser.add_argument("--domain", default="airline",
                       choices=["airline", "retail", "telecom", "all"],
                       help="Tau2 domain")
    parser.add_argument("--num-tasks", type=int, default=2, help="Number of tasks to run")

    args = parser.parse_args()

    await test_tau2_e2e(
        green_agent_url=args.green_agent,
        purple_agent_url=args.purple_agent,
        domain=args.domain,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    asyncio.run(main())
