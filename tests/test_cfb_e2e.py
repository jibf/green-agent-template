"""
End-to-end test for ComplexFuncBench integration.

Usage:
    # Start Purple Agent in one terminal:
    cd ../agent-template && uv run agentbeats serve --port 8000

    # Start Green Agent in another terminal:
    cd green-agent-template && uv run python docker-entrypoint.py --port 8001

    # Run this test:
    python test_cfb_e2e.py --num-tasks 2
"""
import asyncio
import json
from uuid import uuid4
import argparse

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


def validate_cfb_result(result_data: dict) -> bool:
    """Validate ComplexFuncBench result structure matches leaderboard-queries.json requirements."""
    required_fields = {
        "overall_success_rate": (int, float),
        "overall_call_accuracy": (int, float),
        "successful_samples": int,
        "total_samples": int,
        "domain_stats": dict,
    }

    print("\n🔍 Validating JSON structure...")

    for field, expected_types in required_fields.items():
        if field not in result_data:
            print(f"   ❌ Missing field: {field}")
            return False
        if not isinstance(result_data[field], expected_types):
            print(f"   ❌ Wrong type for {field}: expected {expected_types}, got {type(result_data[field])}")
            return False

    # Validate domain_stats structure
    for domain, stats in result_data["domain_stats"].items():
        if not isinstance(stats, dict):
            print(f"   ❌ domain_stats[{domain}] should be dict")
            return False
        required_domain_fields = ["success", "total", "correct_calls", "total_calls"]
        for df in required_domain_fields:
            if df not in stats:
                print(f"   ❌ domain_stats[{domain}] missing '{df}'")
                return False

    print("   ✅ JSON structure is valid")
    return True


async def test_cfb_e2e(
    green_agent_url: str = "http://localhost:8001",
    purple_agent_url: str = "http://localhost:8000",
    num_tasks: int = 2,
    sample_ids: list[str] = None,
):
    """Test ComplexFuncBench integration end-to-end."""
    print("🧪 ComplexFuncBench End-to-End Test")
    print("=" * 60)
    print(f"Green Agent: {green_agent_url}")
    print(f"Purple Agent: {purple_agent_url}")
    if sample_ids:
        print(f"Sample IDs: {sample_ids}")
    else:
        print(f"Tasks: {num_tasks}")
    print()

    # Prepare eval request
    eval_request = {
        "participants": {
            "agent": purple_agent_url
        },
        "config": {
            "benchmark": "cfb",
            "num_tasks": num_tasks,
            "sample_ids": sample_ids,
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

        print("🚀 Starting evaluation...")
        print()

        try:
            final_result = None

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
                                                print(f"📊 {part.text}")

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

            print()
            print("=" * 60)
            print("📊 Final Results:")
            print("=" * 60)

            if final_result:
                # Validate JSON structure
                is_valid = validate_cfb_result(final_result)

                if 'overall_success_rate' in final_result:
                    print(f"   Success Rate: {final_result['overall_success_rate']:.1f}%")
                if 'overall_call_accuracy' in final_result:
                    print(f"   Call Accuracy: {final_result['overall_call_accuracy']:.1f}%")
                if 'successful_samples' in final_result:
                    print(f"   Successful: {final_result['successful_samples']}/{final_result.get('total_samples', 0)}")

                if 'task_results' in final_result:
                    print(f"\n   Task Results:")
                    for task_result in final_result['task_results']:
                        status = "✓" if task_result.get('success') else "✗"
                        task_id = task_result.get('id', 'unknown')
                        print(f"     {status} {task_id}")

                # Save result to file for inspection
                result_file = "test_cfb_result.json"
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
    parser = argparse.ArgumentParser(description="ComplexFuncBench end-to-end test")
    parser.add_argument("--green-agent", default="http://localhost:8001", help="Green Agent URL")
    parser.add_argument("--purple-agent", default="http://localhost:8000", help="Purple Agent URL")
    parser.add_argument("--num-tasks", type=int, default=2, help="Number of tasks to run")
    parser.add_argument("--sample-ids", nargs="+", help="Specific sample IDs to test")

    args = parser.parse_args()

    await test_cfb_e2e(
        green_agent_url=args.green_agent,
        purple_agent_url=args.purple_agent,
        num_tasks=args.num_tasks,
        sample_ids=args.sample_ids,
    )


if __name__ == "__main__":
    asyncio.run(main())
