"""
End-to-end test for BFCL integration.

Usage:
    # Start Purple Agent in one terminal:
    cd ../agent-template && uv run agentbeats serve --port 8000

    # Start Green Agent in another terminal:
    cd green-agent-template && uv run src/server.py --port 8001

    # Run this test:
    python test_bfcl_e2e.py --num-tasks 2
"""
import asyncio
import json
from uuid import uuid4
import argparse

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


def validate_bfcl_result(result_data: dict) -> bool:
    """Validate BFCL result structure matches leaderboard-queries.json requirements."""
    required_fields = {
        "accuracy": float,
        "correct_count": int,
        "total_count": int,
        "category_stats": dict,
    }

    print("\n🔍 Validating JSON structure...")

    for field, expected_type in required_fields.items():
        if field not in result_data:
            print(f"   ❌ Missing field: {field}")
            return False
        if not isinstance(result_data[field], expected_type):
            print(f"   ❌ Wrong type for {field}: expected {expected_type}, got {type(result_data[field])}")
            return False

    # Validate category_stats structure
    for category, stats in result_data["category_stats"].items():
        if not isinstance(stats, dict):
            print(f"   ❌ category_stats[{category}] should be dict")
            return False
        if "success" not in stats or "total" not in stats:
            print(f"   ❌ category_stats[{category}] missing 'success' or 'total'")
            return False

    print("   ✅ JSON structure is valid")
    return True


test_categorys = [
    # "simple_python",
    # "simple_java",
    # "simple_javascript",
    # "multiple",
    # "parallel",
    # "parallel_multiple",
    # "irrelevance",

    # "live_simple",
    # "live_multiple",
    # "live_parallel",
    # "live_parallel_multiple",
    # "live_irrelevance",
    # "live_relevance",

    "multi_turn_base", #
    "multi_turn_miss_func", #
    "multi_turn_miss_param", #
    "multi_turn_long_context", #

    # "web_search_base", #
    # "web_search_no_snippet", #

    # "memory_kv",
    # "memory_vector",
    # "memory_rec_sum",
]


async def test_bfcl_e2e(
    green_agent_url: str = "http://localhost:8001",
    purple_agent_url: str = "http://localhost:8000",
    num_tasks: int = 2,
    test_category: str = None,
    sample_ids: list[str] = None,
):
    """Test BFCL integration end-to-end."""
    print("🧪 BFCL End-to-End Test")
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
            "benchmark": "bfcl",  # Explicitly specify BFCL
            "num_tasks": num_tasks,
            "test_category": test_category,
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
                is_valid = validate_bfcl_result(final_result)

                print("\n📊 Quick Summary:")
                if 'accuracy' in final_result:
                    print(f"   Accuracy: {final_result['accuracy']:.1f}%")
                if 'correct_count' in final_result:
                    print(f"   Correct: {final_result['correct_count']}/{final_result.get('total_count', 0)}")
                if 'task_results' in final_result:
                    print(f"\n   Task Results:")
                    for task_result in final_result['task_results']:
                        status = "✓" if task_result.get('valid') else "✗"
                        error_msg = task_result.get('error', 'Success')
                        if error_msg and isinstance(error_msg, list):
                            error_msg = '; '.join(str(e) for e in error_msg[:2])
                        print(f"     {status} {task_result.get('id')}: {error_msg}")

                # Save result to file for inspection
                result_file = "test_bfcl_result.json"
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
    parser = argparse.ArgumentParser(description="BFCL end-to-end test")
    parser.add_argument("--green-agent", default="http://localhost:8001", help="Green Agent URL")
    parser.add_argument("--purple-agent", default="http://localhost:8000", help="Purple Agent URL")
    parser.add_argument("--num-tasks", type=int, default=-1, help="Number of tasks to run")
    parser.add_argument("--test-category", help="Test category (e.g., single_turn, multi_turn, simple_python)")
    parser.add_argument("--sample-ids", nargs="+", help="Specific sample IDs to test (e.g., parallel_0 parallel_3 multiple_5)")

    args = parser.parse_args()

    # sample_ids = []
    # for category in test_categorys:
    #     for i in range(3):
    #         sample_ids.append(f"{category}_{i}")
    sample_ids = [
        "live_simple_0-0-0", "live_simple_1-1-0", "live_simple_2-2-0",
        "live_multiple_1-0-1", "live_multiple_0-0-0",
        "live_parallel_1-0-1", "live_parallel_0-0-0", "live_parallel_3-0-3",
        "live_parallel_multiple_0-0-0", "live_parallel_multiple_1-1-0",
        "live_irrelevance_0-0-0", "live_irrelevance_1-0-1",
        "live_relevance_0-0-0", "live_relevance_1-1-0",
        "memory_kv_prereq_32-notetaker-0",
    ]


    await test_bfcl_e2e(
        green_agent_url=args.green_agent,
        purple_agent_url=args.purple_agent,
        num_tasks=args.num_tasks,
        test_category=args.test_category,
        sample_ids=args.sample_ids,
    )


if __name__ == "__main__":
    asyncio.run(main())
