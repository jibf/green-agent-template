"""
Simple API tests for CI/CD pipeline.
Tests basic agent functionality without running full benchmark evaluations.
"""
import pytest
import httpx


def test_agent_card(agent_url):
    """Test that agent card endpoint is accessible"""
    response = httpx.get(f"{agent_url}/.well-known/agent-card.json", timeout=10)
    assert response.status_code == 200

    card = response.json()
    assert "name" in card
    assert "skills" in card
    assert len(card["skills"]) > 0

    # Verify multi-benchmark support
    skill = card["skills"][0]
    assert "BFCL" in skill["tags"]
    assert "ComplexFuncBench" in skill["tags"]
    assert "Tau2" in skill["tags"]


def test_health_endpoint(agent_url):
    """Test that the server is responsive"""
    response = httpx.get(f"{agent_url}/.well-known/agent-card.json", timeout=10)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_router_configuration(agent_url):
    """Test that router is properly configured with all benchmarks"""
    async with httpx.AsyncClient(timeout=10) as client:
        # Test agent card endpoint
        response = await client.get(f"{agent_url}/.well-known/agent-card.json")
        assert response.status_code == 200

        card = response.json()

        # Verify router configuration
        assert "name" in card
        assert card["name"] == "MultiBenchmarkGreenAgent"

        assert "skills" in card
        assert len(card["skills"]) == 1

        skill = card["skills"][0]
        assert skill["id"] == "multi_benchmark_evaluation"

        # Verify all three benchmarks are supported
        tags = skill["tags"]
        assert "BFCL" in tags
        assert "ComplexFuncBench" in tags
        assert "Tau2" in tags
        assert "benchmark" in tags
        assert "evaluation" in tags

        # Verify examples for each benchmark
        examples = skill["examples"]
        assert len(examples) == 3

        # Check BFCL example
        assert any("bfcl" in ex.lower() for ex in examples)

        # Check CFB example
        assert any("cfb" in ex.lower() for ex in examples)

        # Check Tau2 example
        assert any("tau2" in ex.lower() for ex in examples)
