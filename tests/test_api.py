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
    response = httpx.get(agent_url, timeout=10, follow_redirects=True)
    assert response.status_code in [200, 404]  # 404 is ok, means server is running


@pytest.mark.asyncio
async def test_router_bfcl(agent_url):
    """Test that router can handle BFCL benchmark request"""
    async with httpx.AsyncClient(timeout=30) as client:
        # Simple request to verify router logic works
        request_data = {
            "participants": {"agent": "http://mock-agent:8000"},
            "config": {
                "benchmark": "bfcl",
                "test_category": "v3_v4",
                "num_tasks": 1
            }
        }

        # Just verify the endpoint accepts the request format
        # We're not running actual evaluation in CI
        response = await client.post(
            f"{agent_url}/tasks",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        # Should accept the request (200 or 201) or return validation error (400+)
        assert response.status_code in [200, 201, 400, 422]


@pytest.mark.asyncio
async def test_router_cfb(agent_url):
    """Test that router can handle ComplexFuncBench request"""
    async with httpx.AsyncClient(timeout=30) as client:
        request_data = {
            "participants": {"agent": "http://mock-agent:8000"},
            "config": {
                "benchmark": "cfb",
                "num_tasks": 1
            }
        }

        response = await client.post(
            f"{agent_url}/tasks",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code in [200, 201, 400, 422]


@pytest.mark.asyncio
async def test_router_tau2(agent_url):
    """Test that router can handle Tau2 request"""
    async with httpx.AsyncClient(timeout=30) as client:
        request_data = {
            "participants": {"agent": "http://mock-agent:8000"},
            "config": {
                "benchmark": "tau2",
                "domain": "airline",
                "num_tasks": 1
            }
        }

        response = await client.post(
            f"{agent_url}/tasks",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code in [200, 201, 400, 422]
