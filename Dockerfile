# Multi-Benchmark Green Agent Docker Image
# Supports BFCL, ComplexFuncBench, and Tau2 benchmarks

FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

# Copy all benchmark modules
COPY BFCL/ BFCL/
COPY ComplexFuncBench/ ComplexFuncBench/
COPY Tau2/ Tau2/

# Copy entrypoint and router
COPY docker-entrypoint.py docker-entrypoint.py
COPY router_executor.py router_executor.py

# Copy dependency files
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

# Copy README for documentation
COPY README.md README.md

# Install dependencies using uv
RUN uv sync --frozen

# Make entrypoint executable
RUN chmod +x docker-entrypoint.py

# Default environment variables (can be overridden at runtime)
ENV HOST=0.0.0.0
ENV PORT=8001

# Expose the default port
EXPOSE 8001

# Set the entrypoint
# AgentBeats platform will call this with --host, --port, and --card-url arguments
ENTRYPOINT ["uv", "run", "python", "docker-entrypoint.py"]

# Default command (can be overridden)
CMD ["--host", "0.0.0.0", "--port", "8001"]
