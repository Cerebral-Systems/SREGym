"""
Cerebral *perception-world-model* agent driver for SREGym.

The reasoning agent itself lives in a sibling repository
(`../cerebral/perception-world-model/agents`, the ``cerebral_agents`` package).
That package ships an SREGym adapter (`SREGymAgentRuntime`) but no
conductor-facing entry point — it expects a host to (1) give it an MCP gateway,
(2) hand it the current ``app_info`` / ``problem``, and (3) drive its
``diagnose`` / ``mitigate`` stages. This module is that host: the equivalent of
``clients/claudecode/driver.py`` for the Cerebral agent.

Because the ``cerebral_agents`` package lives outside the SREGym repo it is not
baked into the agent container image, so this agent is registered with
``container_isolation: false`` and runs on the host (where the sibling repo and
the SREGym virtualenv are available). The path to the package can be overridden
with ``CEREBRAL_AGENTS_PATH``.

Relevant env vars:
  AGENT_MODEL_ID            model id (also accepts CEREBRAL_MODEL); default claude-sonnet-4-6
  CEREBRAL_BRAIN            optional: force cerebral's own provider selection
                            (stub | anthropic | claude | generic)
  CEREBRAL_AGENTS_PATH      path to the cerebral_agents package parent dir
  CEREBRAL_MAX_STEPS        max agent steps per stage (default 40)
  CEREBRAL_SREGYM_MAX_TOKENS model max tokens (default 1800)
  API_HOSTNAME / API_PORT   conductor REST API (default localhost:8000)
  MCP_SERVER_PORT           SREGym MCP server port (default 9954)
  AGENT_LOGS_DIR            directory for trajectory / results output
  SSE_READ_TIMEOUT          MCP SSE read timeout in seconds (default 3600; <0 = none)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import requests
from fastmcp import Client
from fastmcp.client import SSETransport

# Make the SREGym repo importable (clients.*, logger, configs, ...).
sregym_root = Path(__file__).resolve().parents[2]
if str(sregym_root) not in sys.path:
    sys.path.insert(0, str(sregym_root))

from clients.stratus.configs.langgraph_tool_configs import LanggraphToolConfig  # noqa: E402
from logger import init_logger  # noqa: E402

init_logger()
logger = logging.getLogger("all.perception_world_model.driver")


def _load_cerebral_agents():
    """Put the sibling cerebral_agents package on sys.path and import the bits we need."""
    candidates = []
    if os.environ.get("CEREBRAL_AGENTS_PATH"):
        candidates.append(Path(os.environ["CEREBRAL_AGENTS_PATH"]))
    candidates.append(sregym_root.parent / "cerebral" / "perception-world-model" / "agents")

    for candidate in candidates:
        if (candidate / "cerebral_agents").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            logger.info("Loading cerebral_agents from %s", candidate)
            break
    else:
        raise RuntimeError(
            "Could not locate the cerebral_agents package. Set CEREBRAL_AGENTS_PATH to the "
            "directory containing it (e.g. ../cerebral/perception-world-model/agents). "
            f"Looked in: {[str(c) for c in candidates]}"
        )

    from cerebral_agents.provider import (  # noqa: E402
        AnthropicProvider,
        FallbackProvider,
        ModelProvider,
        StubProvider,
    )
    from cerebral_agents.sregym_agent_runtime import (  # noqa: E402
        SREGymAgentRuntime,
        SREGymMcpTool,
        SREGymStageResult,
        provider_from_env,
    )

    return {
        "AnthropicProvider": AnthropicProvider,
        "FallbackProvider": FallbackProvider,
        "ModelProvider": ModelProvider,
        "StubProvider": StubProvider,
        "SREGymAgentRuntime": SREGymAgentRuntime,
        "SREGymMcpTool": SREGymMcpTool,
        "SREGymStageResult": SREGymStageResult,
        "provider_from_env": provider_from_env,
    }


def get_api_base_url() -> str:
    host = os.getenv("API_HOSTNAME", "localhost")
    port = os.getenv("API_PORT", "8000")
    return f"http://{host}:{port}"


def get_stage() -> str | None:
    try:
        resp = requests.get(f"{get_api_base_url()}/status", timeout=30)
        resp.raise_for_status()
        return resp.json().get("stage")
    except Exception as e:  # noqa: BLE001
        logger.debug("status check failed: %s", e)
        return None


def wait_for_ready_stage(timeout: int = 300) -> str:
    """Block until the conductor reaches a submission-ready stage."""
    allowed = {"diagnosis", "mitigation"}
    start = time.time()
    logger.info("Waiting for conductor to reach a submission-ready stage...")
    while time.time() - start < timeout:
        stage = get_stage()
        if stage in allowed:
            logger.info("Conductor ready at stage: %s", stage)
            return stage
        time.sleep(1)
    raise TimeoutError(f"Conductor did not reach a ready stage within {timeout}s")


def get_app_info() -> dict:
    resp = requests.get(LanggraphToolConfig().benchmark_app_info_url, timeout=30)
    resp.raise_for_status()
    info = resp.json()
    logger.info("App info: %s", info)
    return info


def get_problem() -> dict:
    resp = requests.get(LanggraphToolConfig().benchmark_current_problem, timeout=30)
    resp.raise_for_status()
    problem = resp.json()
    logger.info("Problem: %s", problem)
    return problem


def submit(solution: str) -> None:
    url = LanggraphToolConfig().benchmark_submit_url
    logger.info("Submitting to %s (%d chars)", url, len(solution or ""))
    resp = requests.post(url, json={"solution": solution}, timeout=120)
    resp.raise_for_status()
    logger.info("Submission response: %s", resp.text)


class SREGymMcpClient:
    """Concrete SREGymMcpGateway backed by SREGym's MCP-over-SSE servers.

    The cerebral runtime only ever drives the ``kubectl`` endpoint, but we map
    the prometheus/jaeger endpoints too so the gateway is faithful to the
    protocol. Clients are connected lazily and kept open for the run.
    """

    def __init__(self, ltc: LanggraphToolConfig, session_id: str, sse_timeout: float | None):
        self._urls = {
            "kubectl": ltc.kubectl_mcp_url,
            "prometheus": ltc.prometheus_mcp_url,
            "jaeger": ltc.jaeger_mcp_url,
        }
        self._session_id = session_id
        self._sse_timeout = sse_timeout
        self._clients: dict[str, Client] = {}

    async def _client(self, endpoint: str) -> Client:
        if endpoint not in self._clients:
            url = self._urls.get(endpoint)
            if not url:
                raise ValueError(f"unknown MCP endpoint: {endpoint!r}")
            transport = SSETransport(
                url=url,
                headers={"sregym_ssid": self._session_id},
                sse_read_timeout=self._sse_timeout,
            )
            client = Client(transport)
            await client.__aenter__()
            self._clients[endpoint] = client
        return self._clients[endpoint]

    async def list_tools(self, endpoint: str) -> list:
        SREGymMcpTool = CEREBRAL["SREGymMcpTool"]
        client = await self._client(endpoint)
        tools = await client.list_tools()
        return [
            SREGymMcpTool(
                endpoint=endpoint,
                name=t.name,
                description=t.description or "",
                args_schema=t.inputSchema or {},
            )
            for t in tools
        ]

    async def call_tool(self, endpoint: str, tool: str, args: dict) -> str:
        client = await self._client(endpoint)
        result = await client.call_tool(tool, arguments=args)
        return "\n".join(getattr(part, "text", "") for part in result)

    async def aclose(self) -> None:
        for client in self._clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass


def build_provider():
    """Pick the model provider for the cerebral agent.

    Wrapped in FallbackProvider(<primary>, StubProvider) so a model outage
    degrades to deterministic coordination rather than crashing the run — the
    graceful-degradation contract the cerebral agent layer is built around.
    """
    AnthropicProvider = CEREBRAL["AnthropicProvider"]
    FallbackProvider = CEREBRAL["FallbackProvider"]
    StubProvider = CEREBRAL["StubProvider"]
    provider_from_env = CEREBRAL["provider_from_env"]

    # Explicit opt-in to cerebral's own brain selection (stub/anthropic/claude/generic).
    if os.environ.get("CEREBRAL_BRAIN"):
        from cerebral_agents.runner import make_provider

        return make_provider()

    model = os.environ.get("AGENT_MODEL_ID") or os.environ.get("CEREBRAL_MODEL") or "claude-sonnet-4-6"
    # Normalize provider-prefixed ids like "anthropic/claude-..." or "anthropic.claude-...".
    bare_model = model.split("/", 1)[-1]
    max_tokens = int(os.environ.get("CEREBRAL_SREGYM_MAX_TOKENS", "1800"))

    if "claude" in bare_model.lower() and os.environ.get("ANTHROPIC_API_KEY"):
        primary = AnthropicProvider(model=bare_model, max_tokens=max_tokens)
    else:
        primary = provider_from_env(model)

    logger.info("Provider: %s (model=%s)", primary.name, model)
    return FallbackProvider(primary, StubProvider())


def _logs_dir() -> Path:
    base = os.environ.get("AGENT_LOGS_DIR") or "./logs/perception_world_model"
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_stage_result(stage: str, problem_id: str, result) -> None:
    """Write the stage answer + observations as a trajectory JSONL for the visualizer."""
    logs_dir = _logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj = logs_dir / "trajectory" / f"{timestamp}_{problem_id}_{stage}_perception_world_model.jsonl"
    traj.parent.mkdir(parents=True, exist_ok=True)
    with open(traj, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "metadata",
            "agent": "perception-world-model",
            "stage": stage,
            "problem_id": problem_id,
            "timestamp": timestamp,
            "answer": result.answer,
        }) + "\n")
        for idx, obs in enumerate(result.observations):
            f.write(json.dumps({"type": "event", "event_index": idx, **asdict(obs)}) + "\n")
    logger.info("Saved %s trajectory to %s", stage, traj)


async def run() -> int:
    ltc = LanggraphToolConfig()
    overall_timeout = float(os.environ.get("CEREBRAL_RUN_TIMEOUT", "1800"))
    sse_timeout: float | None = float(os.environ.get("SSE_READ_TIMEOUT", "3600"))
    if sse_timeout is not None and sse_timeout < 0:
        sse_timeout = None

    provider = build_provider()
    gateway = SREGymMcpClient(ltc, str(uuid.uuid4()), sse_timeout)
    runtime = CEREBRAL["SREGymAgentRuntime"](
        provider=provider,
        mcp=gateway,
        max_steps=int(os.environ.get("CEREBRAL_MAX_STEPS", "40")),
    )

    try:
        wait_for_ready_stage(timeout=300)
        app_info = get_app_info()
        problem = get_problem()
        problem_id = str(problem.get("problem_id") or "unknown")

        diagnosis_text = ""
        completed: set[str] = set()
        deadline = time.time() + overall_timeout

        while time.time() < deadline:
            stage = get_stage()
            if stage in (None, "setup"):
                time.sleep(2)
                continue
            if stage in ("tearing_down", "done"):
                logger.info("Conductor reached stage %s — finishing.", stage)
                break

            if stage == "diagnosis" and "diagnosis" not in completed:
                logger.info("=== DIAGNOSIS ===")
                result = await runtime.diagnose(app_info, problem)
                diagnosis_text = result.answer
                save_stage_result("diagnosis", problem_id, result)
                logger.info("Diagnosis:\n%s", diagnosis_text)
                submit(diagnosis_text)
                completed.add("diagnosis")
            elif stage == "mitigation" and "mitigation" not in completed:
                logger.info("=== MITIGATION ===")
                result = await runtime.mitigate(app_info, problem, diagnosis_text)
                save_stage_result("mitigation", problem_id, result)
                # mitigate() returns "" once it has actuated; the conductor grades
                # the resulting cluster state, so the empty string is the signal.
                submit(result.answer)
                completed.add("mitigation")
                break
            else:
                time.sleep(2)
        else:
            logger.warning("Run timed out after %ss", overall_timeout)

        return 0
    finally:
        await gateway.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cerebral perception-world-model agent on SREGym")
    parser.add_argument("--model", default=None, help="Override AGENT_MODEL_ID")
    args = parser.parse_args()
    if args.model:
        os.environ["AGENT_MODEL_ID"] = args.model

    logger.info("=" * 80)
    logger.info("Starting Cerebral perception-world-model agent for SREGym")
    logger.info("=" * 80)

    global CEREBRAL
    CEREBRAL = _load_cerebral_agents()

    sys.exit(asyncio.run(run()))


CEREBRAL: dict = {}


if __name__ == "__main__":
    main()
