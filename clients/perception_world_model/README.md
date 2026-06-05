# Perception World Model (Cerebral)

This client runs the **Cerebral** `perception-world-model` agent against SREGym.

The reasoning agent itself is **not** implemented here — it lives in a sibling
repository, the `cerebral_agents` package at
`../cerebral/perception-world-model/agents`. That package ships an SREGym adapter
(`cerebral_agents.sregym_agent_runtime.SREGymAgentRuntime`) but no
conductor-facing entry point. This directory is that entry point: a thin **bridge
driver** that wires the cerebral runtime to SREGym's conductor API and MCP
servers (the analog of `clients/claudecode/driver.py`).

## How it works

`driver.py`:

1. Locates the sibling `cerebral_agents` package and puts it on `sys.path`
   (auto-discovered relative to the SREGym root; override with
   `CEREBRAL_AGENTS_PATH`).
2. Implements `SREGymMcpClient`, a concrete `SREGymMcpGateway` over SREGym's
   MCP-over-SSE servers (`kubectl` / `prometheus` / `jaeger`).
3. Selects a model provider:
   - `AnthropicProvider` when the model is a Claude id and `ANTHROPIC_API_KEY` is set;
   - otherwise cerebral's OpenAI-compatible `provider_from_env` (vLLM/Ollama/OpenAI/DeepSeek/…);
   - `CEREBRAL_BRAIN` forces cerebral's own selection (`stub|anthropic|claude|generic`).
   The provider is wrapped in `FallbackProvider(<primary>, StubProvider())` so a
   model outage degrades to deterministic coordination instead of crashing.
4. Drives the conductor loop: waits for a ready stage, fetches `app_info` /
   `problem`, runs `diagnose()` → `POST /submit`, then `mitigate()` → `POST
   /submit` (the empty mitigation answer is the actuation signal). A trajectory
   JSONL is written per stage under `AGENT_LOGS_DIR`.

## Running

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY / DEEPSEEK_API_KEY for the generic path
python main.py --agent perception-world-model --model claude-sonnet-4-6
```

## Container isolation

This agent is registered with **`container_isolation: false`** in `agents.yaml`.
The `cerebral_agents` source lives outside the SREGym repo, so it is not baked
into the agent container image (and an install-script `pip install` can't reach
it). The driver therefore runs on the host, where both the sibling repo and the
SREGym virtualenv are available. To run it container-isolated instead, vendor or
publish the `cerebral_agents` package so the image can install it.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_MODEL_ID` / `CEREBRAL_MODEL` | `claude-sonnet-4-6` | Model id (set by `--model`) |
| `CEREBRAL_BRAIN` | _unset_ | Force cerebral's provider selection (`stub｜anthropic｜claude｜generic`) |
| `CEREBRAL_AGENTS_PATH` | `../cerebral/perception-world-model/agents` | Path to the `cerebral_agents` package |
| `CEREBRAL_MAX_STEPS` | `40` | Max agent steps per stage |
| `CEREBRAL_SREGYM_MAX_TOKENS` | `1800` | Model max tokens |
| `CEREBRAL_RUN_TIMEOUT` | `1800` | Overall run timeout (seconds) |
| `SSE_READ_TIMEOUT` | `3600` | MCP SSE read timeout (seconds; `<0` = none) |
| `API_HOSTNAME` / `API_PORT` | `localhost` / `8000` | Conductor REST API |
| `MCP_SERVER_PORT` | `9954` | SREGym MCP server port |
| `AGENT_LOGS_DIR` | `./logs/perception_world_model` | Trajectory / results output |
