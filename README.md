# Opinion Dynamics with LLM Conversations

An agent-based opinion dynamics simulation combining NetLogo's multi-agent framework with LLM inference. Agents hold opinions on a topic, are randomly paired each tick to have LLM-generated conversations, and update their opinions based on those exchanges. Implements Smaldino's positive influence model.

Supports two LLM backends:
- **Ollama** — local inference via any Ollama-compatible model (default)
- **Claude API** — Anthropic's Claude models (Haiku by default)

## Prerequisites

- **[NetLogo 7.0+](https://ccl.northwestern.edu/netlogo/download.shtml)** with the Python extension enabled
- **Python 3.7+** (standard library only — no pip dependencies)
- **One of the following LLM backends:**
  - **[Ollama](https://ollama.com)** installed and running locally, with a pulled model (see below)
  - **[Anthropic API key](https://console.anthropic.com/)** for the Claude backend

## Installation

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull qwen2.5:0.5b
```

Other small models that work well:

```bash
ollama pull qwen2.5:1.5b
ollama pull phi3:mini
ollama pull tinyllama
ollama pull gemma2:2b
```

### 2. Set up Claude API (optional)

To use the Claude backend instead of Ollama, create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

The `.env` file should contain:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Install NetLogo

Download NetLogo 7.0+ from [ccl.northwestern.edu/netlogo](https://ccl.northwestern.edu/netlogo/download.shtml).

The Python extension (`py`) is bundled with NetLogo — no separate install needed. NetLogo will use whichever `python3` (or `python`) is on your `PATH`.

### 4. Clone this repository

```bash
git clone <repo-url>
cd influence
```

## Running the Simulation

### Step 1 — Start your backend

**Ollama (default):**

```bash
ollama serve
```

Leave this running in a terminal. By default it listens on `http://localhost:11434`. To use a different address, set the environment variable before starting NetLogo:

```bash
export OLLAMA_URL=http://localhost:11434
```

**Claude API:** No server needed — just ensure your `.env` file contains a valid `ANTHROPIC_API_KEY` (see installation step 2).

### Step 2 — Open the model in NetLogo

Open `positive_influence_llm.nlogox` in NetLogo (File → Open).

### Step 3 — Configure parameters

In the NetLogo interface, set:

| Parameter | Description | Default |
|---|---|---|
| `discussion-topic` | The issue agents will debate | (see UI) |
| `llm-backend` | LLM backend: `ollama` or `claude` | `ollama` |
| `ollama-model` | Ollama model name (used when backend is `ollama`) | `qwen2.5:0.5b` |
| `claude-model` | Claude model ID (used when backend is `claude`) | `claude-haiku-4-5-20251001` |
| `num-agents` | Number of agents (4–100) | `9` |
| `memory-length` | Past conversations included in each prompt | `5` |
| `max-ticks` | Auto-stop after this many ticks (0 = unlimited) | `500` |

Keep `num-agents` low (25 is a good starting point) — each tick requires two sequential LLM calls and can take several seconds.

### Step 4 — Run

1. Click **setup** to initialize agents with random opinions
2. Click **go-once** to run a single conversation tick, or **go** to run continuously

## What You'll See

- Agents displayed on a grid, colored on a **black → white** grayscale based on opinion (`-1` = black/against, `+1` = white/in favor)
- A **histogram** of the current opinion distribution (bottom panel)
- An **Agent opinions over time** scatter plot showing each agent's opinion at every tick, visualizing convergence/divergence patterns
- A **Last Conversation** monitor showing the opening line of the most recent exchange
- Opinions tend to converge over time, consistent with the positive influence model

## Output Files

Each run creates a timestamped directory under `runs/` (e.g. `runs/2026-03-01_143022/`):

| File | Contents |
|---|---|
| `runs/<timestamp>/agent_memories/agent_<id>.txt` | Per-agent conversation history, one entry per tick |
| `runs/<timestamp>/transcript.txt` | Master log of every conversation with tick, agent IDs, full dialogue, and final opinions |
| `runs/<timestamp>/parse_failures.log` | Log of opinion-extraction parse failures |

Previous runs are preserved — each **setup** creates a new directory.

## Architecture

Two files comprise the system:

**`positive_influence_llm.nlogox`** — NetLogo 7 model. Handles the UI, agent grid, and visualization. Each tick it picks a random pair of agents and delegates the conversation to Python.

**`llm_helper.py`** — Python module loaded by NetLogo's Python extension. Makes all LLM API calls (Ollama or Claude), manages per-agent memory files, and writes the transcript. No external dependencies.

**`plot_opinions.py`** — Standalone plotting script. Reads a transcript file and the corresponding agent memory files to reconstruct and plot agent opinions over time. Requires `matplotlib`.

```bash
# Plot from a specific run
python plot_opinions.py runs/2026-03-01_143022/transcript.txt

# Plot from legacy transcript.txt in the current directory (default)
python plot_opinions.py
```

Each tick makes two LLM calls:
1. Generate a 3-turn natural language conversation between the two agents
2. Extract updated opinion scores (`{"a": float, "b": float}`) from the conversation

If opinion parsing fails, the prior stance is preserved with a small random perturbation.

## Troubleshooting

**"Cannot find llm_helper"** — NetLogo's working directory must be the repository root (where `llm_helper.py` lives). Use File → Open from within the repo directory, or set NetLogo's working directory accordingly.

**Slow ticks** — Each tick is gated on two LLM round-trips. Use a smaller/faster model (`qwen2.5:0.5b`, `tinyllama`) or reduce `num-agents`.

**Ollama connection errors** — Make sure `ollama serve` is running and the model has been pulled (`ollama list` to check).

**Python extension errors** — Confirm that `python3` (or `python`) is on your system `PATH` and is version 3.7+. NetLogo's Python extension picks up the system Python automatically.
