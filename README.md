# Opinion Dynamics with LLM Conversations

An agent-based opinion dynamics simulation combining NetLogo's multi-agent framework with local LLM inference via Ollama. Agents hold opinions on a topic, are randomly paired each tick to have LLM-generated conversations, and update their opinions based on those exchanges. Implements Smaldino's positive influence model.

## Prerequisites

- **[NetLogo 6.4+](https://ccl.northwestern.edu/netlogo/download.shtml)** with the Python extension enabled
- **Python 3.7+** (standard library only — no pip dependencies)
- **[Ollama](https://ollama.com)** installed and running locally
- A pulled model (see below)

## Installation

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull phi3:mini
```

Other small models that work well:

```bash
ollama pull tinyllama
ollama pull qwen2:1.5b
ollama pull gemma2:2b
```

### 2. Install NetLogo

Download NetLogo 6.4+ from [ccl.northwestern.edu/netlogo](https://ccl.northwestern.edu/netlogo/download.shtml).

The Python extension (`py`) is bundled with NetLogo — no separate install needed. NetLogo will use whichever `python3` (or `python`) is on your `PATH`.

### 3. Clone this repository

```bash
git clone <repo-url>
cd influence
```

## Running the Simulation

### Step 1 — Start Ollama

```bash
ollama serve
```

Leave this running in a terminal. By default it listens on `http://localhost:11434`. To use a different address, set the environment variable before starting NetLogo:

```bash
export OLLAMA_URL=http://localhost:11434
```

### Step 2 — Open the model in NetLogo

Open `positive_influence_llm.nlogo` in NetLogo (File → Open).

### Step 3 — Configure parameters

In the NetLogo interface, set:

| Parameter | Description | Default |
|---|---|---|
| `discussion-topic` | The issue agents will debate | "whether AI should be heavily regulated..." |
| `ollama-model` | Ollama model name to use | `phi3:mini` |
| `num-agents` | Number of agents (4–100) | `25` |
| `memory-length` | Past conversations included in each prompt | `5` |

Keep `num-agents` low (25 is a good starting point) — each tick requires two sequential LLM calls and can take several seconds.

### Step 4 — Run

1. Click **setup** to initialize agents with random opinions
2. Click **go-once** to run a single conversation tick, or **go** to run continuously

## What You'll See

- Agents displayed on a grid, colored on a **black → white** grayscale based on opinion (`-1` = black/against, `+1` = white/in favor)
- A **histogram** of the current opinion distribution (bottom panel)
- A **Last Conversation** monitor showing the opening line of the most recent exchange
- Opinions tend to converge over time, consistent with the positive influence model

## Output Files

Each run creates (gitignored):

| File | Contents |
|---|---|
| `agent_memories/agent_<id>.txt` | Per-agent conversation history, one entry per tick |
| `transcript.txt` | Master log of every conversation with tick, agent IDs, full dialogue, and final opinions |

These are cleared and recreated on each **setup**.

## Architecture

Two files comprise the system:

**`positive_influence_llm.nlogo`** — NetLogo model. Handles the UI, agent grid, and visualization. Each tick it picks a random pair of agents and delegates the conversation to Python.

**`llm_helper.py`** — Python module loaded by NetLogo's Python extension. Makes all Ollama API calls, manages per-agent memory files, and writes the transcript. No external dependencies.

Each tick makes two Ollama calls:
1. Generate a 3-turn natural language conversation between the two agents
2. Extract updated opinion scores (`{"a": float, "b": float}`) from the conversation

If opinion parsing fails, the prior stance is preserved with a small random perturbation.

## Troubleshooting

**"Cannot find llm_helper"** — NetLogo's working directory must be the repository root (where `llm_helper.py` lives). Use File → Open from within the repo directory, or set NetLogo's working directory accordingly.

**Slow ticks** — Each tick is gated on two LLM round-trips. Use a smaller/faster model (`tinyllama`, `qwen2:1.5b`) or reduce `num-agents`.

**Ollama connection errors** — Make sure `ollama serve` is running and the model has been pulled (`ollama list` to check).

**Python extension errors** — Confirm that `python3` (or `python`) is on your system `PATH` and is version 3.7+. NetLogo's Python extension picks up the system Python automatically.
