# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An agent-based opinion dynamics simulation combining NetLogo's multi-agent framework with local LLM inference via Ollama. Agents hold opinions on a topic, are paired each tick to have LLM-generated conversations, and update their opinions based on those conversations. Implements Smaldino's positive influence model.

## Prerequisites

- **NetLogo 7.0+** with Python extension enabled
- **Python 3.7+** (stdlib only — no pip dependencies)
- **Ollama** running locally: `ollama serve`
- A pulled model: `ollama pull qwen2.5:0.5b`

## Running the Simulation

1. Start Ollama: `ollama serve`
2. Open `positive_influence_llm.nlogox` in NetLogo
3. Set parameters in the UI (topic, model name, agent count, memory length)
4. Click **setup**, then **go** or **go-once**

Environment variable `OLLAMA_URL` overrides the default `http://localhost:11434`.

## Architecture

Three files comprise the system:

### `positive_influence_llm.nlogox`
NetLogo 7 model with three procedures: `setup` (initializes agents and the Python env), `step` (runs one conversation cycle between two randomly selected agents), and `recolor` (maps opinions to grayscale). Each tick, NetLogo selects a pair of agents, delegates to Python, then updates the agents' stored opinion values. Includes two plots: a histogram of current opinion distribution and a scatter plot of agent opinions over time. The `max-ticks` slider auto-stops the simulation after the specified number of ticks (0 = unlimited).

### `llm_helper.py`
Python module loaded by NetLogo's Python extension. Manages all state between ticks via module-level globals and the filesystem.

**Key functions:**
- `setup_agents(num_agents, topic, model_name, memory_length, backend, claude_model)` — creates a timestamped run directory under `runs/`, assigns random initial opinions in [-1.0, 1.0], writes initial stances to per-agent memory files. When backend is `"claude"`, uses the `claude_model` parameter for API calls.
- `run_conversation(agent_a_id, agent_b_id, tick, memory_length)` — makes two sequential Ollama calls: one to generate a 3-turn conversation, one to extract numeric opinion scores; returns `{"opinion_a": float, "opinion_b": float, "snippet": str}`
- `call_ollama(prompt, model, num_predict)` — stateless HTTP POST to Ollama API (temp 0.8, default 300 tokens, 120s timeout)
- `get_agent_memory(agent_id, length)` — reads recent entries from `agent_memories/agent_{id}.txt`

**Runtime files created (under `runs/<timestamp>/`):**
- `agent_memories/agent_{id}.txt` — per-agent conversation history, entries delimited by `---\n`
- `transcript.txt` — master log of all conversations with tick, agent IDs, full dialogue, and final opinions
- `parse_failures.log` — log of opinion-extraction parse failures

### `plot_opinions.py`
Standalone matplotlib script that reconstructs and plots agent opinions over time from a transcript file. Accepts an optional transcript path argument (defaults to `transcript.txt`); derives the `agent_memories/` directory as a sibling of the transcript file. Usage: `python plot_opinions.py runs/<timestamp>/transcript.txt`

**Opinion extraction:** Each agent's turn ends with `OPINION: <float>`. A multi-tier regex extracts the score: (1) exact `OPINION: <float>` match, (2) any float on the same line as `OPINION:`, (3) any float near the word "opinion" (case-insensitive). Markdown bold markers (`**`) are stripped before matching. If all tiers fail, the previous stance is reused with a small random perturbation (±0.1).

## Opinion Scale

Opinions are floats in `[-1.0, 1.0]`. The NetLogo UI shows a real-time histogram of the distribution, a scatter plot of opinions over time, and colors agents on a grayscale based on their current opinion.

## Known Issues & Improvement Notes

### Observed Problems (from transcript analysis, ~2k conversations, updated after 678-tick run)

1. **Opinion-conversation decoupling** — Still the most serious issue. Agents produce identical text but receive opposite scores (e.g., identical arguments yielding -1.0 and 1.0). The 0.5b model cannot reliably map conversational content to a numeric score.
2. **Mode collapse toward extremes** — Middle-range opinions (-0.4 to 0.4) collapsed from 18% to 7.5% over 678 ticks. Agents polarize rather than maintaining diversity. 27% of conversations produce mutual extremization.
3. **Parse failure rate ~16%** — Down from ~35-40% after regex/prompt improvements, but still significant. Remaining failures: model uses "Score:" instead of "OPINION:" (~80 cases), markdown bold wrapping (~52), prose after OPINION: tag (12), no score attempt at all (~77). Each failure injects random noise via the perturbation fallback.
4. **Formulaic repetition** — Late conversations converge on identical bullet points (Job Displacement, Ethical Dilemmas, Sovereignty). Memory accumulates these patterns and agents echo them back.
5. **Weak agent identity** — 104 "as an AI/language model" occurrences across 678 conversations. Rationales are reflected rather than internalized.
6. **Minor: Chinese text bleeding** (12 lines) and empty/truncated turns (~3%) from the Qwen 0.5b model.

### Prompting Improvements — Completed

1. ~~Give agents personas and rationales~~ — done (agent rationales, LLM-generated at setup with 50-token cap)
2. ~~Separate turn generation~~ — done (per-turn generation)
3. ~~Increase `num_predict`~~ — done (300 tokens default, configurable per call)
4. ~~Restructure opinion extraction~~ — done (inline OPINION: extraction with few-shot examples in prompt)
5. ~~Devil's advocate instruction~~ — done (triggers when agents are within 0.3 of each other)
6. ~~Improve parse robustness~~ — done (multi-tier regex, markdown stripping, few-shot examples)

### Improvements (TODO)

1. **Add "Score:" as accepted keyword** — would recover ~80 of 212 remaining parse failures immediately.
2. **Bounded memory decay** — summarize older memories instead of keeping verbatim entries. Would reduce formulaic echo-chamber repetition.
3. **Consider larger model (1.5b/3b)** — the 0.5b model is the root cause of both format non-compliance and opinion-conversation decoupling. Even 1.5b would likely improve structured output reliability and content diversity significantly.

### Architecture Improvements — Completed

1. ~~Track and log parse failure rate~~ — done (parse_failures.log, console logging with running rate)
2. ~~Configurable `num_predict` per call~~ — done (rationale generation uses 50 tokens, conversations use 300)
