"""
llm_helper.py - Python helper for the Positive Influence LLM model.

Handles Ollama API calls, agent memory management, and transcript logging.
Used by positive_influence_llm.nlogo via the NetLogo Python extension.
"""

import os
import json
import random
import re
import urllib.request
import urllib.error

# ── Global state ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(SCRIPT_DIR, "agent_memories")
TRANSCRIPT_PATH = os.path.join(SCRIPT_DIR, "transcript.txt")
PARSE_LOG_PATH = os.path.join(SCRIPT_DIR, "parse_failures.log")

_parse_attempts = 0
_parse_failures = 0

_topic = ""
_model = "phi3:mini"
_memory_length = 5
_num_agents = 25
_ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_backend = "ollama"  # "ollama" or "claude"
_claude_api_key = ""


def _load_env():
    """Load variables from .env file in the script directory."""
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if not os.path.exists(env_path):
        return {}
    env_vars = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                # Strip optional quotes
                value = value.strip().strip("'\"")
                env_vars[key.strip()] = value
    return env_vars


# ── LLM API ──────────────────────────────────────────────────────────────────

def call_llm(prompt, model=None, num_predict=300):
    """Route to the active backend (Ollama or Claude)."""
    if _backend == "claude":
        return call_claude(prompt, model=model, max_tokens=num_predict)
    return call_ollama(prompt, model=model, num_predict=num_predict)


def call_ollama(prompt, model=None, num_predict=300):
    """Send a prompt to the Ollama /api/generate endpoint and return the response text."""
    if model is None:
        model = _model
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "num_predict": num_predict,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{_ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"[llm_helper] Ollama error: {e}")
        return ""


def call_claude(prompt, model=None, max_tokens=300):
    """Send a prompt to the Anthropic Messages API and return the response text."""
    if model is None:
        model = _model
    if not _claude_api_key:
        print("[llm_helper] Claude API key not set — check .env file")
        return ""

    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": _claude_api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            # Extract text from the first content block
            content = body.get("content", [])
            if content and content[0].get("type") == "text":
                return content[0]["text"].strip()
            return ""
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"[llm_helper] Claude API error: {e}")
        if hasattr(e, "read"):
            print(f"[llm_helper] Response: {e.read().decode('utf-8', errors='replace')[:500]}")
        return ""


# ── Memory management ────────────────────────────────────────────────────────

def _memory_path(agent_id):
    return os.path.join(MEMORY_DIR, f"agent_{agent_id}.txt")


def get_agent_memory(agent_id, length=None):
    """Return the last `length` conversation entries from an agent's memory file."""
    if length is None:
        length = _memory_length
    path = _memory_path(agent_id)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    entries = [e.strip() for e in text.split("\n---\n") if e.strip()]
    recent = entries[-length:]
    return "\n---\n".join(recent)


def _append_memory(agent_id, entry):
    """Append a conversation entry to an agent's memory file."""
    path = _memory_path(agent_id)
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry.rstrip() + "\n---\n")


def _get_current_stance(agent_id):
    """Extract the most recent stance line from an agent's memory."""
    path = _memory_path(agent_id)
    if not os.path.exists(path):
        return "no stated opinion yet"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for line in reversed(text.splitlines()):
        if line.startswith("Stance:"):
            return line[len("Stance:"):].strip()
    return "no stated opinion yet"


def _get_current_rationale(agent_id):
    """Extract the most recent rationale line from an agent's memory."""
    path = _memory_path(agent_id)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for line in reversed(text.splitlines()):
        if line.startswith("Rationale:"):
            return line[len("Rationale:"):].strip()
    return ""


# ── Rationale generation ─────────────────────────────────────────────────────

def _generate_rationale(opinion, topic):
    """Prompt the LLM to generate a 1-sentence reason for why an agent holds their opinion."""
    prompt = (
        f'The topic is: "{topic}"\n'
        f"A person's opinion on this is {opinion:.2f} on a scale from -1.0 "
        f"(strongly against) to +1.0 (strongly in favor).\n"
        f"Write ONE specific sentence explaining why they hold this position. "
        f"Be concrete — reference a specific concern, experience, or value. "
        f"Do not be generic."
    )
    response = call_llm(prompt, num_predict=50)
    if response:
        # Take just the first sentence to keep it concise
        first_line = response.strip().split("\n")[0].strip()
        return first_line
    return "No specific reason given."


# ── Opinion extraction from response ─────────────────────────────────────────

def _extract_opinion_from_response(response, fallback):
    """Extract an OPINION: <float> line from an LLM response. Returns the float or fallback."""
    global _parse_attempts, _parse_failures
    _parse_attempts += 1
    # Strip markdown bold markers so **OPINION:** **-0.83** still matches
    cleaned = re.sub(r"\*\*", "", response)
    # Try exact OPINION: <float> pattern first
    match = re.search(r"OPINION:\s*([-+]?\d*\.?\d+)", cleaned)
    if not match:
        # Fallback: find any float on the same line as OPINION
        match = re.search(r"OPINION:.*?([-+]?\d*\.?\d+)", cleaned)
    if not match:
        # Last resort: find any float near "opinion" (case-insensitive) within a few words
        match = re.search(r"(?i)opinion[\s\w]{0,20}([-+]?\d*\.?\d+)", cleaned)
    if match:
        try:
            val = float(match.group(1))
            return max(-1.0, min(1.0, val))
        except ValueError:
            pass
    # Log parse failure
    _parse_failures += 1
    rate = _parse_failures / _parse_attempts * 100
    print(f"[llm_helper] Parse failure ({_parse_failures}/{_parse_attempts}, {rate:.1f}%) — using fallback {fallback:.2f}")
    with open(PARSE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"Failure {_parse_failures}/{_parse_attempts} ({rate:.1f}%) | fallback={fallback:.2f} | response={response[:200]!r}\n")
    # Fallback: return previous opinion with small drift
    return max(-1.0, min(1.0, fallback + random.uniform(-0.1, 0.1)))


# ── Transcript logging ────────────────────────────────────────────────────────

def _log_transcript(tick, agent_a, agent_b, conversation, opinion_a, opinion_b,
                    prior_a=None, prior_b=None):
    """Append a conversation record to the master transcript."""
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(f"=== Tick {tick} | Agent {agent_a} <-> Agent {agent_b} ===\n")
        if prior_a is not None and prior_b is not None:
            f.write(f"PRIOR_STANCE: A({agent_a})={prior_a:.3f}, B({agent_b})={prior_b:.3f}\n")
        f.write(conversation.rstrip() + "\n")
        f.write(f"Opinions after: A({agent_a})={opinion_a:.3f}, B({agent_b})={opinion_b:.3f}\n")
        f.write("\n")


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_agents(num_agents, topic, model_name, memory_length=5, backend="ollama",
                  claude_model="claude-haiku-4-5-20251001"):
    """
    Initialize agent memory files and set random initial stances.
    Returns a list of initial opinion scores (floats in [-1, 1]).
    backend: "ollama" or "claude"
    claude_model: model ID to use when backend is "claude"
    """
    global _topic, _model, _memory_length, _num_agents, _parse_attempts, _parse_failures
    global _backend, _claude_api_key
    global MEMORY_DIR, TRANSCRIPT_PATH, PARSE_LOG_PATH
    _topic = topic
    _model = model_name
    _memory_length = memory_length
    _num_agents = num_agents
    _backend = backend.lower()

    if _backend == "claude":
        env_vars = _load_env()
        _claude_api_key = env_vars.get("ANTHROPIC_API_KEY", "")
        if not _claude_api_key:
            print("[llm_helper] WARNING: ANTHROPIC_API_KEY not found in .env file")
        _model = claude_model
        print(f"[llm_helper] Using Claude model: {_model}")
        print(f"[llm_helper] Backend: Claude API")

    # Create run-specific output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(SCRIPT_DIR, "runs", timestamp)
    MEMORY_DIR = os.path.join(run_dir, "agent_memories")
    TRANSCRIPT_PATH = os.path.join(run_dir, "transcript.txt")
    PARSE_LOG_PATH = os.path.join(run_dir, "parse_failures.log")
    print(f"[llm_helper] Run directory: {run_dir}")

    # Create memory directory
    os.makedirs(MEMORY_DIR, exist_ok=True)

    # Reset parse failure counters
    _parse_attempts = 0
    _parse_failures = 0

    # Initialize transcript and parse failure log
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# Transcript: {topic}\n# Model: {_model}\n# Backend: {_backend}\n\n")
    with open(PARSE_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"# Parse failures: {topic} | {_model}\n")

    # Generate random initial stances and opinions
    initial_opinions = []

    for i in range(num_agents):
        opinion = random.uniform(-1.0, 1.0)
        initial_opinions.append(opinion)

        # Map opinion to a stance description
        if opinion > 0.6:
            stance = f"Strongly in favor of the position on {topic}"
        elif opinion > 0.2:
            stance = f"Somewhat in favor of the position on {topic}"
        elif opinion > -0.2:
            stance = f"Neutral / undecided on {topic}"
        elif opinion > -0.6:
            stance = f"Somewhat against the position on {topic}"
        else:
            stance = f"Strongly against the position on {topic}"

        # Generate a rationale via LLM
        rationale = _generate_rationale(opinion, topic)

        # Write initial memory with stance and rationale
        with open(_memory_path(i), "w", encoding="utf-8") as f:
            f.write(f"Stance: {stance} (opinion score: {opinion:.2f})\n")
            f.write(f"Rationale: {rationale}\n---\n")

        print(f"[llm_helper] Agent {i}/{num_agents} initialized (opinion: {opinion:.2f})")

    return initial_opinions


# ── Conversation ──────────────────────────────────────────────────────────────

def run_conversation(agent_a_id, agent_b_id, tick, memory_length=None):
    """
    Run a 3-turn conversation between two agents via Ollama (one LLM call per turn).
    Returns a dict: {"opinion_a": float, "opinion_b": float, "snippet": str}
    """
    if memory_length is None:
        memory_length = _memory_length

    stance_a = _get_current_stance(agent_a_id)
    stance_b = _get_current_stance(agent_b_id)
    rationale_a = _get_current_rationale(agent_a_id)
    rationale_b = _get_current_rationale(agent_b_id)
    memory_a = get_agent_memory(agent_a_id, memory_length)
    memory_b = get_agent_memory(agent_b_id, memory_length)

    # Extract current numeric opinions for devil's advocate check
    opinion_a_current = _extract_number(stance_a)
    opinion_b_current = _extract_number(stance_b)

    # Devil's advocate instruction when agents are within 0.3 of each other
    devil_advocate = ""
    if abs(opinion_a_current - opinion_b_current) < 0.3:
        devil_advocate = (
            "Challenge the other person's reasoning even if you partly agree. "
            "Play devil's advocate to explore weaknesses in their argument."
        )

    # Build memory context strings
    memory_context_a = ""
    if memory_a:
        memory_context_a = f"\nYour recent conversation history:\n{memory_a}\n"
    memory_context_b = ""
    if memory_b:
        memory_context_b = f"\nYour recent conversation history:\n{memory_b}\n"

    # Build devil's advocate block for prompt injection
    da_block = f"{devil_advocate}\n\n" if devil_advocate else ""

    # ── Turn 1: Agent A opens ──
    turn1_prompt = (
        f'Topic: "{_topic}"\n'
        f"Your position: {stance_a} (score: {opinion_a_current:.2f})\n"
        f"Your reasoning: {rationale_a}\n"
        f"{memory_context_a}"
        f"{da_block}"
        f"State your position on this topic and give ONE specific argument "
        f"supporting it. Be direct — no hedging or seeking common ground. "
        f"Keep it to 2-3 sentences MAX. No headers, bullets, or markdown."
    )
    turn_1 = call_llm(turn1_prompt)
    if not turn_1:
        turn_1 = f"I believe {stance_a}."

    # ── Turn 2: Agent B responds ──
    turn2_prompt = (
        f'Topic: "{_topic}"\n'
        f"Your position: {stance_b} (score: {opinion_b_current:.2f})\n"
        f"Your reasoning: {rationale_b}\n"
        f"{memory_context_b}"
        f"{da_block}"
        f'Someone said: "{turn_1}"\n\n'
        f"Respond to their argument. Defend your own position with a specific "
        f"counterpoint or evidence. Do not simply agree. "
        f"Keep it to 2-3 sentences MAX. No headers, bullets, or markdown.\n\n"
        f"End your response with your opinion score on a new line.\n"
        f"The score must be a number from -1.0 (strongly against) to 1.0 (strongly in favor).\n"
        f"Example: OPINION: 0.35\n"
        f"Example: OPINION: -0.72"
    )
    turn2_raw = call_llm(turn2_prompt)
    if not turn2_raw:
        turn2_raw = f"I disagree. {stance_b}.\nOPINION: {opinion_b_current:.2f}"

    # Parse B's opinion from Turn 2
    opinion_b = _extract_opinion_from_response(turn2_raw, opinion_b_current)
    # Strip the OPINION line from displayed text
    turn_2 = re.sub(r"\n?OPINION:.*", "", turn2_raw).strip()

    # ── Turn 3: Agent A replies ──
    turn3_prompt = (
        f'Topic: "{_topic}"\n'
        f"Your position: {stance_a} (score: {opinion_a_current:.2f})\n"
        f"Your reasoning: {rationale_a}\n"
        f"{da_block}"
        f"Conversation so far:\n"
        f'You said: "{turn_1}"\n'
        f'They replied: "{turn_2}"\n\n'
        f"Respond to their points. You may shift your view if they made a "
        f"compelling argument, or push back if you disagree. Be specific. "
        f"Keep it to 2-3 sentences MAX. No headers, bullets, or markdown.\n\n"
        f"End your response with your opinion score on a new line.\n"
        f"The score must be a number from -1.0 (strongly against) to 1.0 (strongly in favor).\n"
        f"Example: OPINION: 0.35\n"
        f"Example: OPINION: -0.72"
    )
    turn3_raw = call_llm(turn3_prompt)
    if not turn3_raw:
        turn3_raw = f"That's an interesting point, but I maintain my view.\nOPINION: {opinion_a_current:.2f}"

    # Parse A's opinion from Turn 3
    opinion_a = _extract_opinion_from_response(turn3_raw, opinion_a_current)
    # Strip the OPINION line from displayed text
    turn_3 = re.sub(r"\n?OPINION:.*", "", turn3_raw).strip()

    # Build the full conversation text
    conversation = f"A: {turn_1}\nB: {turn_2}\nA: {turn_3}"

    # Build a short snippet for display
    snippet = f"A: {turn_1[:80]}" if turn_1 else "..."

    # Carry forward or update rationale
    rationale_a_updated = rationale_a
    rationale_b_updated = rationale_b

    # Update memories
    entry_a = (
        f"[Tick {tick}] Talked with agent {agent_b_id}:\n"
        f"{conversation}\n"
        f"Stance: Opinion score {opinion_a:.2f}\n"
        f"Rationale: {rationale_a_updated}"
    )
    entry_b = (
        f"[Tick {tick}] Talked with agent {agent_a_id}:\n"
        f"{conversation}\n"
        f"Stance: Opinion score {opinion_b:.2f}\n"
        f"Rationale: {rationale_b_updated}"
    )
    _append_memory(agent_a_id, entry_a)
    _append_memory(agent_b_id, entry_b)

    # Log to transcript
    _log_transcript(tick, agent_a_id, agent_b_id, conversation, opinion_a, opinion_b,
                    prior_a=opinion_a_current, prior_b=opinion_b_current)

    return {"opinion_a": opinion_a, "opinion_b": opinion_b, "snippet": snippet}


def _extract_number(text):
    """Try to extract a float from text like 'opinion score 0.45'."""
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if matches:
        val = float(matches[-1])
        if -1.0 <= val <= 1.0:
            return val
    return 0.0
