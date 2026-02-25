"""
llm_helper.py - Python helper for the Positive Influence LLM model.

Handles Ollama API calls, agent memory management, and transcript logging.
Used by positive_influence_llm.nlogo via the NetLogo Python extension.
"""

import os
import json
import random
import urllib.request
import urllib.error

# ── Global state ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(SCRIPT_DIR, "agent_memories")
TRANSCRIPT_PATH = os.path.join(SCRIPT_DIR, "transcript.txt")

_topic = ""
_model = "phi3:mini"
_memory_length = 5
_num_agents = 25
_ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")


# ── Ollama API ────────────────────────────────────────────────────────────────

def call_ollama(prompt, model=None):
    """Send a prompt to the Ollama /api/generate endpoint and return the response text."""
    if model is None:
        model = _model
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "num_predict": 300,
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
    # Each entry is separated by a blank line
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
    # Look for the last "Stance:" line
    for line in reversed(text.splitlines()):
        if line.startswith("Stance:"):
            return line[len("Stance:"):].strip()
    return "no stated opinion yet"


# ── Transcript logging ────────────────────────────────────────────────────────

def _log_transcript(tick, agent_a, agent_b, conversation, opinion_a, opinion_b):
    """Append a conversation record to the master transcript."""
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(f"=== Tick {tick} | Agent {agent_a} <-> Agent {agent_b} ===\n")
        f.write(conversation.rstrip() + "\n")
        f.write(f"Opinions after: A({agent_a})={opinion_a:.3f}, B({agent_b})={opinion_b:.3f}\n")
        f.write("\n")


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_agents(num_agents, topic, model_name, memory_length=5):
    """
    Initialize agent memory files and set random initial stances.
    Returns a list of initial opinion scores (floats in [-1, 1]).
    """
    global _topic, _model, _memory_length, _num_agents
    _topic = topic
    _model = model_name
    _memory_length = memory_length
    _num_agents = num_agents

    # Create/clear memory directory
    os.makedirs(MEMORY_DIR, exist_ok=True)
    for f in os.listdir(MEMORY_DIR):
        fp = os.path.join(MEMORY_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    # Clear transcript
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# Transcript: {topic}\n# Model: {model_name}\n\n")

    # Generate random initial stances and opinions
    initial_opinions = []
    stance_templates = [
        "Strongly in favor",
        "Somewhat in favor",
        "Slightly in favor",
        "Neutral / undecided",
        "Slightly against",
        "Somewhat against",
        "Strongly against",
    ]

    for i in range(num_agents):
        # Random initial opinion
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

        # Write initial memory
        with open(_memory_path(i), "w", encoding="utf-8") as f:
            f.write(f"Stance: {stance} (opinion score: {opinion:.2f})\n---\n")

    return initial_opinions


# ── Conversation ──────────────────────────────────────────────────────────────

def run_conversation(agent_a_id, agent_b_id, tick, memory_length=None):
    """
    Run a 3-turn conversation between two agents via Ollama.
    Returns a dict: {"opinion_a": float, "opinion_b": float, "snippet": str}
    """
    if memory_length is None:
        memory_length = _memory_length

    stance_a = _get_current_stance(agent_a_id)
    stance_b = _get_current_stance(agent_b_id)
    memory_a = get_agent_memory(agent_a_id, memory_length)
    memory_b = get_agent_memory(agent_b_id, memory_length)

    # Build conversation prompt
    memory_context = ""
    if memory_a:
        memory_context += f"\nPerson A's recent conversation history:\n{memory_a}\n"
    if memory_b:
        memory_context += f"\nPerson B's recent conversation history:\n{memory_b}\n"

    conv_prompt = f"""You are simulating a brief conversation between two people about {_topic}.

Person A's current stance: {stance_a}
Person B's current stance: {stance_b}
{memory_context}
Write a brief, natural 3-turn conversation (A, B, A). Each turn should be 1-2 sentences. People should engage genuinely with each other's points and may shift their views slightly based on good arguments.

Format exactly as:
A: ...
B: ...
A: ..."""

    conversation = call_ollama(conv_prompt)

    if not conversation:
        # Fallback if LLM fails
        conversation = f"A: I think about {_topic}...\nB: Interesting point.\nA: Let's discuss more."

    # Extract opinions via a second LLM call
    opinion_prompt = f"""Given this conversation about {_topic}:

{conversation}

Person A's prior stance: {stance_a}
Person B's prior stance: {stance_b}

Rate each person's final opinion on a scale from -1.0 (strongly against) to +1.0 (strongly in favor).
Return ONLY a JSON object with no other text: {{"a": <float>, "b": <float>}}"""

    opinion_response = call_ollama(opinion_prompt)
    opinion_a, opinion_b = _parse_opinions(opinion_response, agent_a_id, agent_b_id)

    # Build a short snippet for display
    lines = conversation.strip().split("\n")
    snippet = lines[0][:80] if lines else "..."

    # Update memories
    entry_a = f"[Tick {tick}] Talked with agent {agent_b_id}:\n{conversation}\nStance: Opinion score {opinion_a:.2f}"
    entry_b = f"[Tick {tick}] Talked with agent {agent_a_id}:\n{conversation}\nStance: Opinion score {opinion_b:.2f}"
    _append_memory(agent_a_id, entry_a)
    _append_memory(agent_b_id, entry_b)

    # Log to transcript
    _log_transcript(tick, agent_a_id, agent_b_id, conversation, opinion_a, opinion_b)

    return {"opinion_a": opinion_a, "opinion_b": opinion_b, "snippet": snippet}


def _parse_opinions(response, agent_a_id, agent_b_id):
    """Parse the JSON opinion response from the LLM. Returns (float, float)."""
    # Try to extract JSON from the response
    try:
        # Find JSON object in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            a = float(data.get("a", 0.0))
            b = float(data.get("b", 0.0))
            # Clamp to [-1, 1]
            a = max(-1.0, min(1.0, a))
            b = max(-1.0, min(1.0, b))
            return a, b
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Fallback: return current opinions with small random drift
    print(f"[llm_helper] Could not parse opinions from: {response[:100]}")
    # Read current stances and add noise
    try:
        stance_a = _get_current_stance(agent_a_id)
        stance_b = _get_current_stance(agent_b_id)
        # Try to extract numbers from stance
        a = _extract_number(stance_a)
        b = _extract_number(stance_b)
    except Exception:
        a, b = 0.0, 0.0
    # Add small random perturbation
    a = max(-1.0, min(1.0, a + random.uniform(-0.1, 0.1)))
    b = max(-1.0, min(1.0, b + random.uniform(-0.1, 0.1)))
    return a, b


def _extract_number(text):
    """Try to extract a float from text like 'opinion score 0.45'."""
    import re
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if matches:
        val = float(matches[-1])
        if -1.0 <= val <= 1.0:
            return val
    return 0.0
