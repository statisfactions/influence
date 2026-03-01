"""Plot agent opinions over time from transcript.txt.

Reconstructs full state at every tick by carrying forward each agent's
last known opinion, so all agents appear on every sampled tick.
"""

import argparse
import os
import re
import matplotlib.pyplot as plt

TARGET_POINTS = 200  # aim for ~this many data points per agent


def get_initial_opinions(memory_dir):
    """Read initial opinion scores from agent memory files."""
    opinions = {}
    for fname in os.listdir(memory_dir):
        m = re.match(r"agent_(\d+)\.txt", fname)
        if not m:
            continue
        agent_id = int(m.group(1))
        with open(os.path.join(memory_dir, fname), "r", encoding="utf-8") as f:
            first_line = f.readline()
        score_match = re.search(r"opinion score:\s*([-\d.]+)", first_line)
        if score_match:
            opinions[agent_id] = float(score_match.group(1))
    return opinions


def parse_transcript(path, initial_opinions):
    """Build full opinion snapshots at every tick.

    Returns (ticks, {agent_id: [opinion_at_tick_0, opinion_at_tick_1, ...]})
    """
    # Collect all updates: list of (tick, agent_id, opinion)
    updates = []
    max_tick = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^=== Tick (\d+) \| Agent (\d+) <-> Agent (\d+) ===", line)
            if m:
                tick = int(m.group(1))
                max_tick = max(max_tick, tick)
                continue
            m = re.match(
                r"Opinions after: A\((\d+)\)=([-\d.]+), B\((\d+)\)=([-\d.]+)", line
            )
            if m:
                updates.append((tick, int(m.group(1)), float(m.group(2))))
                updates.append((tick, int(m.group(3)), float(m.group(4))))

    # Build current state, snapshot at each tick
    current = dict(initial_opinions)
    all_agents = sorted(current.keys())
    tick_list = list(range(max_tick + 1))

    # Pre-index updates by tick
    updates_by_tick = {}
    for t, aid, op in updates:
        if t not in updates_by_tick:
            updates_by_tick[t] = []
        updates_by_tick[t].append((aid, op))

    # Walk through ticks, carry forward state
    series = {aid: [] for aid in all_agents}
    for t in tick_list:
        if t in updates_by_tick:
            for aid, op in updates_by_tick[t]:
                current[aid] = op
        for aid in all_agents:
            series[aid].append(current.get(aid, 0.0))

    return tick_list, series


def plot(tick_list, series, sample_every=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Auto-calculate sampling interval if not provided
    if sample_every is None:
        sample_every = max(1, len(tick_list) // TARGET_POINTS)

    # Sample ticks
    sampled_idx = [i for i, t in enumerate(tick_list) if t % sample_every == 0]
    if not sampled_idx or sampled_idx[-1] != len(tick_list) - 1:
        sampled_idx.append(len(tick_list) - 1)
    sampled_ticks = [tick_list[i] for i in sampled_idx]

    for aid in sorted(series):
        opinions = [series[aid][i] for i in sampled_idx]
        ax.scatter(sampled_ticks, opinions, s=4, alpha=0.35, color="steelblue", edgecolors="none")

    ax.set_xlabel("Tick")
    ax.set_ylabel("Opinion")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Agent Opinions Over Time")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig("opinions_over_time.png", dpi=150)
    print("Saved opinions_over_time.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot agent opinions over time from a transcript file.")
    parser.add_argument("transcript", nargs="?", default="transcript.txt",
                        help="Path to transcript.txt (default: transcript.txt)")
    args = parser.parse_args()

    transcript_path = args.transcript
    # Derive agent_memories dir as sibling of the transcript file
    memory_dir = os.path.join(os.path.dirname(transcript_path), "agent_memories")

    initial = get_initial_opinions(memory_dir)
    print(f"Found {len(initial)} agents with initial opinions")
    tick_list, series = parse_transcript(transcript_path, initial)
    print(f"Reconstructed {len(tick_list)} ticks")
    plot(tick_list, series)
