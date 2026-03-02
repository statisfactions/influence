"""Plot a histogram of final agent opinions from a transcript file.

Reads initial opinions from agent memory files, replays the transcript
to get final opinions, and plots the distribution.
"""

import argparse
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def get_final_opinions(transcript_path, initial_opinions):
    """Replay transcript and return final opinion for each agent."""
    current = dict(initial_opinions)
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(
                r"Opinions after: A\((\d+)\)=([-\d.]+), B\((\d+)\)=([-\d.]+)", line
            )
            if m:
                current[int(m.group(1))] = float(m.group(2))
                current[int(m.group(3))] = float(m.group(4))
    return current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot histogram of final agent opinions from a transcript."
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        default="transcript.txt",
        help="Path to transcript.txt (default: transcript.txt)",
    )
    args = parser.parse_args()

    transcript_path = args.transcript
    memory_dir = os.path.join(os.path.dirname(transcript_path), "agent_memories")

    initial = get_initial_opinions(memory_dir)
    print(f"Found {len(initial)} agents")

    final = get_final_opinions(transcript_path, initial)

    opinions = list(final.values())
    run_dir = os.path.dirname(transcript_path) or "."

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(opinions, bins=20, range=(-1.0, 1.0), color="steelblue", edgecolor="white")
    ax.set_xlabel("Opinion")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Final Opinion Distribution")
    ax.set_xlim(-1.1, 1.1)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()

    out_path = os.path.join(run_dir, "final_histogram.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    # plt.show()
