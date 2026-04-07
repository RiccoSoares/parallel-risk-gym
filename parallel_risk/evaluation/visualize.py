"""
Visualization utilities for Parallel Risk training and evaluation results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_win_rate_curve(
    results_json_path: str,
    output_path: Optional[str] = None,
    title: str = "Win Rate vs. Training Iteration"
):
    """
    Plot win rate progression over training iterations.

    Args:
        results_json_path: Path to evaluation results JSON
        output_path: Path to save plot (defaults to same dir as JSON)
        title: Plot title
    """
    # Load results
    with open(results_json_path, 'r') as f:
        results = json.load(f)

    # Extract data
    # Results should be a dict mapping iteration -> eval results
    if isinstance(results, dict) and "evaluations" in results:
        evaluations = results["evaluations"]
    else:
        # Single evaluation result
        evaluations = {0: results}

    iterations = sorted(evaluations.keys(), key=int)
    win_rates = [evaluations[str(it)]["win_rate"] for it in iterations]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iterations, win_rates, marker='o', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random baseline (50%)')
    ax.axhline(y=0.7, color='g', linestyle='--', label='Target (70%)')

    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Win Rate vs. Random', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set y-axis limits
    ax.set_ylim([0, 1])

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = Path(results_json_path).parent / f"{Path(results_json_path).stem}_win_rate.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Win rate curve saved to {output_path}")
    return output_path


def plot_episode_length_curve(
    results_json_path: str,
    output_path: Optional[str] = None,
    title: str = "Episode Length vs. Training Iteration"
):
    """
    Plot average episode length progression over training.

    Args:
        results_json_path: Path to evaluation results JSON
        output_path: Path to save plot (defaults to same dir as JSON)
        title: Plot title
    """
    # Load results
    with open(results_json_path, 'r') as f:
        results = json.load(f)

    # Extract data
    if isinstance(results, dict) and "evaluations" in results:
        evaluations = results["evaluations"]
    else:
        evaluations = {0: results}

    iterations = sorted(evaluations.keys(), key=int)
    avg_lengths = [evaluations[str(it)]["avg_episode_length"] for it in iterations]
    std_lengths = [evaluations[str(it)].get("std_episode_length", 0) for it in iterations]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iterations, avg_lengths, marker='o', linewidth=2, markersize=8)
    ax.fill_between(
        iterations,
        np.array(avg_lengths) - np.array(std_lengths),
        np.array(avg_lengths) + np.array(std_lengths),
        alpha=0.3
    )

    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Average Episode Length (turns)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = Path(results_json_path).parent / f"{Path(results_json_path).stem}_episode_length.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Episode length curve saved to {output_path}")
    return output_path


def plot_reward_distribution(
    results_json_path: str,
    iteration: Optional[int] = None,
    output_path: Optional[str] = None,
    title: str = "Episode Reward Distribution"
):
    """
    Plot histogram of episode rewards.

    Args:
        results_json_path: Path to evaluation results JSON
        iteration: Specific iteration to plot (None = latest)
        output_path: Path to save plot (defaults to same dir as JSON)
        title: Plot title
    """
    # Load results
    with open(results_json_path, 'r') as f:
        results = json.load(f)

    # Extract data
    if isinstance(results, dict) and "evaluations" in results:
        evaluations = results["evaluations"]
        if iteration is None:
            # Use latest iteration
            iteration = max(map(int, evaluations.keys()))
        eval_data = evaluations[str(iteration)]
    else:
        eval_data = results
        iteration = 0

    # Get episode lengths as proxy for rewards (wins = long episodes)
    episode_lengths = eval_data["episode_lengths"]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(episode_lengths, bins=20, edgecolor='black', alpha=0.7)

    ax.set_xlabel('Episode Length (turns)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f"{title} (Iteration {iteration})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    ax.text(
        0.98, 0.98,
        f"Mean: {mean_length:.1f}\nStd: {std_length:.1f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = Path(results_json_path).parent / f"{Path(results_json_path).stem}_reward_dist.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Reward distribution plot saved to {output_path}")
    return output_path


def plot_all(results_json_path: str, output_dir: Optional[str] = None):
    """
    Generate all plots from evaluation results.

    Args:
        results_json_path: Path to evaluation results JSON
        output_dir: Directory to save plots (defaults to same dir as JSON)
    """
    if output_dir is None:
        output_dir = Path(results_json_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")

    # Generate all plots
    plot_win_rate_curve(
        results_json_path,
        output_path=output_dir / "win_rate_curve.png"
    )

    plot_episode_length_curve(
        results_json_path,
        output_path=output_dir / "episode_length_curve.png"
    )

    plot_reward_distribution(
        results_json_path,
        output_path=output_dir / "reward_distribution.png"
    )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize.py <results_json_path> [output_dir]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    plot_all(results_path, output_dir)
