"""
League visualization for Parallel Risk.

Generate plots showing learning progress across multiple opponents.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_multi_opponent_win_rates(
    league_results: Dict,
    output_path: str,
    title: str = "Win Rate vs Multiple Opponents"
):
    """
    Plot win rate curves for each opponent type.

    Args:
        league_results: Nested dict {iteration: {opponent_name: results}}
        output_path: Path to save plot
        title: Plot title
    """
    # Extract opponent names (consistent across iterations)
    iterations = sorted(league_results.keys(), key=int)
    if not iterations:
        print("No evaluation data to plot")
        return

    first_iter = iterations[0]
    opponent_names = list(league_results[str(first_iter)].keys())

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each opponent
    for opponent_name in opponent_names:
        win_rates = []
        for it in iterations:
            results = league_results[str(it)].get(opponent_name, {})
            win_rates.append(results.get("win_rate", 0))

        # Different style for random baseline
        if "random" in opponent_name.lower():
            ax.plot(iterations, win_rates, marker='s', linewidth=3,
                   markersize=10, label=opponent_name, linestyle='--', alpha=0.8)
        else:
            ax.plot(iterations, win_rates, marker='o', linewidth=2,
                   markersize=8, label=opponent_name, alpha=0.8)

    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5,
              label='50% (parity)', alpha=0.5)
    ax.axhline(y=0.7, color='green', linestyle=':', linewidth=1.5,
              label='70% (target)', alpha=0.5)

    ax.set_xlabel('Training Iteration', fontsize=13)
    ax.set_ylabel('Win Rate', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Multi-opponent win rate curve saved to {output_path}")


def plot_win_rate_heatmap(
    league_results: Dict,
    output_path: str,
    title: str = "Win Rate Heatmap (Iteration vs Opponent)"
):
    """
    Plot heatmap showing win rates across iterations and opponents.

    Args:
        league_results: Nested dict {iteration: {opponent_name: results}}
        output_path: Path to save plot
        title: Plot title
    """
    iterations = sorted(league_results.keys(), key=int)
    if not iterations:
        print("No evaluation data to plot")
        return

    # Get opponent names
    first_iter = iterations[0]
    opponent_names = list(league_results[str(first_iter)].keys())

    # Build matrix: rows = iterations, cols = opponents
    matrix = []
    for it in iterations:
        row = []
        for opponent_name in opponent_names:
            results = league_results[str(it)].get(opponent_name, {})
            row.append(results.get("win_rate", 0))
        matrix.append(row)

    matrix = np.array(matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(opponent_names) * 1.2), max(8, len(iterations) * 0.3)))

    sns.heatmap(
        matrix,
        xticklabels=opponent_names,
        yticklabels=iterations,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Win Rate'},
        ax=ax,
        linewidths=0.5
    )

    ax.set_xlabel('Opponent', fontsize=13)
    ax.set_ylabel('Training Iteration', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Win rate heatmap saved to {output_path}")


def plot_aggregate_learning_curve(
    league_results: Dict,
    output_path: str,
    title: str = "Aggregate Learning Curve (All Opponents)"
):
    """
    Plot aggregate learning progress across all opponents.

    Shows mean and std of win rates at each iteration.

    Args:
        league_results: Nested dict {iteration: {opponent_name: results}}
        output_path: Path to save plot
        title: Plot title
    """
    iterations = sorted(league_results.keys(), key=int)
    if not iterations:
        print("No evaluation data to plot")
        return

    # Compute mean and std at each iteration
    mean_win_rates = []
    std_win_rates = []

    for it in iterations:
        win_rates = [
            results.get("win_rate", 0)
            for results in league_results[str(it)].values()
        ]
        mean_win_rates.append(np.mean(win_rates))
        std_win_rates.append(np.std(win_rates))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iterations, mean_win_rates, marker='o', linewidth=3,
           markersize=10, label='Mean win rate', color='blue')
    ax.fill_between(
        iterations,
        np.array(mean_win_rates) - np.array(std_win_rates),
        np.array(mean_win_rates) + np.array(std_win_rates),
        alpha=0.3,
        color='blue',
        label='± 1 std'
    )

    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5,
              label='50% (parity)', alpha=0.7)
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1.5,
              label='70% (target)', alpha=0.7)

    ax.set_xlabel('Training Iteration', fontsize=13)
    ax.set_ylabel('Win Rate', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Aggregate learning curve saved to {output_path}")


def plot_episode_length_by_opponent(
    league_results: Dict,
    output_path: str,
    title: str = "Episode Length Progression by Opponent"
):
    """
    Plot episode length trends for each opponent type.

    Args:
        league_results: Nested dict {iteration: {opponent_name: results}}
        output_path: Path to save plot
        title: Plot title
    """
    iterations = sorted(league_results.keys(), key=int)
    if not iterations:
        print("No evaluation data to plot")
        return

    first_iter = iterations[0]
    opponent_names = list(league_results[str(first_iter)].keys())

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each opponent
    for opponent_name in opponent_names:
        avg_lengths = []
        for it in iterations:
            results = league_results[str(it)].get(opponent_name, {})
            avg_lengths.append(results.get("avg_episode_length", 0))

        # Different style for random baseline
        if "random" in opponent_name.lower():
            ax.plot(iterations, avg_lengths, marker='s', linewidth=3,
                   markersize=10, label=opponent_name, linestyle='--', alpha=0.8)
        else:
            ax.plot(iterations, avg_lengths, marker='o', linewidth=2,
                   markersize=8, label=opponent_name, alpha=0.8)

    ax.set_xlabel('Training Iteration', fontsize=13)
    ax.set_ylabel('Average Episode Length (turns)', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Episode length by opponent curve saved to {output_path}")


def plot_league_dashboard(
    league_results: Dict,
    output_path: str,
    title: str = "Self-Play League Dashboard"
):
    """
    Create a comprehensive dashboard with multiple subplots.

    Args:
        league_results: Nested dict {iteration: {opponent_name: results}}
        output_path: Path to save plot
        title: Plot title
    """
    iterations = sorted(league_results.keys(), key=int)
    if not iterations:
        print("No evaluation data to plot")
        return

    first_iter = iterations[0]
    opponent_names = list(league_results[str(first_iter)].keys())

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: Multi-opponent win rates
    ax1 = fig.add_subplot(gs[0, :])
    for opponent_name in opponent_names:
        win_rates = [league_results[str(it)].get(opponent_name, {}).get("win_rate", 0)
                    for it in iterations]
        if "random" in opponent_name.lower():
            ax1.plot(iterations, win_rates, marker='s', linewidth=2.5,
                    markersize=8, label=opponent_name, linestyle='--', alpha=0.8)
        else:
            ax1.plot(iterations, win_rates, marker='o', linewidth=2,
                    markersize=7, label=opponent_name, alpha=0.8)
    ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(y=0.7, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Win Rate vs All Opponents')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Subplot 2: Aggregate learning curve
    ax2 = fig.add_subplot(gs[1, 0])
    mean_win_rates = [np.mean([results.get("win_rate", 0)
                               for results in league_results[str(it)].values()])
                     for it in iterations]
    std_win_rates = [np.std([results.get("win_rate", 0)
                             for results in league_results[str(it)].values()])
                    for it in iterations]
    ax2.plot(iterations, mean_win_rates, marker='o', linewidth=2.5,
            markersize=8, label='Mean', color='blue')
    ax2.fill_between(iterations,
                     np.array(mean_win_rates) - np.array(std_win_rates),
                     np.array(mean_win_rates) + np.array(std_win_rates),
                     alpha=0.3, color='blue')
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Aggregate Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Subplot 3: Episode lengths
    ax3 = fig.add_subplot(gs[1, 1])
    for opponent_name in opponent_names:
        avg_lengths = [league_results[str(it)].get(opponent_name, {}).get("avg_episode_length", 0)
                      for it in iterations]
        if "random" in opponent_name.lower():
            ax3.plot(iterations, avg_lengths, marker='s', linewidth=2.5,
                    markersize=8, label=opponent_name, linestyle='--', alpha=0.8)
        else:
            ax3.plot(iterations, avg_lengths, marker='o', linewidth=2,
                    markersize=7, label=opponent_name, alpha=0.8)
    ax3.set_xlabel('Training Iteration')
    ax3.set_ylabel('Avg Episode Length (turns)')
    ax3.set_title('Episode Length Trends')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"League dashboard saved to {output_path}")


def plot_league_results(results_json_path: str, output_dir: Optional[str] = None):
    """
    Generate all league plots from results JSON.

    Args:
        results_json_path: Path to league results JSON
        output_dir: Directory to save plots (defaults to same dir as JSON)
    """
    # Load results
    with open(results_json_path, 'r') as f:
        data = json.load(f)

    league_results = data.get("evaluations", {})
    if not league_results:
        print("No evaluation data found in results JSON")
        return

    # Setup output directory
    if output_dir is None:
        output_dir = Path(results_json_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating league plots...")

    # Generate all plots
    plot_multi_opponent_win_rates(
        league_results,
        output_path=str(output_dir / "league_win_rates.png")
    )

    plot_win_rate_heatmap(
        league_results,
        output_path=str(output_dir / "league_heatmap.png")
    )

    plot_aggregate_learning_curve(
        league_results,
        output_path=str(output_dir / "league_aggregate.png")
    )

    plot_episode_length_by_opponent(
        league_results,
        output_path=str(output_dir / "league_episode_lengths.png")
    )

    plot_league_dashboard(
        league_results,
        output_path=str(output_dir / "league_dashboard.png")
    )

    print(f"\nAll league plots saved to {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python league_visualize.py <results_json_path> [output_dir]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    plot_league_results(results_path, output_dir)
