"""Plot mask all-reduce performance results from output.log.

Uses only matplotlib + numpy (no pandas/seaborn required).

Usage:
    python plot.py --plot v0
    python plot.py --plot v0 --base vN

Generates PNG plots under ./plots/:
    by_size_v0.png
    by_sparsity_v0.png
    by_distribution_v0.png
"""

import argparse
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parse log
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path):
    """Parse output.log into a list of row dicts."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean_lines = [_ANSI_RE.sub("", line) for line in lines]

    header_idx = None
    for i, line in enumerate(clean_lines):
        if "size(MB)" in line and "distribution" in line and "version" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find performance table header in {path}")

    header = clean_lines[header_idx].split()
    rows = []
    for line in clean_lines[header_idx + 1 :]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != len(header):
            continue
        row = dict(zip(header, parts))
        rows.append(
            {
                "size_mb": int(row["size(MB)"]),
                "sparsity": float(row["sparsity"].rstrip("%")) / 100.0,
                "distribution": row["distribution"],
                "real_sparsity": float(row["real_sparsity"].rstrip("%")) / 100.0,
                "version": row["version"],
                "no_mask_baseline_ms": float(row["no_mask_baseline(ms)"]),
                "no_mask_ms": float(row["no_mask(ms)"]),
                "mask_ms": float(row["mask(ms)"]),
                "speedup": float(row["speedup"].rstrip("x")),
                "no_mask_vs_baseline": float(row["no_mask_vs_baseline"].rstrip("x")),
                "no_mask_gbs": float(row["no_mask(GB/s)"]),
                "mask_gbs": float(row["mask(GB/s)"]),
                "check": row["check"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def group_agg(rows, group_key, value_key):
    """Return sorted keys, means, and stds for value_key grouped by group_key."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r[value_key])
    keys = sorted(groups.keys())
    means = [np.mean(groups[k]) for k in keys]
    stds = [np.std(groups[k]) for k in keys]
    return keys, means, stds


def group_multi_agg(rows, group_keys, value_keys):
    """Group by tuple of keys and compute mean for each value key."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        gkey = tuple(r[k] for k in group_keys)
        for vk in value_keys:
            groups[gkey][vk].append(r[vk])
    result = {}
    for gkey, vals in groups.items():
        result[gkey] = {vk: np.mean(vs) for vk, vs in vals.items()}
    return result


def make_lookup(rows):
    """Lookup rows by (size_mb, distribution, real_sparsity)."""
    lookup = {}
    for r in rows:
        key = (r["size_mb"], r["distribution"], r["real_sparsity"])
        lookup[key] = r
    return lookup


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

TAB10 = plt.cm.tab10(np.linspace(0, 1, 10))


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _calc_vs_base(df_plot, df_base):
    """Compute vs-base mask speedup for matching (size, dist, sparsity)."""
    if df_base is None:
        return []
    base_lookup = make_lookup(df_base)
    vs_rows = []
    for r in df_plot:
        key = (r["size_mb"], r["distribution"], r["real_sparsity"])
        if key in base_lookup:
            vs_rows.append(
                {
                    "size_mb": r["size_mb"],
                    "distribution": r["distribution"],
                    "real_sparsity": r["real_sparsity"],
                    "vs_base": base_lookup[key]["mask_ms"] / r["mask_ms"],
                }
            )
    return vs_rows


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------
def plot_by_size(df_plot, df_base, version, base_version, outdir):
    """Latency and speedup as a function of tensor size."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    dists = sorted({r["distribution"] for r in df_plot})

    # Top: latency
    ax = axes[0]
    sizes, baseline, _ = group_agg(df_plot, "size_mb", "no_mask_baseline_ms")
    ax.plot(
        sizes, baseline, marker="o", linewidth=2, markersize=5, label="no_mask_baseline"
    )

    for i, dist in enumerate(dists):
        sub = [r for r in df_plot if r["distribution"] == dist]
        xs, means, stds = group_agg(sub, "size_mb", "mask_ms")
        color = TAB10[i % 10]
        ax.plot(xs, means, marker="^", linewidth=2, markersize=5, color=color, label=f"mask ({dist})")
        ax.fill_between(
            xs,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.15,
        )

    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency vs Size — {version}")
    ax.legend(loc="best", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Bottom: speedup
    ax = axes[1]
    for i, dist in enumerate(dists):
        sub = [r for r in df_plot if r["distribution"] == dist]
        xs, means, stds = group_agg(sub, "size_mb", "speedup")
        color = TAB10[i % 10]
        ax.plot(xs, means, marker="o", linewidth=2, markersize=5, color=color, label=dist)
        ax.fill_between(
            xs,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.15,
        )

    vs_rows = _calc_vs_base(df_plot, df_base)
    if vs_rows:
        xs, means, _ = group_agg(vs_rows, "size_mb", "vs_base")
        ax.plot(
            xs,
            means,
            marker="D",
            linestyle="--",
            linewidth=2,
            markersize=5,
            color="black",
            label=f"vs {base_version}",
        )

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Size (MB)")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Size")
    ax.legend(loc="best", fontsize=8)
    ax.set_xscale("log")

    plt.tight_layout()
    return _save(fig, outdir, f"by_size_{version}.png")


def plot_by_sparsity(df_plot, df_base, version, base_version, outdir):
    """Latency and speedup as a function of real sparsity."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    dists = sorted({r["distribution"] for r in df_plot})

    # Top: latency
    ax = axes[0]
    spars, baseline, _ = group_agg(df_plot, "real_sparsity", "no_mask_baseline_ms")
    ax.plot(
        np.array(spars) * 100,
        baseline,
        marker="o",
        linewidth=2,
        markersize=5,
        label="no_mask_baseline",
    )

    for i, dist in enumerate(dists):
        sub = [r for r in df_plot if r["distribution"] == dist]
        xs, means, stds = group_agg(sub, "real_sparsity", "mask_ms")
        color = TAB10[i % 10]
        ax.plot(
            np.array(xs) * 100,
            means,
            marker="^",
            linewidth=2,
            markersize=5,
            color=color,
            label=f"mask ({dist})",
        )
        ax.fill_between(
            np.array(xs) * 100,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.15,
        )

    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency vs Real Sparsity — {version}")
    ax.legend(loc="best", fontsize=8)
    ax.set_yscale("log")

    # Bottom: speedup
    ax = axes[1]
    for i, dist in enumerate(dists):
        sub = [r for r in df_plot if r["distribution"] == dist]
        xs, means, stds = group_agg(sub, "real_sparsity", "speedup")
        color = TAB10[i % 10]
        ax.plot(
            np.array(xs) * 100,
            means,
            marker="o",
            linewidth=2,
            markersize=5,
            color=color,
            label=dist,
        )
        ax.fill_between(
            np.array(xs) * 100,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.15,
        )

    vs_rows = _calc_vs_base(df_plot, df_base)
    if vs_rows:
        xs, means, _ = group_agg(vs_rows, "real_sparsity", "vs_base")
        ax.plot(
            np.array(xs) * 100,
            means,
            marker="D",
            linestyle="--",
            linewidth=2,
            markersize=5,
            color="black",
            label=f"vs {base_version}",
        )

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Real Sparsity (%)")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Real Sparsity")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    return _save(fig, outdir, f"by_sparsity_{version}.png")


def plot_by_distribution(df_plot, df_base, version, base_version, outdir):
    """Latency and speedup grouped by distribution pattern."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    dists = sorted({r["distribution"] for r in df_plot})

    # Top: grouped bars for latency
    ax = axes[0]
    summary = group_multi_agg(
        df_plot, ["distribution"], ["no_mask_baseline_ms", "mask_ms"]
    )
    x = np.arange(len(dists))
    width = 0.35

    ax.bar(
        x - width / 2,
        [summary[(d,)]["no_mask_baseline_ms"] for d in dists],
        width,
        label="no_mask_baseline",
        color=TAB10[0],
    )
    ax.bar(
        x + width / 2,
        [summary[(d,)]["mask_ms"] for d in dists],
        width,
        label="mask",
        color=TAB10[2],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(dists, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency by Distribution — {version}")
    ax.legend(loc="best")
    ax.set_yscale("log")

    # Bottom: speedup bars
    ax = axes[1]
    speedup_summary = group_multi_agg(df_plot, ["distribution"], ["speedup"])
    heights = [speedup_summary[(d,)]["speedup"] for d in dists]
    bars = ax.bar(dists, heights, color="steelblue")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup by Distribution")
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(dists, rotation=15, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # vs base overlay
    vs_rows = _calc_vs_base(df_plot, df_base)
    if vs_rows:
        vs_summary = group_multi_agg(vs_rows, ["distribution"], ["vs_base"])
        vs_heights = [vs_summary[(d,)]["vs_base"] for d in dists]
        ax2 = ax.twinx()
        ax2.plot(
            dists,
            vs_heights,
            marker="D",
            linestyle="--",
            color="black",
            linewidth=2,
            markersize=6,
            label=f"vs {base_version}",
        )
        ax2.set_ylabel(f"Speedup vs {base_version}")
        ax2.legend(loc="upper left")

    plt.tight_layout()
    return _save(fig, outdir, f"by_distribution_{version}.png")


def plot_no_mask_comparison(df_plot, version, outdir):
    """Dedicated figure comparing no_mask_baseline and no_mask (mask=None)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    sizes, baseline, _ = group_agg(df_plot, "size_mb", "no_mask_baseline_ms")
    _, no_mask, _ = group_agg(df_plot, "size_mb", "no_mask_ms")

    # Top: absolute latency
    ax = axes[0]
    ax.plot(
        sizes,
        baseline,
        marker="o",
        linewidth=2,
        markersize=5,
        label="no_mask_baseline",
    )
    ax.plot(
        sizes,
        no_mask,
        marker="s",
        linewidth=2,
        markersize=5,
        label="no_mask",
    )
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"No-Mask Latency Comparison — {version}")
    ax.legend(loc="best")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Bottom: ratio
    ax = axes[1]
    ratio = np.array(no_mask) / np.array(baseline)
    ax.plot(sizes, ratio, marker="o", linewidth=2, markersize=5, color="purple")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Size (MB)")
    ax.set_ylabel("no_mask / no_mask_baseline")
    ax.set_title("No-Mask Ratio to Baseline")
    ax.set_xscale("log")

    plt.tight_layout()
    return _save(fig, outdir, f"no_mask_comparison_{version}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot NCCL mask all-reduce results")
    parser.add_argument("--log", default="output.log", help="Path to output.log")
    parser.add_argument("--plot", required=True, help="Version to plot, e.g. v0")
    parser.add_argument(
        "--base",
        default=None,
        help="Base version for comparison (parsed but not plotted alone)",
    )
    parser.add_argument(
        "--outdir", default="plots", help="Directory to save PNG files"
    )
    args = parser.parse_args()

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"Log file not found: {args.log}")

    rows = parse_log(args.log)

    df_plot = [r for r in rows if r["version"] == args.plot]
    if not df_plot:
        raise ValueError(f"No data for plot version {args.plot}")

    df_base = None
    if args.base:
        df_base = [r for r in rows if r["version"] == args.base]
        if not df_base:
            raise ValueError(f"No data for base version {args.base}")

    os.makedirs(args.outdir, exist_ok=True)

    paths = []
    paths.append(plot_by_size(df_plot, df_base, args.plot, args.base, args.outdir))
    paths.append(
        plot_by_sparsity(df_plot, df_base, args.plot, args.base, args.outdir)
    )
    paths.append(
        plot_by_distribution(df_plot, df_base, args.plot, args.base, args.outdir)
    )
    paths.append(plot_no_mask_comparison(df_plot, args.plot, args.outdir))

    print(f"Generated {len(paths)} plots in {args.outdir}:")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
