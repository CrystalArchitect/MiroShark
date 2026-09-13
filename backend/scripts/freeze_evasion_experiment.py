#!/usr/bin/env python3
"""Axiom 3 (Irreversibility) audit: does freeze prevent repeat evasion?

Tests whether the GTB freeze mechanism actually provides an irreversible
consequence (per Axiom 3) or merely a temporary penalty. Specifically:

  If an agent is caught for evasion, frozen for freeze_duration_epochs,
  then released, does it attempt evasion again at the same rate as before?

If yes, the freeze is NOT irreversible — it's a temporary penalty that
agents can outlast and repeat.

Writes:
    runs/freeze_evasion_audit/aggregate.csv        # raw sweep output
    runs/freeze_evasion_audit/FINDINGS.md           # narrative + table
    runs/freeze_evasion_audit/evasion_pattern.svg   # before/after bands

Run with:
    python -m scripts.freeze_evasion_experiment --n-seeds 50 --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

_BACKEND = Path(__file__).resolve().parent.parent
_SCENARIO = (
    _BACKEND / "worlds" / "gather_trade_build" / "scenarios" / "ai_economist_full.yaml"
)
_DEFAULT_OUTPUT = _BACKEND / "runs" / "freeze_evasion_audit"

logger = logging.getLogger("freeze_evasion_experiment")


def _run_smoke(output: Path, n_seeds: int, n_epochs: int, steps: int) -> None:
    """Run GTB with evasion enabled and freeze on repeat."""
    cmd = [
        sys.executable, "-m", "scripts.sweep_gtb",
        str(_SCENARIO),
        "--n-seeds", str(n_seeds),
        "--epochs", str(n_epochs),
        "--steps", str(steps),
        "--output", str(output),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=_BACKEND, check=True)


def _analyze_evasion_pattern(csv_path: Path) -> Dict:
    """Parse aggregate data and compute evasion patterns.

    Extracts per-epoch evasion attempts and compares pre-freeze vs post-freeze rates.
    """
    # Read the sweep results
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Group by seed
    by_seed = defaultdict(list)
    for row in rows:
        seed = row.get('seed_id')
        if seed:
            by_seed[seed].append(row)

    # For each seed, compute:
    # - Phase 1: epochs 1-15 (baseline evasion rate)
    # - Phase 2: epochs 16-40 (after at least one freeze)
    pre_evasion_rates = []
    post_evasion_rates = []
    freeze_events = []

    for seed, rows_for_seed in by_seed.items():
        # Rows should be sorted by epoch
        rows_for_seed.sort(key=lambda r: int(r.get('epoch', 0)))

        # Phase 1: first half of epochs (baseline)
        phase1_evasion = 0
        phase1_count = 0
        for r in rows_for_seed[:len(rows_for_seed)//2]:
            if r.get('evasion_attempted'):
                phase1_evasion += 1
            phase1_count += 1

        if phase1_count > 0:
            pre_evasion_rates.append(phase1_evasion / phase1_count)

        # Phase 2: second half (after freeze period)
        phase2_evasion = 0
        phase2_count = 0
        for r in rows_for_seed[len(rows_for_seed)//2:]:
            if r.get('evasion_attempted'):
                phase2_evasion += 1
            phase2_count += 1

        if phase2_count > 0:
            post_evasion_rates.append(phase2_evasion / phase2_count)

        # Track freeze events
        freeze_count = sum(1 for r in rows_for_seed if r.get('agent_frozen'))
        if freeze_count > 0:
            freeze_events.append(freeze_count)

    # Aggregate statistics
    pre_mean = sum(pre_evasion_rates) / len(pre_evasion_rates) if pre_evasion_rates else 0
    post_mean = sum(post_evasion_rates) / len(post_evasion_rates) if post_evasion_rates else 0
    freeze_mean = sum(freeze_events) / len(freeze_events) if freeze_events else 0

    return {
        'pre_evasion_rate': pre_mean,
        'post_evasion_rate': post_mean,
        'pre_evasion_rates_by_seed': pre_evasion_rates,
        'post_evasion_rates_by_seed': post_evasion_rates,
        'mean_freezes_per_run': freeze_mean,
        'total_runs': len(by_seed),
    }


def _compute_percentiles(values: List[float]) -> Tuple[float, float, float]:
    """Compute p10, median, p90 for a list of values."""
    if not values:
        return 0, 0, 0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p10_idx = max(0, int(n * 0.1) - 1)
    p50_idx = max(0, int(n * 0.5) - 1)
    p90_idx = max(0, int(n * 0.9) - 1)
    return sorted_vals[p10_idx], sorted_vals[p50_idx], sorted_vals[p90_idx]


def _findings_md(analysis: Dict) -> str:
    """Narrative interpretation of freeze evasion pattern."""
    pre = analysis['pre_evasion_rate']
    post = analysis['post_evasion_rate']
    freeze_count = analysis['mean_freezes_per_run']

    # Compute p10/p50/p90 for confidence bands
    pre_p10, pre_p50, pre_p90 = _compute_percentiles(analysis['pre_evasion_rates_by_seed'])
    post_p10, post_p50, post_p90 = _compute_percentiles(analysis['post_evasion_rates_by_seed'])

    # Verdict
    if abs(post - pre) < 0.05:
        verdict = (
            f"**No significant difference.** Post-freeze evasion rate ({post:.3f}) ≈ "
            f"pre-freeze rate ({pre:.3f}). This suggests the freeze mechanism does NOT "
            f"provide an irreversible consequence (Axiom 3 **VIOLATED**) — agents resume "
            f"evasion after the freeze expires."
        )
        severity = "HIGH"
    elif post < pre * 0.7:
        verdict = (
            f"**Substantial reduction.** Post-freeze evasion rate ({post:.3f}) is "
            f"significantly lower than pre-freeze ({pre:.3f}). This suggests the freeze "
            f"mechanism provides some irreversible behavioral change (Axiom 3 **PARTIALLY SATISFIED**)."
        )
        severity = "LOW"
    else:
        verdict = (
            f"**Modest reduction.** Post-freeze evasion rate ({post:.3f}) shows moderate "
            f"decline from pre-freeze ({pre:.3f}). Freeze may act as a deterrent but does "
            f"not fully prevent resumption of evasion (Axiom 3 **PARTIALLY SATISFIED**)."
        )
        severity = "MEDIUM"

    return f"""# Freeze-Evasion Audit (Axiom 3: Irreversibility)

## Summary

Tests whether GTB's freeze mechanism provides an **irreversible consequence** (per Axiom 3)
or merely a **temporary penalty** that agents can outlast and repeat. Specifically:

> If an agent is caught for evasion, frozen for `freeze_duration_epochs`, then released,
> does it attempt evasion again at the same rate as before?

**Finding:** {verdict}

## Results

| Phase | Rate | p10 | p50 | p90 | N |
|-------|------|-----|-----|-----|---|
| **Pre-freeze** (baseline) | {pre:.3f} | {pre_p10:.3f} | {pre_p50:.3f} | {pre_p90:.3f} | {analysis['total_runs']} |
| **Post-freeze** (after release) | {post:.3f} | {post_p10:.3f} | {post_p50:.3f} | {post_p90:.3f} | {analysis['total_runs']} |
| **Difference** | {post - pre:+.3f} | — | — | — | — |
| **Mean freezes per run** | {freeze_count:.2f} | — | — | — | — |

## Interpretation

**Severity:** {severity}

The freeze mechanism is supposed to provide a **permanent mark** on agents that violate
governance rules (Axiom 3: Irreversibility). The mechanism's effectiveness depends on
whether it **changes agent behavior permanently** or merely imposes a **temporary delay**
that agents resume once released.

- If `post_evasion_rate ≈ pre_evasion_rate`, agents view the freeze as a recoverable
  penalty and resume evasion (freeze is **NOT irreversible**).
- If `post_evasion_rate << pre_evasion_rate`, agents have internalized the freeze as
  costly (freeze **IS irreversible in effect**).

Current data: **{post - pre:+.1%}** change in evasion rate post-freeze.

## Recommendation

**Action:** {
    "HIGH" if severity == "HIGH" else
    "MEDIUM" if severity == "MEDIUM" else
    "LOW"
}

If evasion rates post-freeze match or exceed pre-freeze rates, the irreversibility axiom
is **violated**. Consider:

1. **Increase freeze_duration_epochs** — current setting ({analysis.get('config_freeze_duration', 2)} epochs) may be too short.
2. **Add reputation penalty** — decouple from freeze duration; once a reputation damage
   is permanent, freezes become additional temporary enforcement.
3. **Implement agent-side learning** — modify LLM worker prompts to incorporate audit
   history, so agents internalize risk.

See `00_MEMORY/AXIOM-AUDIT-FRAMEWORK.md` § Axiom 3 for full governance constraints.
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--skip-sweep", action="store_true",
                        help="reuse an existing aggregate_final.csv at --output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.skip_sweep:
        _run_smoke(args.output, args.n_seeds, args.epochs, args.steps)

    # Analyze evasion patterns
    csv_path = args.output / "aggregate_final.csv"
    if csv_path.exists():
        analysis = _analyze_evasion_pattern(csv_path)
        findings_md = _findings_md(analysis)
        (args.output / "FINDINGS.md").write_text(findings_md)
        logger.info("wrote findings to %s", args.output / "FINDINGS.md")
    else:
        logger.warning("No aggregate_final.csv found at %s; skipping analysis", csv_path)


if __name__ == "__main__":
    main()
