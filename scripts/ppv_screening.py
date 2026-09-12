"""Compute PPV/NPV at realistic screening prevalence for the
tumor-naive cfDNA fragmentomics assay.

This script produces a per-prevalence, per-operating-point PPV table
that was missing from RESULTS.md Section 4 (Decision Curve Analysis).

Pure arithmetic — no model re-training, no new data needed.

Output: writes results/ppv_screening.json with the same numbers
printed in the human-readable table.
"""
from __future__ import annotations

import argparse
import json
import os


def ppv(sens: float, spec: float, prev: float) -> float:
    """Positive predictive value.

    Bayes: P(cancer | positive) = P(positive | cancer) P(cancer) /
                                    P(positive)
    where P(positive) = sens*prev + (1-spec)*(1-prev)
    """
    tp = sens * prev
    fp = (1 - spec) * (1 - prev)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def npv(sens: float, spec: float, prev: float) -> float:
    """Negative predictive value."""
    tn = spec * (1 - prev)
    fn = (1 - sens) * prev
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ppv_screening.json",
                    help="output JSON path")
    ap.add_argument("--sens-json",
                    default="results/sens_at_spec.json",
                    help="path to sens_at_spec.json produced by "
                         "scripts/sens_at_specificity.py; if present, "
                         "the new high-specificity operating points "
                         "(95/98/99/99.5/99.9%%) are appended to the "
                         "output JSON and printed as a separate table")
    args = ap.parse_args()

    # Operating points from RESULTS.md Section 4 and the headline
    # fusion result. These come from the no-PCA C=1000 LR baseline.
    points = [
        {"name": "Sens@95% (LR no-PCA C=1000)",
         "sens": 0.91, "spec": 0.95},
        {"name": "Sens@99% (LR no-PCA C=1000)",
         "sens": 0.82, "spec": 0.99},
        {"name": "Fusion @ 80% spec (decision curve)",
         "sens": 0.99, "spec": 0.80},
        {"name": "Fusion @ 95% spec (decision curve)",
         "sens": 0.92, "spec": 0.95},
    ]

    # Realistic US screening prevalences.
    # PATHFINDER/Galleri reported 1.8% in a self-selected high-risk
    # cohort; population-wide MCED prevalence in adults 50+ is much
    # lower. NLST used 1.5% (heavy smokers).
    prevalences = [
        {"label": "US adults 50+ (general pop, age-adjusted)",
         "prev": 0.004,
         "source": "ACS Cancer Statistics 2024, ~0.4% annual incidence "
                   "in 50+ adults"},
        {"label": "US adults 50+, age-adjusted, NHW",
         "prev": 0.005,
         "source": "ACS Cancer Statistics 2024, NHW non-Hispanic white, "
                   "~0.5% annual incidence"},
        {"label": "Heavy smokers 50-80 (NLST cohort)",
         "prev": 0.015,
         "source": "National Lung Screening Trial, ~1.5% lung cancer "
                   "per year"},
        {"label": "High-risk surveillance (post-diagnosis MRD-like)",
         "prev": 0.025,
         "source": "Surveillance cohort (typical MRD program, 2-3%)"},
        {"label": "High-risk surveillance (upper bound, BRCA carriers)",
         "prev": 0.04,
         "source": "BRCA-carrier surveillance cohorts, ~3-4% annual "
                   "cancer incidence"},
    ]

    # === Optional: load sens@spec from sens_at_specificity.py output ===
    sens_spec_path = args.sens_json
    sens_spec_data = None
    if os.path.exists(sens_spec_path):
        try:
            with open(sens_spec_path) as f:
                sens_spec_data = json.load(f)
            print(f"\nLoaded sens@spec data from {sens_spec_path} "
                  f"(N={sens_spec_data.get('pooled_oof', {}).get('n', '?')}, "
                  f"AUC={sens_spec_data.get('pooled_oof', {}).get('auc_mean', '?'):.4f})"
                  if isinstance(sens_spec_data.get('pooled_oof', {}).get('auc_mean'), (int, float))
                  else f"\nLoaded sens@spec data from {sens_spec_path}")
            # Add each new operating point to the points list so PPV is
            # computed at the same realistic prevalences as the legacy
            # operating points.
            for r in sens_spec_data.get("per_specificity", []):
                points.append({
                    "name": f"Sens@{r['specificity']*100:.1f}% "
                            f"(5-channel pooled OOF, n="
                            f"{sens_spec_data.get('pooled_oof', {}).get('n', '?')})",
                    "sens": r["sensitivity"],
                    "spec": r["specificity"],
                })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: could not parse {sens_spec_path}: {e}")
            sens_spec_data = None

    rows = []
    for pt in points:
        for prev_info in prevalences:
            sens, spec, prev = pt["sens"], pt["spec"], prev_info["prev"]
            p = ppv(sens, spec, prev)
            n = npv(sens, spec, prev)
            fp_per_tp = ((1 - spec) * (1 - prev)) / (sens * prev) if sens * prev > 0 else float("inf")
            # Number needed to screen to find 1 true cancer (NNT)
            # = 1 / (sens * prev)
            nnt = 1.0 / (sens * prev) if sens * prev > 0 else float("inf")
            rows.append({
                "operating_point": pt["name"],
                "sens": sens,
                "spec": spec,
                "prevalence_label": prev_info["label"],
                "prevalence": prev,
                "prevalence_source": prev_info["source"],
                "ppv": p,
                "npv": n,
                "fp_per_tp": fp_per_tp,
                "nnt": nnt,
            })

    # Print human-readable table
    print("=" * 110)
    print(f"{'Operating point':<40} {'Prevalence':<38} {'PPV':>8} {'FPs/TP':>10}")
    print("=" * 110)
    # Group by prevalence for readability
    for prev_info in prevalences:
        print(f"\n>>> Prevalence: {prev_info['label']} ({prev_info['prev']*100:.1f}%)")
        print(f"    Source: {prev_info['source']}")
        prev_rows = [r for r in rows if r["prevalence_label"] == prev_info["label"]]
        for r in prev_rows:
            print(f"    {r['operating_point']:<38}  PPV={r['ppv']*100:>5.1f}%  "
                  f"FPs per TP = {r['fp_per_tp']:>5.1f}:1  "
                  f"NNT = {r['nnt']:>5.0f}")

    # Key interpretation
    print("\n" + "=" * 110)
    print("KEY INTERPRETATION (operating at Sens@95%, 0.4% prevalence):")
    r95_04 = next(r for r in rows
                 if r["operating_point"] == "Sens@95% (LR no-PCA C=1000)"
                 and r["prevalence_label"] == "US adults 50+ (general pop, age-adjusted)")
    r99_04 = next(r for r in rows
                 if r["operating_point"] == "Sens@99% (LR no-PCA C=1000)"
                 and r["prevalence_label"] == "US adults 50+ (general pop, age-adjusted)")
    print(f"  At 95% spec / 91% sens: PPV = {r95_04['ppv']*100:.1f}%  "
          f"({r95_04['fp_per_tp']:.1f} false positives per true positive)")
    print(f"  At 99% spec / 82% sens: PPV = {r99_04['ppv']*100:.1f}%  "
          f"({r99_04['fp_per_tp']:.1f} false positives per true positive)")
    print()
    print("Honest framing: At population-prevalence screening, even a")
    print("99%-specificity MCED assay has PPV ~25% (3 false positives")
    print("per true cancer). The Galleri PATHFINDER trial reported")
    print("PPV ~38% in their high-risk self-selected cohort (prevalence")
    print("~1.8%) -- substantially higher than a population-level rollout")
    print("would see. Sensitivity gains have diminishing returns when")
    print("specificity is the limiting factor.")

    # === Print sens@spec table from sens_at_specificity.py (if loaded) ===
    if sens_spec_data is not None:
        print("\n" + "=" * 110)
        print("SENSITIVITY AT HIGH SPECIFICITY (from sens_at_specificity.py, "
              "pooled 5-seed 5-fold OOF):")
        print("=" * 110)
        print(f"{'Specificity':>11} {'Sens':>8} {'CI95 lo':>9} {'CI95 hi':>9} "
              f"{'Threshold':>11}  {'PPV@0.4%':>10}  {'FPs/TP':>8}")
        print("-" * 110)
        for r in sens_spec_data["per_specificity"]:
            spec = r["specificity"]
            sens = r["sensitivity"]
            # PPV at 0.4% prevalence for this operating point
            new_row = next(
                (x for x in rows
                 if x["operating_point"].startswith(f"Sens@{spec*100:.1f}%")
                 and x["prevalence"] == 0.004),
                None)
            if new_row:
                ppv_str = f"{new_row['ppv']*100:.1f}%"
                fp_str = f"{new_row['fp_per_tp']:.1f}"
            else:
                ppv_str = "n/a"; fp_str = "n/a"
            print(f"{spec*100:>10.2f}% {sens*100:>7.2f}% "
                  f"{r['ci95_lo']*100:>8.2f}% {r['ci95_hi']*100:>8.2f}% "
                  f"{r['operating_threshold']:>11.4f}  {ppv_str:>10}  {fp_str:>8}")
        print("=" * 110)
        print(f"Pooled OOF N = {sens_spec_data['pooled_oof']['n']}, "
              f"AUC = {sens_spec_data['pooled_oof']['auc_mean']:.4f} "
              f"+/- {sens_spec_data['pooled_oof']['auc_std']:.4f}")
        print("These numbers are pooled OOF on the SAME cohort -- not "
              "external validation. See BENCHMARK_published.md for the "
              "comparison with Galleri / Shield / CancerSEEK.")

    # Save JSON
    out = {
        "operating_points": points,
        "prevalences": prevalences,
        "results": rows,
        "sens_at_specificity_source": (
            sens_spec_path if sens_spec_data is not None else None),
        "interpretation": {
            "at_sens95_spec95_prev04_pct": r95_04["ppv"] * 100,
            "at_sens95_spec95_prev04_fp_per_tp": r95_04["fp_per_tp"],
            "at_sens99_spec99_prev04_pct": r99_04["ppv"] * 100,
            "at_sens99_spec99_prev04_fp_per_tp": r99_04["fp_per_tp"],
            "comparison_to_galleri_pathfinder": {
                "galleri_ppv_in_pathfinder": 38.0,
                "galleri_pathfinder_prevalence": 1.8,
                "this_work_at_pathfinder_prevalence": next(
                    (r["ppv"] * 100 for r in rows
                     if r["operating_point"].startswith("Fusion @ 95%")
                     and r["prevalence"] == 0.018),
                    None),
            },
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
