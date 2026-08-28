# AUDIT REPORT 4 — Performance, Usability, Fresh-Clone Experience

**Date**: 2026-08-28
**Reviewer**: senior ML engineer
**Scope**: 7 questions about runtime profile, reusability, test coverage,
reproducibility on a different machine, disk/memory footprint, CI robustness,
and documentation gaps for fresh-clone usage.
**Source files**: `scripts/*.py`, `test/*.py`, `pyproject.toml`,
`.github/workflows/pipeline-tests.yml`, `README.md`, `AUDIT_REPORT_2.md`,
`AUDIT_REPORT_3.md`.

This audit is intentionally narrower than rounds 1-3 — it focuses on the
*operational* surface of the pipeline (someone cloning it fresh), not on
methodology bugs. Where round 1-3 fixed specific code defects (NaN
leakage, `honest_benchmark.py` --help crash, hardcoded paths), this round
fixes the surface that meets a new user.

---

## 1. Runtime profile

**Measured (from AUDIT_REPORT_2 / this round):**

| Script | Wall-clock | Realistic? |
|---|---|---|
| `nuc_ablation.py` 5-seed | **290s** | Acceptable; 4 configs × 5 seeds × 5 folds |
| `lr_no_pca_vs_pca200.py` 10-seed | **~1 min** | Fine |
| `lr_regularization_sweep.py --skip-l1` 5-seed C=1000 | **83s** | Fine |
| `lr_regularization_sweep.py` (default, all C, all penalties) | **~35 min** | Should default to `--skip-l1` |
| `honest_benchmark.py` Section A only | ~1 min | Fine |
| `honest_benchmark.py` Sections A–E | ~10 min | Documented |
| `model_ablation.py` (RF + GB + 2 LR variants × 3 seeds × 5 folds) | ~5–10 min | GB is the bottleneck |
| `auc_reproducibility_gate.py` | ~30 s | Fine |

**Finding 1: Default of `lr_regularization_sweep.py` is 35 minutes.**

The L1 saga sweep runs at C=[0.01, 0.1, 1.0] × 5 seeds × 5 folds on 60k
features — ~7 minutes per C value. Even after `--skip-l1` was added
(round 2), the **default still includes L1**. Every fresh user who
runs the script without reading `--help` will wait 35 min.

**Severity: Nice-to-have** (the `--skip-l1` flag exists; the doc
in the help string is clear. But the default is footgun-y.)
**Effort: 5 minutes** — change `default` of `--skip-l1` to
`action="store_true"` flipped to a positive `--l1` flag, OR keep
the flag but make the first run of `--help` say something like
"NOTE: default omits L1 (35min); pass --include-l1 to add it".

**Concrete fix** — `scripts/lr_regularization_sweep.py:38-42`:

```python
ap.add_argument("--skip-l1", action="store_true",
                help="Skip the L1 saga sweep. By default it runs "
                     "L1 at C=[0.01, 0.1, 1.0] which takes ~35 "
                     "minutes and produces all-zero coefficients "
                     "(see BENCHMARK.md Appendix E.2). DEFAULT: skip "
                     "(use --include-l1 to re-enable).")
ap.add_argument("--include-l1", action="store_true",
                help="Re-enable the slow L1 saga sweep.")
```

And in the body, change `if not args.skip_l1:` to
`if args.include_l1:`. Also update README.md's "Reproduce" lines
to add `--include-l1` if they want the 35-min version.

**Finding 2: GradientBoosting in `model_ablation.py` is single-threaded.**

The script uses `n_jobs=-1` for RF but `GradientBoostingClassifier` has
no `n_jobs` (sklearn limitation). The GB step dominates the runtime
even though it's the least interesting result. With `n_estimators=100,
max_depth=3, learning_rate=0.05`, GB takes ~3 min on 627×60k.

**Severity: Nice-to-have** (the result is the same regardless of speed).
**Effort: 10 minutes** — drop GB to `n_estimators=50` or accept the 3 min.
Or, better, switch to `HistGradientBoostingClassifier` which supports
`n_jobs`.

**Finding 3: Per-fold PCA in `nuc_ablation.py` runs 4× (one per feature config).**

`_evaluate` is called 4 times with `pca_n=200`. Each call does
5-fold × 5-seed = 25 PCA fits. The 4 configs differ only in 3 or 6
extra columns — they could share a single `StandardScaler+PCA` fit
per fold (the standard scaler/PCA doesn't see the response, so it's
safe to compute once per fold and apply to both `X_5ch` and `X_5ch+
nuc`). For 4 configs × 25 = 100 PCA fits → 25 PCA fits per fold.

**Severity: Nice-to-have** — speedup is ~4× on the PCA portion
(~30s of the 290s total).
**Effort: 30 minutes** — refactor `_evaluate` to take pre-fitted PCA
arrays, or compute the PCA-transformed versions of each config
once per fold rather than once per config per fold. Non-trivial
because the harmonization depends on per-fold arrays and the
PCA-once-per-fold requires returning arrays of shape `(n_train, pca_n)`
and `(n_test, pca_n)` per config.

---

## 2. Pipeline reusability (adding a new feature / new bin width)

**Current extension points:**

- **New motif size (e.g. 5-mer instead of 4-mer):**
  touch `scripts/extract_motifs.py` (line ~30, `k=4` → `k=5`) AND
  `scripts/extract_motifs_frag.py` (same constant). The downstream
  code reads `len(freqs)` from JSON, so the classifier auto-adapts.
  But `motifs.json` files for existing samples have 256 keys;
  re-extraction is required for all samples.
  → **2 files, ~1 line each.**

- **New FSD bin width (e.g. 10bp instead of 5bp):**
  touch `scripts/extract_fsd.py:91` (`bins=range(20, 1001, 5)` →
  `bins=range(20, 1001, 10)`) AND `scripts/nuc_features.py:43`
  (`FSD_BIN_CENTERS = 20 + 5 * np.arange(196)`). But the bin-index
  constants in `nuc_features.py` (e.g. `_SUBNUC_IDX = _range_to_indices(*SUBNUCLEOSOME_RANGE)`
  using `// 5`) hardcode the 5bp stride.
  → **2 files, ~5 lines + nuc_features.py:46 (the `_range_to_indices` helper)**.

- **New nucleosome feature (e.g. 6-mer motif group):**
  Add to `scripts/nuc_features.py` as a new `compute_*_from_fsd`
  function; call from `nuc_ablation.py` `_build_*` functions.
  → **2 files.**

**Finding 4: The 5bp bin stride is hardcoded in 4 places.**

`extract_fsd.py:91` uses `range(20, 1001, 5)` and `nuc_features.py`
uses `// 5` in `_range_to_indices` and `* 5` in `FSD_BIN_CENTERS`.
Any change to bin width requires synchronized edits in both files;
if you forget one, the bin indices silently go wrong.

**Severity: Should-fix** (the next person who touches this will
spend an hour figuring out why their new features look wrong).
**Effort: 30 minutes.**

**Concrete fix** — `scripts/nuc_features.py:42-58`:

```python
# Single source of truth for FSD bin layout. extract_fsd.py must
# use the same values.
FSD_BIN_START = 20      # lowest bp
FSD_BIN_WIDTH = 5       # bp per bin
FSD_BIN_END = 1000      # highest bp (exclusive)
N_BINS = (FSD_BIN_END - FSD_BIN_START) // FSD_BIN_WIDTH  # = 196

def _range_to_indices(low: int, high: int) -> tuple[int, int]:
    return ((low - FSD_BIN_START) // FSD_BIN_WIDTH,
            (high - FSD_BIN_START) // FSD_BIN_WIDTH + 1)
```

And expose these from `scripts/extract_fsd.py` via:

```python
from nuc_features import FSD_BIN_START, FSD_BIN_END, FSD_BIN_WIDTH
...
hist, edges = np.histogram(lengths,
                           bins=range(FSD_BIN_START, FSD_BIN_END + 1,
                                      FSD_BIN_WIDTH))
```

(Plus a sanity check at import time that
`FSD_BIN_END == max(int(k.split('-')[1]) for k in size_bins)`
in any loaded FSD.)

---

## 3. Test coverage of the actual scripts

**Current coverage:**

- 57 unit tests across 11 files.
- Direct script tests: `test_pipeline_scripts.py` (12),
  `test_lr_sweep_smoke.py` (1 — static AST check only),
  `test_no_nan_leakage.py` (4), `test_gemma_parse.py` (4),
  `test_honest_benchmark_cli.py` (2 — `--help` smoke only),
  `test_auc_gate.py` (2), `test_nuc_features.py` (12),
  `test_ppv_screening.py` (6), `test_8channel_eval.py` (2),
  `test_fetch_finaledb.py` (6), `test_gemma_baseline.py` (6).
- Scripts NOT covered end-to-end (would require data/features/):
  `lr_no_pca_vs_pca200.py`, `lr_regularization_sweep.py`,
  `nuc_ablation.py`, `model_ablation.py`, `eval_8channel.py`,
  `ppv_screening.py`, `honest_benchmark.py`.

**Finding 5: No script is tested end-to-end without data/features/.**

The closest thing is `test_pipeline_scripts.py` which calls
`summarize()` (the FSD function) on synthetic lengths, but never
runs the full `nuc_ablation.py` or `lr_no_pca_vs_pca200.py` path.
The CI's `auc-gate` job runs `auc_reproducibility_gate.py` end-to-end
on a synthetic cohort — it's the only full-script CI test.

This is acceptable because:
1. The synthetic gate catches feature-pipeline bugs.
2. The static checks catch class-of-bugs (like the L1
   `n_nonzero` NameError).
3. A full E2E test would require committing `data/features/` or
   a small fixture (~30 samples), which conflicts with the
   "data not in repo" philosophy.

**Severity: Nice-to-have.**
**Effort: 2 hours.**

**Concrete fix** — add `test/test_e2e_smoke.py` with a 30-sample
synthetic cohort fixture committed under `test/fixtures/synth_cohort/`:

```python
"""Smoke E2E: build a tiny synthetic 5-channel cohort and run
each major script's main() against it. Catches argparse
regressions and data-path bugs that static checks miss."""
import os, subprocess, sys, tempfile, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "test" / "fixtures" / "synth_cohort"

def _run(cmd, timeout=180):
    return subprocess.run([sys.executable, *cmd], capture_output=True,
                          text=True, cwd=str(ROOT), timeout=timeout,
                          env={**os.environ, "PYTHONPATH": ""})

def test_lr_no_pca_runs():
    r = _run(["scripts/lr_no_pca_vs_pca200.py",
              "--features-dir", str(FIX),
              "--seeds", "2", "--out", "/tmp/_test_lr.json"])
    assert r.returncode == 0, r.stderr[-500:]
    j = json.load(open("/tmp/_test_lr.json"))
    assert "results" in j
    assert all("auc_mean" in x for x in j["results"])
```

The fixture would be ~30 samples × {5mb_ratio, 5mb_coverage,
100kb_ratio, 100kb_counts, fsd.json, motifs.json, gc_corrected.npy}
= ~5MB committed. Add a generator script
`test/fixtures/build_synth_cohort.py` so the fixture can be
regenerated if the FSD bin layout changes.

This makes the CI catch: argparse renames, file-loading path
regressions, "I forgot to call `_harmonize`" bugs, and JSON-output
schema regressions.

---

## 4. Reproducibility on a different machine

**Finding 6: `model_ablation.py` has no argparse, runs at import.**

Per AUDIT_REPORT_3: "4 of 5 with argparse now work from `/tmp`".
The 5th script — `model_ablation.py` — runs `main()` immediately when
imported (no argparse, no `if __name__ == "__main__":` guard around
a heavy function — wait, it does have one, but no argparse means you
can't change `--seeds`, `--out`, or `--features-dir`).

```bash
cd /tmp && python /path/to/cfdna-fragmentomics-pipeline/scripts/model_ablation.py
# → prints "Loading cohort..." and either crashes (data missing) or
#   runs the 3-seed × 4-model × 5-fold benchmark
```

For a fresh-clone user this is the same crash as
`honest_benchmark.py --help` was before round 2's fix.

**Severity: Should-fix** (it's the one outlier the round 3 audit
explicitly flagged; not fixed yet).
**Effort: 15 minutes.**

**Concrete fix** — add argparse at the top of `main()`:

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="/tmp/model_ablation.json")
    ap.add_argument("--features-dir", default=str(FEAT_DIR))
    ap.add_argument("--labels-tsv",
                    default=str(LABELS_CROSS_STUDY_TSV))
    args = ap.parse_args()
    ...  # use args.out, args.features_dir, args.labels_tsv
```

And update the `if __name__ == "__main__":` guard to call
`sys.exit(main())`.

**Finding 7: `lr_regularization_sweep.py` defaults to slow L1.**

Mentioned in Finding 1 — same severity.

---

## 5. Disk and memory footprint

**Measured:**
- `data/features/`: 407 MB (gitignored)
- Gemma 2 9B Q4_K_M: 5.4 GB
- Working set for `lr_no_pca_vs_pca200.py` 10-seed run: ~3 GB
  (627 × ~60k float64 X matrix after harmonization)

**Finding 8: 407 MB of features is gitignored but reproducible.**

A fresh user needs to run `python run_cross_study.py` which downloads
~100 GB of raw FinaleDB fragments and takes 1-3 hours to extract.
The 407 MB gitignore is honest — those features depend on FinaleDB
being available. But the README doesn't quantify:

- **Disk** — "100 GB download" but no mention of 407 MB intermediate.
- **Time** — "1-3 hours" is a wide range; on M1 Pro / SATA SSD it's
  closer to 1.5 hours.
- **Memory** — feature extraction holds each sample's ~28M fragments
  in memory (`np.histogram`), peaking at ~1 GB per sample. The
  parallel path uses `--parallel 8` so peak RSS is ~8 GB.

**Severity: Should-fix** (fresh users will hit OOM on laptops with
16 GB RAM if they don't know to use `--parallel 2`).
**Effort: 15 minutes** (doc only).

**Concrete fix** — extend README.md "Data not in repo" callout
(line ~13) with:

```markdown
> **Resource budget** (FinaleDB repro):
> - Disk: 100 GB raw + 407 MB features
> - Time: 1-3 hours (network + extraction)
> - RAM: peak ~8 GB with `--parallel 8`; for 16 GB laptops use
>   `--parallel 2` (slower wall-clock but safer)

> **Smoke-test path** (no FinaleDB needed, ~30 s):
> `python scripts/auc_reproducibility_gate.py` — verifies the
> pipeline path with a synthetic cohort.
```

**Finding 9: Gemma 2 9B model is 5.4 GB and gated.**

The Gemma baseline uses `DEFAULT_GEMMA_MODEL_PATH = ~/models/gemma-2-9b-it-Q4_K_M.gguf`.
There's no documented fallback if the model is missing — the script
crashes with `FileNotFoundError`. The model is also not on HuggingFace
under a permissive license without accepting Google's terms.

**Severity: Nice-to-have** (the Gemma baseline is a stretch goal;
crashing without it doesn't break the headline numbers).
**Effort: 30 minutes.**

**Concrete fix** — `scripts/gemma_baseline.py:295` (or wherever
the model load happens) should print a clear "model missing,
skip Gemma baseline" message and exit 0 (or skip with a recorded
result of `null`) instead of stack-tracing.

---

## 6. CI / GitHub Actions robustness

**Current CI** (`.github/workflows/pipeline-tests.yml`):
- `test` job: `pytest test/ -v` on ubuntu-latest, Python 3.11.
- `auc-gate` job: runs `auc_reproducibility_gate.py` (synthetic, ~30s).
- `cli-smoke` job: 4 console_scripts `--help`.

**Finding 10: CI does not catch script-import regressions.**

If a script imports `from train_classifier import evaluate_cv` and
someone renames the function, the unit tests still pass (they don't
import the script), but the script crashes for fresh users.
The `cli-smoke` job checks the 4 console_scripts, but `lr_*`,
`nuc_ablation.py`, `model_ablation.py`, `eval_8channel.py`,
`ppv_screening.py`, `honest_benchmark.py` are NOT console_scripts
(documented in pyproject.toml) and NOT in `cli-smoke`.

**Severity: Should-fix** (this is exactly the "silent failure"
class — passes CI, breaks fresh user).
**Effort: 20 minutes.**

**Concrete fix** — extend `cli-smoke` job with one line:

```yaml
- name: All scripts import without error
  run: |
    for script in scripts/lr_no_pca_vs_pca200.py \
                  scripts/lr_regularization_sweep.py \
                  scripts/nuc_ablation.py \
                  scripts/model_ablation.py \
                  scripts/eval_8channel.py \
                  scripts/ppv_screening.py \
                  scripts/honest_benchmark.py \
                  scripts/auc_reproducibility_gate.py \
                  scripts/train_classifier.py; do
      python -c "import importlib.util, sys; \
        spec = importlib.util.spec_from_file_location('m', '$script'); \
        m = importlib.util.module_from_spec(spec); \
        spec.loader.exec_module(m); \
        print('OK:', '$script')" || (echo "FAIL: $script"; exit 1)
    done
```

This catches: syntax errors, top-level `NameError` regressions,
circular imports, broken `_paths.py` edits. Does NOT catch
data-path bugs (those still need the E2E test from Finding 5).

**Finding 11: CI never tests the L1 regression directly.**

The `test_lr_sweep_smoke.py` test is a STATIC check — it parses
`lr_regularization_sweep.py` source and asserts the bug token isn't
there. This is great for guarding against re-introduction, but if someone
makes the L1 path silent (e.g. always returns empty) the static
check still passes.

**Severity: Nice-to-have.**
**Effort: 1 hour** — would require committing a 30-sample synthetic
cohort fixture and a 1-fold L1 sanity test. Same fixture as Finding 5.

**Finding 12: No CI lint job.**

`pyproject.toml` declares `ruff>=0.1` as a dev dep but no CI job
runs it. A 30-line addition would catch unused imports, undefined
names (ruff F821), and bare except clauses.

**Severity: Nice-to-have.**
**Effort: 5 minutes.**

**Concrete fix** — add to `pipeline-tests.yml`:

```yaml
  lint:
    name: ruff
    runs-on: ubuntu-latest
    timeout-minutes: 2
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -q ruff
      - run: ruff check scripts/ test/
```

---

## 7. Documentation gaps for fresh-clone users

**Finding 13: README says "Python 3.11" but `.venv` is 3.14.**

`pyproject.toml` declares `requires-python = ">=3.10"`. README has
"Python 3.11" in envs/environment.yml. The dev `.venv` is 3.14.
The CI runs 3.11. None of these break anything (`pip install -e .`
works in all three), but a fresh user who reads the README and
installs 3.11 will see slight version mismatches in `pip freeze`
that are harmless but confusing.

**Severity: Nice-to-have.**
**Effort: 5 minutes** — change README to "Python 3.10+".

**Finding 14: No "minimum reproducible path" section.**

The README documents the full `run_cross_study.py` workflow but
doesn't say "if you only have 30 minutes, here's the 3-command
minimum" — that would be:

```bash
git clone ...
cd cfdna-fragmentomics-pipeline
pip install -e ".[dev]"
pytest test/ -v                       # 5s — all tests pass
python scripts/auc_reproducibility_gate.py   # 30s — synthetic gate
```

This is in the "Data not in repo" callout implicitly, but a
fresh user who lands on the README might miss it.

**Severity: Should-fix.**
**Effort: 10 minutes** — add a "Minimum reproducible path" section
to README.md right after the "Data not in repo" callout:

```markdown
## Minimum reproducible path (no data needed)

If you just want to verify the pipeline runs end-to-end on a
fresh checkout, three commands and ~30 seconds:

\`\`\`bash
pip install -e ".[dev]"                 # ~10s
pytest test/ -v                          # ~5s, 57 tests
python scripts/auc_reproducibility_gate.py  # ~30s, AUC >= 0.80
\`\`\`

This exercises the full pipeline path (data loading, harmonization,
PCA, LR, CV) on a synthetic 200-sample cohort with a known signal.
No FinaleDB download required.
```

**Finding 15: No `--list-scripts` or "what does each script do"
section.**

A fresh user who clones the repo sees 20 scripts in `scripts/`
and has to read each docstring to figure out what they do. A 1-page
table in the README would help:

| Script | What it does | Time |
|---|---|---|
| `auc_reproducibility_gate.py` | synthetic-cohort AUC gate | 30s |
| `lr_no_pca_vs_pca200.py` | headline 5-seed AUC | 1 min |
| `lr_regularization_sweep.py` | (penalty, C) grid | 83s (--skip-l1) |
| `nuc_ablation.py` | nucleosome-feature ablation | 290s |
| `model_ablation.py` | LR/RF/GB comparison | ~5 min |
| `eval_8channel.py` | 8-channel vs 5-channel | 1 min |
| `ppv_screening.py` | PPV/NPV at 4 prevalences | <10s |
| `honest_benchmark.py` | full multi-cohort | 10+ min |
| `gemma_baseline.py` | LLM baseline (needs Gemma) | ~30 min |
| `extract_fsd.py` | FSD per-sample | ~5s/sample |
| `extract_delfi.py` | DELFI per-sample | ~5s/sample |
| `fetch_finaledb.py` | FinaleDB downloader | 1-3 min/sample |

**Severity: Nice-to-have.**
**Effort: 20 minutes.**

**Finding 16: README "Reproducibility" section claims
`results/classifier_results.json` exists but it doesn't.**

README line ~228: "Full result JSON in `results/classifier_results.json`".
This file is not produced by any current script — they produce
`/tmp/lr_no_pca_vs_pca200.json`, `/tmp/lr_reg_sweep.json`,
`/tmp/nuc_ablation.json`, etc. (all in /tmp).

**Severity: Should-fix** (a fresh user looking for the headline
numbers will find nothing).
**Effort: 5 minutes** — change to "Per-script result JSONs in
`/tmp/{lr_no_pca_vs_pca200,lr_reg_sweep,nuc_ablation,...}.json`
or under `results/` if `--out` is overridden."

**Finding 17: No "what changed in the last 3 rounds" note.**

A reviewer coming to the repo cold has to read all three
AUDIT_REPORT*.md files to understand what bugs have been fixed.
A 5-line summary at the top of README would help:

```markdown
## Recent audit rounds (2026-08)

- **Round 1** (13 issues): NaN-leakage in CV, Gemma parse-failure
  silent bias, hardcoded paths, threshold-optimization on test set.
- **Round 2** (4 fixes): honest_benchmark.py --help crash, PPV at
  screening prevalence added, 8-channel evaluation, 7→8/10 benchmark
  corrections.
- **Round 3** (4 fixes): nuc_ablation duplicate _evaluate removed,
  paths centralized in scripts/_paths.py, NaN imputation locked to
  train fold, Gemma parse-failure tracking.

See `AUDIT_REPORT.md`, `AUDIT_REPORT_2.md`, `AUDIT_REPORT_4.md`
for full detail.
```

**Severity: Nice-to-have.**
**Effort: 10 minutes.**

---

## Summary table

| # | Question | Finding | Severity | Effort |
|---|---|---|---|---|
| 1 | Runtime | `lr_regularization_sweep.py` default = 35 min | Nice-to-have | 5 min |
| 2 | Runtime | GB in `model_ablation.py` is slow | Nice-to-have | 10 min |
| 3 | Runtime | 4× redundant PCA fits in `nuc_ablation.py` | Nice-to-have | 30 min |
| 4 | Reusability | 5bp bin stride hardcoded in 2+ files | Should-fix | 30 min |
| 5 | Test coverage | No script tested E2E without data/features | Nice-to-have | 2 h |
| 6 | Repro | `model_ablation.py` lacks argparse | Should-fix | 15 min |
| 7 | Repro | (same as #1) | — | — |
| 8 | Disk/mem | README doesn't quantify RAM/Disk/Time | Should-fix | 15 min |
| 9 | Disk/mem | Gemma model missing → FileNotFoundError | Nice-to-have | 30 min |
| 10 | CI | No script-import regression check | Should-fix | 20 min |
| 11 | CI | L1 path not dynamically tested | Nice-to-have | 1 h |
| 12 | CI | No `ruff` lint job | Nice-to-have | 5 min |
| 13 | Docs | README says Python 3.11 (actually 3.10+) | Nice-to-have | 5 min |
| 14 | Docs | No "minimum reproducible path" section | Should-fix | 10 min |
| 15 | Docs | No "what each script does" table | Nice-to-have | 20 min |
| 16 | Docs | README refs `results/classifier_results.json` (doesn't exist) | Should-fix | 5 min |
| 17 | Docs | No "what changed in last N rounds" note | Nice-to-have | 10 min |

**Should-fix items total effort: ~1.5 hours.**
**Nice-to-have items total effort: ~5 hours.**

**Priorities for next round (this audit's recommendations):**

1. **Finding 6** (15 min) — add argparse to `model_ablation.py`.
2. **Finding 16** (5 min) — fix README's `results/classifier_results.json` reference.
3. **Finding 10** (20 min) — add CI script-import check.
4. **Finding 14** (10 min) — add "Minimum reproducible path" section.
5. **Finding 4** (30 min) — centralize FSD bin layout constants.

That's ~1.5 hours of work to address every Should-fix in this round.
Nice-to-haves can be batched into a single follow-up PR.

---

## What was NOT audited (out of scope)

- Methodology correctness (covered in rounds 1-3 STATISTICAL_REVIEW,
  SCIENTIFIC_REVIEW).
- Per-cancer-type AUC table (S1 from round 2 — cancer type not in
  current labels file, can't do without upstream data changes).
- ComBat / limma-style harmonization (needs new dependency).
- Documentation of per-script JSON output schemas (would help with
  Finding 5's E2E test design).