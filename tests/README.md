# Test suite

    python3 -m venv .venv-test
    .venv-test/bin/pip install -r requirements-dev.txt
    .venv-test/bin/pytest -m "not probe"     # the green gate
    .venv-test/bin/pytest -m probe -s        # the diagnostics (expected to fail)

## Layout

    tests/conftest.py           import shims for scripts/ and vlm/, shared fixtures
    tests/fixtures/synth.py     synthetic data factory -- the backbone of the suite
    tests/control/              negative/positive controls: leakage, confounding, calibration
    tests/pipeline/             CT phantoms: segmentation, mouse identity, TBR strategies
    tests/torch/                model internals on a small randomly-initialised Llama
    tests/unit/                 estimator behaviour, parsing round-trips, determinism

## Markers

| Marker | Meaning |
|---|---|
| `unit` | pure logic, no torch, <1s |
| `torch` | needs torch/transformers on CPU (a small randomly-initialised Llama) |
| `slow` | permutation/bootstrap suites, many model fits |
| `realdata` | needs the real dataset; skipped unless `SCAI_DATA_ROOT` is set |
| `probe` | **a diagnostic that is expected to FAIL on unfixed code — the failure is the finding** |

## About `probe`

Most of this suite is not a regression guard. The failure mode that matters in this
project is not "the code crashes" — it is "the code quietly reports a number that came
from leakage or from noise." Unit tests cannot catch that; negative controls can.

A `probe` test feeds the real analysis code data with *no signal* and asserts the metric
collapses to chance. When it fails, something is producing signal that cannot be there.
Each probe's assertion message states the mechanism, the measured magnitude, and the fix.

Fix the underlying issue and the probe goes green — at which point it moves into the CI
gate as a permanent guard. F4, F6, F7, F8 and F9 have already made that transition;
their tests now assert the fixed behaviour and would catch a regression.

## Running against real data

    SCAI_DATA_ROOT=/data1/Processed_NIfTI_Test .venv-test/bin/pytest -m realdata -s

Two `realdata` tests are the ones that settle open questions:

* `test_genotype_bit_influence_realdata` — measures how much the genotype conditioning
  bit steers the longitudinal MLP's predictions on the actual RAD-DINO embeddings.
* `test_session_confound_realdata` — compares genotype AUROC under leave-one-subject-out
  vs leave-one-session-out. Currently blocked: the `.npz` contract has no `session_id`.

## Findings

`docs/FINDINGS.md` is the register: F1–F12, each with status, measured magnitude, the
published claims it affects, and the test that demonstrates it. `docs/TESTING_PLAN.md`
holds the wider plan and the original read-derived hypotheses.

Two of twelve came back **refuted** (F11, F12) — plausible-sounding mechanisms that the
tests ruled out. Recorded as results, not deleted.
