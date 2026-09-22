# Test suite

Use the existing recorded research environment on this server:

    /home/ryab/miniconda3/envs/vlm_env/bin/python -m venv --system-site-packages .venv-test
    .venv-test/bin/pip install pytest pytest-cov hypothesis
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/home/ryab/miniconda3/envs/vlm_env/lib .venv-test/bin/python -m pytest -m "not probe and not realdata" -q

Do not recreate an existing working environment unnecessarily. For a different machine, install the recorded research dependencies before adding test tooling; historical claims about whatever torch wheel is latest are not a reproducibility specification.

This also tests against the same torch/transformers the pipeline actually runs on, rather
than whatever latest resolves to.

Timings: the green gate takes ~60-80 s. The `probe` suite takes ~10 min — it fits many
tiny LLMs. Don't kill a slow `pip install` partway through: deleting the tree mid-install
leaves a `.venv-test` with orphaned `site-packages` and no interpreter, which then fails
confusingly. Let it finish, or `rm -rf` it only once pip has exited.

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

The suite includes regression guards and separately marked historical diagnostic probes. The failure mode that matters in this
project is not "the code crashes" — it is "the code quietly reports a number that came
from leakage or from noise." Different tests cover different failure modes; a passing negative control does not rule out all leakage or scientific confounding.

A `probe` test feeds the real analysis code data with *no signal* and asserts the metric
collapses to chance. When it fails, something is producing signal that cannot be there.
Each probe's assertion message states the mechanism, the measured magnitude, and the fix.

Fix the underlying issue and the probe goes green — at which point it moves into the CI
gate as a permanent guard. F4, F6, F7, F8 and F9 have already made that transition;
their tests now assert the fixed behaviour and would catch a regression.

## Running against real data

    SCAI_DATA_ROOT=/data1/Processed_NIfTI_Test .venv-test/bin/pytest -m realdata -s

Historical exploratory real-data probes include:

* `test_genotype_bit_influence_realdata` — measures how much the genotype conditioning
  bit steers the longitudinal MLP's predictions on the actual RAD-DINO embeddings.
* `test_session_confound_realdata` — compares genotype AUROC under leave-one-subject-out
  vs leave-one-session-out. The current manifest joins resolve session IDs; acquisition components are the relevant genotype grouping.

## Findings

`docs/FINDINGS.md` preserves the historical register: F1–F24, each with status, measured magnitude, the
published claims it affects, and the test that demonstrates it. `docs/TESTING_PLAN.md`
holds the historical plan and original hypotheses. Current interpretation is in [STATUS.md](../docs/STATUS.md).

Two of twelve came back **refuted** (F11, F12) — plausible-sounding mechanisms that the
tests ruled out. Recorded as results, not deleted.

**`libstdc++` ABI error (`CXXABI_1.3.15 not found`) when importing SimpleITK.** Import-order
dependent: if a script loads a system-linked extension before SimpleITK, the older system
`libstdc++` is already mapped and SimpleITK's newer symbols are missing. It bit
`get_colipri_embeddings.py` in a fresh tmux shell and, earlier, a venv sanity check. Fix is
to put conda's lib first: `export LD_LIBRARY_PATH=/home/ryab/miniconda3/envs/vlm_env/lib`.

## Fixed-protocol closure gate (2026-09-14)

The consolidated CPU gate passed 114 tests; the separate 29-test follow-up included the current real crop-manifest check. The 18 excluded probe/real-data cases are not all claimed to have passed. Actual quantized-backbone, corrected-crop, target-recomputation and nested-export checks have separate evidence in [the closure report](../docs/audit_2026-09-14/PIPELINE_CLOSURE.md).

Use `SCAI_DATA_ROOT=/data1/Processed_NIfTI_Test` for the real identity check. Noncontiguous crop positions can be valid after explicit phantom exclusions; the test checks source filename positions and overrides, not contiguous numbering alone.

## Later focused checks

The completed four-fold amendment, residual forecaster/VLM, and genotype variants have their own tests and artifact checks. The latest genotype variants passed six focused tests and 24 exact final checkpoint reloads; their metrics were independently verified. See [all results](../docs/POST_AUDIT_RESULTS.md) and [experiment evidence](../docs/audit_2026-09-14/README.md). The recorded 114-test count above belongs to the original gate, not every later revision of the suite.

## Current pre-commit gate (2026-09-21)

The current suite passed **135 tests, with 18 probe/real-data cases deselected**, in 162.24 seconds. This CPU-only, offline run did not repeat the historical GPU or real-data experiments. The Transformers `evaluation_strategy` deprecation warning does not affect this result. [Full output](../docs/audit_2026-09-14/precommit_tests_2026-09-21.log).

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/scai-matplotlib \
NUMBA_CACHE_DIR=/tmp/scai-numba LD_LIBRARY_PATH=/home/ryab/miniconda3/envs/vlm_env/lib \
.venv-test/bin/python -m pytest -m 'not probe and not realdata' -q
```
