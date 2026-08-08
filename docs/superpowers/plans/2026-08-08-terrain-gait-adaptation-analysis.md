# Terrain-Conditioned Gait Adaptation Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the offline analysis so full zero-delay evaluations feed existing gait-dynamics aggregation and produce a compact, command-matched LEP terrain-adaptation figure and tables without running simulation.

**Architecture:** Generalize evaluation metadata and candidate selection in the existing discovery modules, then add a pure NumPy/Pandas terrain-analysis module that consumes `metrics_summary.json` plus `plots/sim_data.npz`. `analyze_run_range.py` remains the orchestrator and preserves every existing output while adding terrain-specific CSV, Markdown, manifest, and figure artifacts.

**Tech Stack:** Python 3.11, NumPy, Pandas, Matplotlib, pytest; existing `metrics_utils.py`, `gait_dynamics_aggregate.py`, and `analyze_run_range.py` utilities.

## Global Constraints

- Work in `/home/kordoslo/dev/terrain_gait_adaptation_analysis`; do not create a worktree or environment.
- Do not launch `eval.py`, training, Isaac Sim, or any process that allocates VRAM.
- Preserve all existing LP--LEP gait-dynamics metrics, reports, and plot families.
- Accept full zero-delay evaluations even when `rebuttal_scenarios_only` is false; retain purpose-built rebuttal evaluations as fallback.
- Use `walk_x_flat_terrain_1.0mps` and `fast_walk_x_uneven_terrain`, both commanded at `(1.0, 0.0, 0.0)`, for terrain claims.
- Reconstruct contact as resultant force strictly greater than `1.0 N` and exclude the first `1.0 s` plus reset-clipped data.
- Use one trained run as the statistical unit; do not pool stride events as independent seeds.
- Add only a small focused test set and run it in the existing virtual environment.

---

### Task 1: Generalize Evaluation Discovery and Deterministic Selection

**Files:**
- Modify: `scripts/gait_dynamics_aggregate.py`
- Modify: `scripts/analyze_run_range.py:583-774`
- Create: `tests/test_evaluation_discovery.py`

**Interfaces:**
- Produces: `extract_evaluation_metadata(summary: dict[str, Any], json_path: Path) -> EvaluationMetadata`
- Produces: generalized `discover_rebuttal_gait_dynamics_files(...)` accepting full and rebuttal-only zero-delay summaries.
- Produces: general JSON discovery that parses suffixed checkpoint directories and prefers full zero-delay data at equal checkpoints.
- Preserves: `RebuttalGaitRunData.metrics_summary`, `build_rebuttal_per_run_dataframe(...)`, and every existing aggregate metric.

- [ ] **Step 1: Write focused discovery tests**

Create temporary `metrics_summary.json` fixtures covering full zero-delay eligibility, rebuttal-only fallback, full-over-rebuttal tie-breaking, and non-zero-delay exclusion:

```python
def test_full_zero_delay_summary_is_selected(tmp_path):
    summary = make_summary(rebuttal_scenarios_only=False, action_delay_steps=0)
    write_summary(tmp_path / "run/eval_checkpoint_100_seed_46_action_delay_0", summary)
    selected, manifest = discover_rebuttal_gait_dynamics_files([tmp_path])
    assert selected[("env", "2026-01-01-00-00-00")].evaluation_scope == "full"
    assert manifest.loc[manifest["selected"], "selection_reason"].item() == "highest_checkpoint_full_zero_delay"


def test_rebuttal_only_is_fallback_and_delayed_eval_is_excluded(tmp_path):
    write_summary(
        tmp_path / "run/eval_checkpoint_100_seed_46_action_delay_0_rebuttal_scenarios",
        make_summary(rebuttal_scenarios_only=True, action_delay_steps=0),
    )
    write_summary(
        tmp_path / "run/eval_checkpoint_101_seed_46_action_delay_1",
        make_summary(rebuttal_scenarios_only=False, action_delay_steps=1),
    )
    selected, manifest = discover_rebuttal_gait_dynamics_files([tmp_path])
    assert selected[("env", "2026-01-01-00-00-00")].evaluation_scope == "rebuttal_only"
    assert not manifest.loc[manifest["action_delay_steps"] == 1, "eligible"].item()
```

- [ ] **Step 2: Run the discovery tests and verify the current code fails**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m pytest -q tests/test_evaluation_discovery.py
```

Expected: failures because full summaries are rejected, scope/selection metadata does not exist, and suffixed paths are not consistently resolved.

- [ ] **Step 3: Implement shared evaluation metadata and eligibility**

Add a frozen metadata record and path parser that handles arbitrary suffixes after the seed:

```python
@dataclass(frozen=True)
class EvaluationMetadata:
    checkpoint: int | None
    action_delay_steps: int
    evaluation_scope: str


def extract_evaluation_metadata(summary: dict[str, Any], json_path: Path) -> EvaluationMetadata:
    checkpoint = _extract_checkpoint_with_suffix(json_path)
    action_delay = int(summary.get("action_delay_steps", 0))
    scope = "rebuttal_only" if summary.get("rebuttal_scenarios_only") is True else "full"
    return EvaluationMetadata(checkpoint, action_delay, scope)
```

Replace the `rebuttal_scenarios_only is True` eligibility gate with a non-empty gait-dynamics mapping plus zero action delay. Prefer full scope whenever available, then select the highest checkpoint within that scope, and raise only if equally preferred candidates remain.

- [ ] **Step 4: Generalize normal JSON checkpoint resolution**

Update `extract_checkpoint_from_path()` to accept suffixes and use the same preference at equal checkpoints:

```python
pattern = re.compile(r"^eval_checkpoint_(\d+)(?:_seed_\d+)?(?:_.*)?$")
```

Prefer zero action delay and full scope so `analyze_run_range.py` obtains full scenario data rather than rebuttal-only data, then choose the highest checkpoint within that scope. If an old unsuffixed full evaluation and a current full evaluation still tie, prefer explicit zero-delay/current-schema data (a non-empty gait-dynamics payload); fail only when equally preferred candidates remain. Add `action_delay_steps`, `evaluation_scope`, `eligible`, `selected`, and `selection_reason` to discovery manifests.

- [ ] **Step 5: Run the focused tests**

Run the Task 1 pytest command. Expected: all discovery tests pass.

- [ ] **Step 6: Commit discovery support**

```bash
git add scripts/gait_dynamics_aggregate.py scripts/analyze_run_range.py tests/test_evaluation_discovery.py
git commit -m "fix: include full evals in gait analysis discovery"
```

---

### Task 2: Implement Pure Terrain Contact Metrics

**Files:**
- Create: `scripts/terrain_adaptation_analysis.py`
- Create: `tests/test_terrain_adaptation_analysis.py`

**Interfaces:**
- Produces: `TerrainRunSource` containing label/run identity, checkpoint, JSON path, and summary.
- Produces: `compute_scenario_contact_metrics(sim_data, mask, foot_labels, step_dt) -> dict[str, Any]`.
- Produces: `analyze_terrain_sources(sources, flat_tag, uneven_tag) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` for per-scenario rows, paired rows, and manifest rows.
- Consumes: `build_scenario_analysis_mask`, `compute_stance_segments`, and `compute_swing_heights` from `metrics_utils.py`.

- [ ] **Step 1: Write synthetic contact-metric tests**

Use foot labels in a deliberately noncanonical order and construct periodic versus timing-modulated contact arrays:

```python
def test_contact_metrics_resolve_diagonal_feet_by_label_and_exclude_clipped_segments():
    labels = ["RR_foot", "FL_foot", "FR_foot", "RL_foot"]
    contacts = make_periodic_trot(labels, stance_steps=4, swing_steps=4, cycles=5)
    metrics = compute_scenario_contact_metrics(
        sim_data=make_sim_data(contacts, labels),
        analysis_mask=np.ones(len(contacts), dtype=bool),
        foot_labels=labels,
        step_dt=0.02,
    )
    assert metrics["diagonal_support_occupancy"] == pytest.approx(1.0)
    assert metrics["stance_duration_iqr_s"] == pytest.approx(0.0)


def test_modulated_contacts_increase_duration_iqr():
    flat = compute_metrics_for_durations([4, 4, 4, 4])
    uneven = compute_metrics_for_durations([2, 4, 6, 8])
    assert uneven["stance_duration_iqr_s"] > flat["stance_duration_iqr_s"]
    assert uneven["swing_duration_iqr_s"] > flat["swing_duration_iqr_s"]
```

- [ ] **Step 2: Run the metric tests and verify they fail**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m pytest -q tests/test_terrain_adaptation_analysis.py
```

Expected: import/definition failures because the new module does not exist.

- [ ] **Step 3: Implement reset-safe contact and duration metrics**

Implement force-threshold reconstruction, label-based diagonal masks, complete-segment duration extraction, IQR, swing height, planar RMSE, completion, and analyzed duration. Validate time/foot dimensions and return `None` for metrics without valid events rather than fabricated zero values.

```python
contact_state = np.asarray(sim_data["contact_forces_array"], dtype=float) > 1.0
fl, fr, rl, rr = resolve_foot_indices(foot_labels)
diagonal = (
    contact_state[:, fl] & contact_state[:, rr] & ~contact_state[:, fr] & ~contact_state[:, rl]
) | (
    contact_state[:, fr] & contact_state[:, rl] & ~contact_state[:, fl] & ~contact_state[:, rr]
)
diagonal_support_occupancy = float(diagonal[analysis_mask].mean())
```

- [ ] **Step 4: Implement source-level offline analysis**

For each source, resolve `plots/sim_data.npz`, infer the two scenario ranges from recorded metadata, apply `1.0 s` warm-up and reset truncation, compute both rows, and create a paired row with flat, uneven, and uneven-minus-flat values. Missing data appends a manifest warning and skips only that run's terrain rows.

- [ ] **Step 5: Run the metric tests**

Run the Task 2 pytest command. Expected: all pure metric tests pass without importing Isaac Lab.

- [ ] **Step 6: Commit terrain metric computation**

```bash
git add scripts/terrain_adaptation_analysis.py tests/test_terrain_adaptation_analysis.py
git commit -m "feat: compute offline terrain contact adaptation metrics"
```

---

### Task 3: Add Aggregate Tables, Report, and Compact Figure

**Files:**
- Modify: `scripts/terrain_adaptation_analysis.py`
- Modify: `tests/test_terrain_adaptation_analysis.py`

**Interfaces:**
- Produces: `aggregate_terrain_adaptation(paired_df) -> pd.DataFrame` with seed-level mean/std/SEM/CI/count.
- Produces: `select_representative_run(paired_df, label) -> str` using distance to median headline deltas.
- Produces: `write_terrain_adaptation_outputs(...) -> dict[str, str]`.
- Outputs: `terrain_adaptation_per_scenario.csv`, `terrain_adaptation_paired.csv`, `terrain_adaptation_summary.csv`, `terrain_adaptation_manifest.csv`, `terrain_adaptation_report.md`, and `plot_terrain_adaptation.<format>`.

- [ ] **Step 1: Add aggregation and representative-selection tests**

```python
def test_aggregation_uses_one_paired_value_per_run():
    paired = pd.DataFrame(make_three_run_pairs())
    summary = aggregate_terrain_adaptation(paired)
    row = summary.query("metric == 'diagonal_support_occupancy_delta'").iloc[0]
    assert row["n"] == 3
    assert row["mean"] == pytest.approx(np.mean([-0.10, -0.20, -0.15]))


def test_representative_run_is_closest_to_median_headline_changes():
    paired = pd.DataFrame(make_three_run_pairs())
    assert select_representative_run(paired, "LEP") == "median-run"
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run the Task 2 pytest command. Expected: only the newly added aggregation/output assertions fail.

- [ ] **Step 3: Implement paired aggregation and report rendering**

Aggregate per label from paired run rows only. Render explicit metric definitions, command matching, scenario completion warnings, data-source paths, and the boundary that the analysis demonstrates terrain-conditioned timing modulation rather than discrete gait-family transitions.

- [ ] **Step 4: Implement the compact four-panel figure**

Create two contact-raster axes for the deterministically selected LEP run and two paired run-level axes. Plot individual paired lines lightly and overlay mean plus 95% CI. Use the requested export formats and existing publication style conventions.

```python
figure = plt.figure(figsize=(figure_width, figure_height), layout="constrained")
grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
# left: flat/uneven contact rasters; right: occupancy and stance-IQR paired plots
```

- [ ] **Step 5: Run the terrain-analysis tests**

Run the Task 2 pytest command. Expected: all metric, aggregation, representative-selection, and output tests pass.

- [ ] **Step 6: Commit output generation**

```bash
git add scripts/terrain_adaptation_analysis.py tests/test_terrain_adaptation_analysis.py
git commit -m "feat: plot terrain-conditioned contact adaptation"
```

---

### Task 4: Integrate with `analyze_run_range.py` and Correct Step-Height Defaults

**Files:**
- Modify: `scripts/analyze_run_range.py`
- Modify: `tests/test_evaluation_discovery.py`

**Interfaces:**
- Consumes: selected full `JsonRunData` entries and converts them to `TerrainRunSource`.
- Consumes: `analyze_terrain_sources(...)` and `write_terrain_adaptation_outputs(...)`.
- Changes default: `DEFAULT_STEP_HEIGHT_UNEVEN_SCENARIO_TAG = "fast_walk_x_uneven_terrain"`.
- Adds CLI: `--terrain_adaptation_primary_label` with default `LEP`.

- [ ] **Step 1: Add an integration test for a full non-rebuttal evaluation**

Build a small temporary full evaluation containing `metrics_summary.json` and `plots/sim_data.npz`, invoke the local-analysis integration functions, and assert that both existing gait discovery and terrain outputs use it despite `rebuttal_scenarios_only == false`.

```python
assert selected_rebuttal[("env", run_name)].evaluation_scope == "full"
assert set(paired_df["run_name"]) == {run_name}
assert paired_df.iloc[0]["uneven_scenario"] == "fast_walk_x_uneven_terrain"
```

- [ ] **Step 2: Run the integration test and verify it fails**

Run:

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m pytest -q tests/test_evaluation_discovery.py tests/test_terrain_adaptation_analysis.py
```

Expected: the integration assertion fails until orchestration is connected.

- [ ] **Step 3: Integrate terrain outputs without altering existing outputs**

After existing step-height and gait-dynamics output generation, construct sources only from selected local JSON runs, run the terrain analysis, save its manifest/tables/report/figure, and log skips. Do not change the existing `write_rebuttal_aggregate_outputs(...)` invocation or metric families.

- [ ] **Step 4: Correct the matched-speed step-height default**

Change only the default uneven tag from `medium_walk_x_uneven_terrain` to `fast_walk_x_uneven_terrain`; retain both CLI overrides so older analyses remain reproducible.

- [ ] **Step 5: Run the focused suite**

Run the Task 4 pytest command. Expected: all tests pass.

- [ ] **Step 6: Commit integration**

```bash
git add scripts/analyze_run_range.py tests/test_evaluation_discovery.py
git commit -m "feat: integrate terrain adaptation run analysis"
```

---

### Task 5: Verify Against Existing Archived Evaluations

**Files:**
- Modify only if verification exposes a defect: files from Tasks 1--4.

**Interfaces:**
- Verifies: no-GPU compatibility with old full evaluation arrays.
- Verifies: full summaries without `rebuttal_scenarios_only` participate in gait aggregation.
- Verifies: purpose-only summaries remain discoverable when full summaries are absent.

- [ ] **Step 1: Run syntax and focused tests**

```bash
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m py_compile scripts/gait_dynamics_aggregate.py scripts/terrain_adaptation_analysis.py scripts/analyze_run_range.py
/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python -m pytest -q tests/test_evaluation_discovery.py tests/test_terrain_adaptation_analysis.py
```

Expected: compilation succeeds and all focused tests pass.

- [ ] **Step 2: Run one archived LEP source through the offline terrain path**

Use the existing unsuffixed full evaluation under run `2026-05-30-00-15-54`, calling only terrain-analysis functions. Expected results are approximately:

```text
flat diagonal-support occupancy > uneven diagonal-support occupancy
flat stance-duration IQR < uneven stance-duration IQR
flat swing height < uneven swing height
```

No Isaac Lab import, simulator startup, or GPU allocation may appear in output.

- [ ] **Step 3: Audit real discovery over both supplied roots**

Call the discovery functions read-only over the LP and LEP roots. Confirm:

```text
completed full zero-delay summaries are eligible
rebuttal-only summaries remain eligible fallback candidates
action-delay 1/2 summaries are excluded from paper aggregation
no duplicate-at-latest-checkpoint exception occurs for a full/rebuttal pair
```

- [ ] **Step 4: Inspect generated CSV/report and figure**

Confirm that scenario commands match, run counts use trained runs, missing-scenario warnings are explicit, the raster is readable, and the figure contains no duty-factor or conditional-two-foot metric.

- [ ] **Step 5: Run repository checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intentional implementation changes are present.

- [ ] **Step 6: Commit any verification fixes**

If Task 5 required code changes:

```bash
git add scripts tests
git commit -m "fix: harden offline terrain adaptation analysis"
```

If no fixes were required, do not create an empty commit.
