# Terrain-Conditioned Gait Adaptation Analysis Design

## Objective

Add compact, rebuttal-ready evidence that one LEP policy changes its contact timing on terrain while retaining an efficient nominal trot. Preserve the existing LP--LEP gait-dynamics metrics, plots, and reports used to explain sim-to-real transfer. All new analysis must run offline from completed evaluation artifacts; it must not require training or simulation.

## Scientific Claim and Scope

The analysis will support the narrower claim that LEP performs continuous, terrain-conditioned modulation of foot-contact timing and clearance around a nominal trotting gait. It will not claim that the fixed global energy weight induces discrete velocity-dependent gait-family transitions.

The primary controlled comparison is:

- Flat: `walk_x_flat_terrain_1.0mps`
- Uneven: `fast_walk_x_uneven_terrain`

Both scenarios command `(vx, vy, yaw) = (1.0, 0.0, 0.0)`. Among the archived command-matched terrain scenarios, the uneven scenario has the largest changes from flat in diagonal-support occupancy, stance-duration variability, and swing-duration variability. Command-confounded diagonal and slower scenarios will not be used to make the terrain-causality claim.

The statistical unit is one independently trained run. Scenario values are reduced to one value per run before paired flat-to-uneven changes and across-run 95% confidence intervals are computed. The final run selection must use the same label, environment, and timestamp ranges passed to `analyze_run_range.py`; exploratory counts over every archived directory are not final paper sample counts.

## Metric Definitions

Contact is reconstructed from `contact_forces_array` using the existing strict resultant-force threshold of greater than `1.0 N`. Foot identities are resolved from `foot_labels`; diagonal pairs are `FL--RR` and `FR--RL`, without assuming a fixed array order.

Each scenario excludes the first `1.0 s` as gait-initiation warm-up. Analysis stops before a premature automatic reset and excludes the expected timeout/reset sample. Stance and swing segments clipped by an analysis-window boundary are excluded from duration distributions.

The rebuttal-facing metrics are:

1. **Diagonal-support occupancy:** fraction of analyzed control steps in exactly the `FL+RR` or `FR+RL` contact state. This is an observed contact-state statistic, not an imposed contact-count or gait prior.
2. **Stance-duration IQR:** within-run interquartile range of complete stance durations, in seconds. This robustly measures terrain-induced contact-timing modulation without allowing isolated one-control-step contacts to dominate the statistic.

Supporting exported metrics are:

- swing-duration IQR;
- mean maximum swing height above local terrain for complete swings;
- planar velocity-tracking RMSE;
- analyzed duration and completion status.

Duty factor, contact-state entropy, and periodicity will not appear in the one-page rebuttal figure. Existing base excitation, GRF, aerial-phase, joint-demand, completion, COT, and operational-limit metrics remain unchanged for the separate LP--LEP sim-to-real argument.

## Rebuttal Figure

Generate one compact figure with:

- two aligned LEP contact rasters at the same `1.0 m/s` forward command, one flat and one uneven;
- paired run-level flat-to-uneven points for diagonal-support occupancy;
- paired run-level flat-to-uneven points for stance-duration IQR.

The figure caption/report will state the swing-duration-IQR and matched-speed swing-height changes. A representative raster run must be selected deterministically as the run closest to the across-run median of the two headline flat-to-uneven changes, rather than selected manually.

The existing paper evidence will be referenced in prose rather than duplicated: LEP versus blind LE terrain progress establishes the role of elevation-map perception, and the corrected matched-speed swing-height comparison establishes terrain-dependent clearance.

## Offline Data Flow

1. `analyze_run_range.py` discovers evaluation summaries under all supplied `--metrics_summary_root_dir` roots.
2. Evaluation metadata parsing recognizes checkpoint directories with or without suffixes such as `_action_delay_0` and `_rebuttal_scenarios`.
3. Full zero-delay evaluations are preferred for general and terrain-adaptation analysis because they contain all fixed scenarios and `plots/sim_data.npz`.
4. Purpose-built rebuttal evaluations remain eligible for the existing LP--LEP gait-dynamics aggregation and are used as a fallback when no full zero-delay evaluation is available.
5. Terrain-adaptation analysis reads the selected evaluation's `metrics_summary.json` and adjacent `plots/sim_data.npz`, infers fixed-scenario ranges from the recorded scenario list and total/random step counts, computes run-level metrics, and writes aggregate tables and the compact figure.
6. A selected evaluation missing either matched scenario or a required array is skipped only for terrain-adaptation outputs with an explicit manifest warning. It remains available to every existing analysis it can support.

No evaluator, simulator, checkpoint, or GPU process is invoked by this path.

## Evaluation Discovery and Duplicate Resolution

A gait-dynamics summary is eligible when it contains a non-empty `fixed_command_scenarios_gait_dynamics_metrics` mapping. Eligibility does not require `rebuttal_scenarios_only == true`.

For each `(env_name, run_name)`, candidate selection is deterministic:

1. reject non-zero action-delay candidates from default paper aggregation;
2. retain the highest parseable checkpoint;
3. at the same checkpoint, prefer a full evaluation over a purpose-built rebuttal-only evaluation;
4. if candidates remain tied with conflicting paths or payloads, fail with a diagnostic listing the candidates rather than silently choosing one.

The discovery manifest records checkpoint, action delay, evaluation scope, eligibility, selection status, and selection reason. `selected_runs.csv` continues to expose the selected paths. These rules apply without deleting or renaming any existing output.

## Outputs

The new terrain-adaptation layer writes:

- one row per run and scenario;
- one paired flat-to-uneven row per run;
- across-run summaries with mean, standard deviation, SEM, 95% CI, and run count;
- a discovery/data-quality manifest;
- the compact rebuttal figure in the requested export formats;
- a concise Markdown report containing metric definitions, values, warnings, and interpretation boundaries.

Existing rebuttal gait-dynamics CSVs, reports, and plot families retain their names and content.

## Error Handling and Compatibility

- Old full evaluations are supported by reconstructing contact from `contact_forces_array`; they do not need the newer direct acceleration or force-history arrays.
- Full evaluations created without `--rebuttal_scenarios_only` must feed the existing gait-dynamics aggregate when their summaries contain the metrics.
- Rebuttal-only evaluations without `fast_walk_x_uneven_terrain` continue to feed the existing dynamics aggregate but produce a terrain-analysis warning rather than a failure.
- Non-zero action-delay evaluations are visible in manifests but excluded from the default paper aggregate.
- Missing or malformed scenario metadata, inconsistent array lengths, or absent required arrays produce run-specific diagnostics.

## Verification Scope

Use a small focused test set rather than broad simulator tests:

1. a full zero-delay gait-dynamics summary is eligible and selected;
2. a purpose-built zero-delay summary remains eligible as fallback;
3. a full evaluation wins a same-checkpoint tie over rebuttal-only data, while non-zero-delay data is excluded;
4. synthetic flat periodic-trot and uneven modulated-contact arrays produce the expected diagonal occupancy and duration IQR values;
5. an existing archived full evaluation can be processed offline end-to-end without Isaac Lab imports or GPU use.

Verification must not launch `eval.py`, training, Isaac Sim, or any command that allocates VRAM.
