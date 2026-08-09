# Evaluation Selection and Terrain-Analysis Compatibility Design

## Objective

Keep the existing `analyze_run_range.py` metric and plotting pipeline working while adding terrain-adaptation analysis from archived evaluations. Evaluation discovery must choose candidates predictably, retain delayed evaluations instead of silently dropping them, and explain all ambiguous or nonstandard selections at the end of the script.

## Evaluation ranking

Discovery records every `metrics_summary.json` candidate for each `(env_name, run_name)` pair. Each candidate includes:

- parsed checkpoint number;
- number of fixed-command scenarios actually recorded;
- action-delay steps;
- evaluation scope (`full`, `rebuttal_only`, or `single_scenario`);
- availability of gait-dynamics payloads; and
- availability of the saved arrays and matched scenarios required by terrain analysis.

All consumers use the same deterministic ranking:

1. highest checkpoint;
2. greatest number of available fixed-command scenarios;
3. zero action delay;
4. deterministic path order.

No candidate is excluded solely because it is delayed, rebuttal-only, or restricted to one scenario. A higher-checkpoint restricted evaluation therefore outranks a lower-checkpoint full evaluation, as explicitly requested.

## Primary selection and capable fallbacks

The highest-ranked candidate is the primary evaluation for the existing general metrics. Gait-dynamics and terrain-adaptation analysis reuse the primary evaluation when it contains their required payloads.

If the primary evaluation cannot support an optional analysis, that analysis uses the highest-ranked capable candidate according to the same ranking. This preserves existing gait outputs and produces terrain outputs whenever suitable archived data exist without silently mixing sources. Every fallback is recorded in the final audit summary and discovery manifest.

The default `eval.py` invocation already contains the full fixed-command scenario set. It also computes gait-dynamics payloads for every scenario listed in `REBUTTAL_DYNAMICS_SCENARIOS`, so a current default evaluation normally supports general, gait, and terrain analysis from one source.

## Compatibility

The change does not modify training, simulation, `eval.py`, or existing metric formulas. Existing output filenames and metric families remain available. Terrain analysis runs after the existing outputs and adds its own CSV, Markdown, manifest, and figure artifacts.

The step-height comparison continues using the command-matched pair `walk_x_flat_terrain_1.0mps` and `fast_walk_x_uneven_terrain`. Terrain contact uses the current strict force threshold of greater than 1 N. The observed 0.08% difference from older saved contact booleans is accepted.

The uncommitted addition of `fast_walk_x_uneven_terrain` to the main checkout's `REBUTTAL_DYNAMICS_SCENARIOS` must be preserved. Aggregate scenario counts, expected-row checks, and report prose must be derived dynamically so four-scenario archives and new five-scenario evaluations are handled explicitly rather than described incorrectly.

## Final audit warnings

`analyze_run_range.py` accumulates compatibility findings and prints one consolidated audit section near script completion. It reports:

- each run with multiple evaluation candidates;
- the selected path and ranking reason;
- every selected primary or fallback evaluation with nonzero action delay;
- every metric family that used a fallback evaluation; and
- every terrain-analysis source skipped for missing scenarios, arrays, or incomplete data.

Delayed evaluations remain usable. The warning is informational and prominent; delay does not make a candidate ineligible.

## Failure behavior

Missing optional gait or terrain data must not prevent the existing analysis pipeline from finishing. The affected optional output is skipped or uses a capable fallback, and the reason is included in the manifest and final audit summary. Invalid ambiguity after all ranking keys is resolved deterministically by path and reported rather than raising solely because multiple evals exist.

Malformed mandatory summary data continues to fail with a clear error where the existing pipeline cannot safely interpret it.

## Validation

Tests will establish the ranking and warning behavior before implementation. They cover:

- checkpoint priority over scenario count;
- scenario-count priority at equal checkpoint;
- zero-delay priority at equal checkpoint and scenario count;
- deterministic resolution and duplicate reporting;
- delayed candidates remaining eligible and producing final warnings;
- gait and terrain capable fallbacks;
- default full evaluations containing gait payloads;
- rebuttal-only and single-scenario compatibility;
- missing optional terrain arrays not breaking existing outputs; and
- dynamic handling of four- and five-scenario gait summaries.

After focused tests and compilation checks, the LP and LEP `analyze_run_range.py` commands will be run against archived data only. No simulator evaluation or training is required for this validation.
