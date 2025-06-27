#!/usr/bin/env python3
"""
Analyse correlation between linear-velocity-tracking reward and terrain difficulty.

Supports:
  • wandb runs  (via public REST API)      --source wandb
  • TensorBoard event files (tfevents)     --source tfevents

Outputs:
  • combined_data.csv    tidy, long-format table
  • *.pdf plots          scatter, heatmap, ACF, etc.
  • summary.md           human-readable statistics
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import textwrap
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ────────────────────────────── VISUALISATION ────────────────────────────── #
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────── STATISTICS ──────────────────────────────── #
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.formula.api import mixedlm # mixed-effects model
from matplotlib.cm import get_cmap # colormap for time-encoded scatter

# For reading TensorBoard event files
from tensorboard.backend.event_processing import event_accumulator

# Optional import – only required when --source wandb
try:
    import wandb
    from wandb.apis.public import Run as WandbRun
except ModuleNotFoundError:
    wandb = None  # handled later

def load_wandb_runs(entity: str, project: str, keys: Tuple[str, str, str], tag_filter: List[str] | None = None, max_runs: int | None = None, warmup_iterations: int = 0) -> pd.DataFrame:
    """Download history for every run in a wandb project."""
    if wandb is None:
        raise RuntimeError("wandb Python package not installed.")
    api = wandb.Api(timeout=300)
    path = f"{entity}/{project}"
    runs: List[WandbRun] = api.runs(path)
    if tag_filter:
        runs = [r for r in runs if set(tag_filter).issubset(set(r.tags))]
    if max_runs:
        runs = runs[:max_runs]

    all_df: List[pd.DataFrame] = []
    for run in tqdm(runs, desc="Downloading runs"):
        try:
            #history = run.history(keys=list(keys), pandas=True, samples=25000)
            all_records = list(run.scan_history(keys=list(keys), page_size=10000))
            history   = pd.DataFrame(all_records)
        except wandb.CommError:
            print(f"⚠️  Skipping run {run.id} due to network error.", file=sys.stderr)
            continue
        history = history.rename(
            columns={
                keys[0]: "iteration",
                keys[1]: "terrain_difficulty",
                keys[2]: "velocity_reward",
            }
        ).dropna(subset=["terrain_difficulty", "velocity_reward"])
        history["run_id"] = run.id
        if warmup_iterations > 0:
            history = history[history["iteration"] >= warmup_iterations]
        all_df.append(history)
    if not all_df:
        raise RuntimeError("No runs retrieved from wandb.")
    return pd.concat(all_df, ignore_index=True)


def load_tfevents(event_dir: str, scalars_map: Tuple[str, str, str]) -> pd.DataFrame:
    """
    Read *.tfevents.* files under event_dir.

    scalars_map = (iteration_tag, terrain_tag, reward_tag)
    """
    event_paths = list(pathlib.Path(event_dir).rglob("events.out.tfevents.*"))
    if not event_paths:
        raise FileNotFoundError(f"No event files found in {event_dir}")
    all_df = []
    for path in tqdm(event_paths, desc="Reading tfevents"):
        ea = event_accumulator.EventAccumulator(str(path))
        try:
            ea.Reload()
        except Exception as exc:
            print(f"⚠️  Skipping {path}: {exc}", file=sys.stderr)
            continue

        iter_vals = ea.Scalars(scalars_map[0])
        terr_vals = ea.Scalars(scalars_map[1])
        rew_vals  = ea.Scalars(scalars_map[2])
        # assure equal lengths
        min_len = min(len(iter_vals), len(terr_vals), len(rew_vals))
        df = pd.DataFrame({
            "iteration": [v.step for v in iter_vals[:min_len]],
            "terrain_difficulty": [v.value for v in terr_vals[:min_len]],
            "velocity_reward": [v.value for v in rew_vals[:min_len]],
        })
        df["run_id"] = path.stem
        all_df.append(df)
    if not all_df:
        raise RuntimeError("No scalars extracted from event files.")
    return pd.concat(all_df, ignore_index=True)

def better_add_detrended_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append residual (detrended) columns for velocity reward and terrain difficulty.

    Removes the linear component explained by `iteration`, mitigating the
    spurious-correlation problem in sequential logs.  See Granger & Newbold (1974)
    for theory on spurious regression with trending variables.
    """
    X_iter = sm.add_constant(df["iteration"])
    resid_reward = sm.OLS(df["velocity_reward"], X_iter).fit().resid
    resid_diff   = sm.OLS(df["terrain_difficulty"], X_iter).fit().resid
    out          = df.copy()
    out["vel_reward_detr"]      = resid_reward
    out["terrain_difficulty_detr"] = resid_diff
    return out

def better_plot_scatter_time_colored(df: pd.DataFrame, out: pathlib.Path) -> None:
    """
    Scatter of raw Reward vs Difficulty, encoding `iteration`
    via a continuous colour map. Reveals Simpson-style reversals.
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(16, 16))

    # Use the current API for colormaps
    cmap = plt.colormaps["viridis"]
    norm = plt.Normalize(df["iteration"].min(), df["iteration"].max())

    # Plot scatter on the specific axes
    scatter = ax.scatter(
        df["velocity_reward"],
        df["terrain_difficulty"],
        c=cmap(norm(df["iteration"].values)),
        s=8,
        alpha=0.8
    )

    # Create a ScalarMappable for the colorbar (no need to set_array to [])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(df["iteration"].values)

    # Attach the colorbar to our axes
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("Iteration")

    # Labels, title, layout
    ax.set_xlabel("Linear Velocity Tracking Reward")
    ax.set_ylabel("Terrain Difficulty")
    ax.set_title("Reward vs Difficulty (colour = iteration)")
    fig.tight_layout()

    # Save and close
    fig.savefig(out / "scatter_reward_vs_difficulty_coloured_by_time.pdf", dpi=600)
    plt.close(fig)

def better_plot_scatter_detrended(df: pd.DataFrame, out: pathlib.Path) -> None:
    """
    Same as the basic scatter but uses detrended columns so that the
    slope shows the *partial* relationship (iteration removed).
    """
    plt.figure(figsize=(16, 16))
    sns.scatterplot(x="vel_reward_detr", y="terrain_difficulty_detr",
                    hue="run_id", data=df, s=8, alpha=0.8, palette="tab20")
    plt.xlabel("Velocity Reward (detrended)")
    plt.ylabel("Terrain Difficulty (detrended)")
    plt.title("Scatter of detrended variables")
    plt.tight_layout()
    plt.savefig(out / "scatter_detrended.pdf", dpi=600)
    plt.close()


def better_plot_variance_vs_reward_detrended(df: pd.DataFrame,
                                             out: pathlib.Path,
                                             num_bins: int = 10) -> pd.DataFrame:
    """
    Bin **detrended** reward and show variance of **detrended** difficulty.
    Neutralises heteroskedasticity induced by training phases.
    """
    reward_bins = pd.cut(df["vel_reward_detr"], bins=num_bins, include_lowest=True)
    var_per_bin = (df.groupby(reward_bins, observed=True)["terrain_difficulty_detr"]
                     .var()
                     .reset_index()
                     .rename(columns={"terrain_difficulty_detr": "difficulty_variance"}))
    var_per_bin["bin_mid"] = var_per_bin["vel_reward_detr"].apply(lambda x: x.mid)

    plt.figure(figsize=(16, 8))
    sns.barplot(x="bin_mid", y="difficulty_variance", data=var_per_bin)
    plt.xlabel("Detrended Reward bin centre")
    plt.ylabel("Variance of detrended Difficulty")
    plt.title("Difficulty variance across detrended reward bins")
    plt.tight_layout()
    plt.savefig(out / "variance_difficulty_over_detrended_reward_bins.pdf",
                dpi=600)
    plt.close()

    return var_per_bin

def better_first_difference_correlation(df: pd.DataFrame) -> dict:
    """
    Correlation on first differences removes any *deterministic* common trend.
    Returns Pearson & Spearman stats on Δ series.
    """
    diff_df = (df.sort_values(["run_id", "iteration"])
                 .groupby("run_id")[["velocity_reward", "terrain_difficulty"]]
                 .diff()
                 .dropna())
    pearson_r, pearson_p = stats.pearsonr(diff_df["velocity_reward"],
                                          diff_df["terrain_difficulty"])
    spearman_r, spearman_p = stats.spearmanr(diff_df["velocity_reward"],
                                             diff_df["terrain_difficulty"])
    return {"diff_pearson_r": pearson_r,
            "diff_pearson_p": pearson_p,
            "diff_spearman_r": spearman_r,
            "diff_spearman_p": spearman_p}

def better_plot_within_window_variance(df: pd.DataFrame,
                                       out: pathlib.Path,
                                       num_iter_bins: int = 20,
                                       num_reward_bins: int = 10) -> pd.DataFrame:
    """
    Two-way stratification: split ITERATION into equal-width bins and REWARD into equal-frequency bins.
    Compute variance of difficulty in each cell → heat-map.
    """
    iter_bins   = pd.cut(df["iteration"], bins=num_iter_bins, include_lowest=True)
    reward_bins = pd.qcut(df["velocity_reward"], q=num_reward_bins,
                          duplicates="drop")
    heat = (df.groupby([iter_bins, reward_bins], observed=True)
              ["terrain_difficulty"]
              .var()
              .unstack(level=0))  # rows: reward bins, cols: iteration bins

    plt.figure(figsize=(18, 10))
    sns.heatmap(heat, cmap="mako", cbar_kws={"label": "Var(difficulty)"})
    plt.xlabel("Iteration bin")
    plt.ylabel("Reward quantile")
    plt.title("Difficulty variance conditioned on time *and* reward")
    plt.tight_layout()
    plt.savefig(out / "heatmap_variance_iter_and_reward.pdf", dpi=600)
    plt.close()

    return heat

def plot_scatter(df: pd.DataFrame, out: pathlib.Path) -> None:
    plt.figure(figsize=(16, 16))
    sns.scatterplot(x="velocity_reward", y="terrain_difficulty", hue="run_id", data=df, s=8, alpha=0.8, palette="tab20")
    plt.title("Scatter: Terrain Difficulty vs Linear Velocity Reward")
    plt.savefig(out / "scatter_plot_terrain_difficulty_vs_reward.pdf", dpi=600)
    plt.close()

def plot_hexbin(df: pd.DataFrame, out: pathlib.Path) -> None:
    plt.figure(figsize=(16, 16))
    plt.hexbin(df["velocity_reward"], df["terrain_difficulty"], gridsize=50, bins="log")
    plt.xlabel("Linear Velocity Tracking Reward")
    plt.ylabel("Terrain difficulty")
    plt.title("Hexbin Density")
    cb = plt.colorbar()
    cb.set_label("log10(N)")
    plt.savefig(out / "hexbin_density.pdf", dpi=600)
    plt.close()

def plot_correlation_over_time(df: pd.DataFrame, out: pathlib.Path, num_bins: int = 20, min_points: int = 30) -> pd.DataFrame:
    """
    Slice training into `num_bins` equal-width iteration windows, compute the
    Pearson correlation inside each slice, and draw a line plot.

    Args:
        df: tidy data frame with columns iteration, velocity_reward, terrain_difficulty
        out: directory for the PDF
        num_bins: how many slices along the iteration axis
        min_points: skip a slice if it has fewer points than this
    """
    bins = pd.cut(df["iteration"], bins=num_bins, include_lowest=True)
    rows = []
    for interval, grp in df.groupby(bins, observed=True):
        if len(grp) < min_points:
            continue
        r, _ = stats.pearsonr(grp["velocity_reward"], grp["terrain_difficulty"])
        rows.append({"bin_mid": interval.mid, "pearson_r": r, "count": len(grp)})
    if not rows:  # nothing to plot
        return
        return pd.DataFrame(columns=["bin_mid", "pearson_r", "count"])
    corr_df = pd.DataFrame(rows).sort_values("bin_mid")

    plt.figure(figsize=(16, 8))
    sns.lineplot(x="bin_mid", y="pearson_r", data=corr_df, marker="o")
    plt.xlabel("Iteration (bin centre)")
    plt.ylabel("Pearson r  (reward → difficulty)")
    plt.title("Correlation between reward and difficulty over training time")
    plt.tight_layout()
    plt.savefig(out / "correlation_over_time.pdf", dpi=600)
    plt.close()

    return corr_df

def better_run_mixed_effects(df: pd.DataFrame):
    """
    Fit Difficulty ~ Reward + Iteration  with random intercept per run_id.
    Helps disentangle within-run vs between-run effects.
    """
    # MixedLM wants endog/ exog arrays + groups
    md = mixedlm("terrain_difficulty ~ velocity_reward + iteration", df, groups=df["run_id"])
    m  = md.fit(reml=False)
    return m

def better_plot_diff_scatter(df: pd.DataFrame, out: pathlib.Path) -> None:
    """
    Visual check for first-difference relationship.
    """
    diff_df = (df.sort_values(["run_id", "iteration"])
                 .groupby("run_id")[["velocity_reward", "terrain_difficulty"]]
                 .diff()
                 .dropna())
    plt.figure(figsize=(16, 16))
    sns.scatterplot(x="velocity_reward", y="terrain_difficulty",
                    data=diff_df, s=8, alpha=0.8)
    plt.xlabel("Δ Velocity Reward")
    plt.ylabel("Δ Terrain Difficulty")
    plt.title("First-difference scatter")
    plt.tight_layout()
    plt.savefig(out / "scatter_first_difference.pdf", dpi=600)
    plt.close()

def plot_variance_vs_reward(df: pd.DataFrame, out: pathlib.Path, num_bins: int = 10) -> None:
    """
    Bin velocity_reward and show the variance of terrain_difficulty per bin.
    """
    reward_bins = pd.cut(df["velocity_reward"], bins=num_bins, include_lowest=True)
    var_per_bin = (
        df.groupby(reward_bins, observed=True)["terrain_difficulty"]
        .var()
        .reset_index()
        .rename(columns={"terrain_difficulty": "difficulty_variance"})
    )
    var_per_bin["bin_mid"] = var_per_bin["velocity_reward"].apply(lambda x: x.mid)

    plt.figure(figsize=(16, 8))
    sns.barplot(
        x="bin_mid",
        y="difficulty_variance",
        data=var_per_bin,
    )
    plt.xlabel("Velocity reward bin centre")
    plt.ylabel("Variance of terrain difficulty")
    plt.title("Difficulty variance across reward bins")
    plt.tight_layout()
    plt.savefig(out / "variance_difficulty_over_reward_bins.pdf", dpi=600)
    plt.close()

def partial_correlation_reward_difficulty(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Pearson correlation between reward and difficulty *after removing* their
    shared linear trend with iteration (controls for the 'time' confound).

    Returns
    -------
    r_partial, p_value
    """
    # Remove iteration trend from reward
    X_iter = sm.add_constant(df["iteration"])
    resid_reward = sm.OLS(df["velocity_reward"], X_iter).fit().resid

    # Remove iteration trend from difficulty
    resid_diff = sm.OLS(df["terrain_difficulty"], X_iter).fit().resid

    r_partial, p_partial = stats.pearsonr(resid_reward, resid_diff)
    # Spearman “partial” (monotonic association controlling for iteration)
    rho_partial, p_spear  = stats.spearmanr(resid_reward, resid_diff)
    
    return float(r_partial), float(p_partial), float(rho_partial), float(p_spear)


def plot_local_corr_by_reward(
    df: pd.DataFrame,
    out: pathlib.Path,
    num_bins: int = 10,
    min_points: int = 30,
) -> pd.DataFrame:
    """
    Slice the data into equal-count reward bins (deciles by default) and plot the
    correlation *inside* each slice.  This shows how well reward predicts
    difficulty when reward is roughly ‘held constant’.

    Skips a bin if it has fewer than `min_points` samples.
    """
    reward_bins = pd.qcut(df["velocity_reward"], q=num_bins, duplicates="drop")
    rows = []
    for interval, grp in df.groupby(reward_bins, observed=True):
        if len(grp) < min_points:
            continue
        r, _ = stats.pearsonr(grp["velocity_reward"], grp["terrain_difficulty"])
        rows.append(
            {
                "bin_mid": interval.mid,
                "pearson_r": r,
                "count": len(grp),
            }
        )
    if not rows:  # nothing to draw
        return pd.DataFrame(columns=["bin_mid", "pearson_r", "count"])
    corr_df = pd.DataFrame(rows).sort_values("bin_mid")

    plt.figure(figsize=(16, 8))
    sns.lineplot(x="bin_mid", y="pearson_r", data=corr_df, marker="o")
    plt.xlabel("Velocity reward (bin centre)")
    plt.ylabel("Pearson r  (within bin)")
    plt.title("Local reward–difficulty correlation across reward deciles")
    plt.tight_layout()
    plt.savefig(out / "local_corr_by_reward.pdf", dpi=600)
    plt.close()

    return corr_df

def correlations(df: pd.DataFrame) -> dict:
    pearson_r, pearson_p = stats.pearsonr(df["terrain_difficulty"], df["velocity_reward"])
    spearman_r, spearman_p = stats.spearmanr(df["terrain_difficulty"], df["velocity_reward"])
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }

def run_linear_regression(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResults:
    X = sm.add_constant(df["velocity_reward"]) # Predictor
    model = sm.OLS(df["terrain_difficulty"], X) # Reponse / predicted
    results = model.fit()
    return results

def optimal_polynomial_regression(df: pd.DataFrame, max_degree: int = 25, cv_splits: int = 5) -> Tuple[int, float]:
    """
    Returns best_degree, cv_score using R^2 (higher is better).
    """
    X = df[["velocity_reward"]].to_numpy()
    y = df["terrain_difficulty"].to_numpy()
    best_degree, best_score = 1, -np.inf
    for d in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=d, include_bias=False)
        Xd = poly.fit_transform(X)
        model = LinearRegression()
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, Xd, y, cv=cv, scoring="r2")
        score = np.mean(scores)
        if score > best_score:
            best_degree, best_score = d, score
    # Fit final model with best degree for residuals analysis
    poly = PolynomialFeatures(degree=best_degree, include_bias=False)
    Xd = poly.fit_transform(X)
    model = LinearRegression().fit(Xd, y)
    df["poly_pred"] = model.predict(Xd)
    return best_degree, best_score

def auto_correlation(df: pd.DataFrame, out: pathlib.Path) -> float:
    """
    Compute lag-1 autocorrelation of residuals and produce ACF plot.
    """
    # Fit simple linear regression to get residuals
    results = run_linear_regression(df)
    residuals = results.resid
    lag1_corr = residuals.autocorr(lag=1)

    # ACF plot
    from statsmodels.graphics.tsaplots import plot_acf
    plt.figure(figsize=(16, 14))
    plot_acf(residuals, lags=40, title="Residual Autocorrelation (ACF)")
    plt.tight_layout()
    plt.savefig(out / "acf_residuals.pdf", dpi=600)
    plt.close()
    return lag1_corr

def durbin_watson_stat(df: pd.DataFrame) -> float:
    results = run_linear_regression(df)
    return durbin_watson(results.resid)

def mutual_information_score(df: pd.DataFrame) -> float:
    """
    Mutual Information between difficulty and reward.
    (Uses k-nearest neighbours estimator from scikit-learn.)
    """
    X = df[["terrain_difficulty"]].to_numpy()
    y = df["velocity_reward"].to_numpy()
    mi = mutual_info_regression(X, y, random_state=42)
    return float(mi[0])

def perform_all_analyses(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "combined_data.csv", index=False)

    # 1 ─ Visual EDA
    plot_scatter(df, output_dir)
    plot_hexbin(df, output_dir)
    plot_variance_vs_reward(df, output_dir, num_bins=10)
    corr_over_time_df = plot_correlation_over_time(df, output_dir, num_bins=20)
    local_corr_by_reward_df = plot_local_corr_by_reward(df, output_dir, num_bins=10)

    better_plot_scatter_time_colored(df, output_dir)
    df_detr = better_add_detrended_columns(df)
    better_plot_scatter_detrended(df_detr, output_dir)
    var_detr_df = better_plot_variance_vs_reward_detrended(df_detr, output_dir)
    within_window_df = better_plot_within_window_variance(df, output_dir)

    # 2 ─ Correlations
    corr_results = correlations(df)

    # 3 ─ Linear regression
    lin_results = run_linear_regression(df)

    # 4 ─ Polynomial regression with CV
    best_deg, best_cv_r2 = optimal_polynomial_regression(df)

    # 5 ─ Autocorrelation + Durbin-Watson
    lag1 = auto_correlation(df, output_dir)
    dw_stat = durbin_watson_stat(df)

    # 6 ─ Mutual Information
    mi = mutual_information_score(df)

    r_partial, p_partial, rho_partial, p_spear = partial_correlation_reward_difficulty(df)
    
    diff_corr = better_first_difference_correlation(df)
    mix_model = better_run_mixed_effects(df)

    # ─── build the main markdown with all your existing metrics ──────────
    base_md = textwrap.dedent(f"""
    # Analysis Summary

    **Rows analysed:** {len(df):,}  
    **Distinct runs:** {df['run_id'].nunique()}

    ## Correlation Coefficients
    | Metric | ρ / r | p-value |
    |--------|-------|---------|
    | Pearson | {corr_results['pearson_r']:.4f} | {corr_results['pearson_p']:.2e} |
    | Spearman| {corr_results['spearman_r']:.4f} | {corr_results['spearman_p']:.2e} |
    | Partial Pearson (iter-controlled) | {r_partial:.4f} | {p_partial:.2e} |
    | Partial Spearman (iter-controlled) | {rho_partial:.4f} | {p_spear:.2e} |

    Linear Regression (Difficulty ~ β₀ + β₁·Reward)  
    β₀ = {lin_results.params['const']:.4f} (± {lin_results.bse['const']:.4f})  
    β₁ = {lin_results.params['velocity_reward']:.4f} (± {lin_results.bse['velocity_reward']:.4f})  
    R² = {lin_results.rsquared:.4f}

    ## Polynomial Regression  
    Optimal degree = **{best_deg}** (cross-validated R² = {best_cv_r2:.4f})

    ## Autocorrelation Diagnostics  
    Lag-1 residual autocorrelation = {lag1:.4f}  
    Durbin-Watson statistic = {dw_stat:.4f}

    ## Mutual Information  
    MI(difficulty; reward) = {mi:.4f} nats

    ### First-difference correlation  
    Pearson Δr = {diff_corr['diff_pearson_r']:.4f} (p={diff_corr['diff_pearson_p']:.2e})  
    Spearman Δρ = {diff_corr['diff_spearman_r']:.4f} (p={diff_corr['diff_spearman_p']:.2e})

    ### Mixed-effects  
    {mix_model.summary().as_text()}
    """).strip()

    # ─── Over-Time table ─────────────────────────────────────────────────────
    ot_lines = [
        "## Correlation Over Time Data",
        "| Iteration Bin Centre | Pearson r | Count |",
        "|---|---|---|",
    ]
    for _, row in corr_over_time_df.iterrows():
        ot_lines.append(f"| {row['bin_mid']:.2f} | {row['pearson_r']:.4f} | {int(row['count'])} |")
    ot_md = "\n".join(ot_lines)

    # ─── Over-Reward table ───────────────────────────────────────────────────
    or_lines = [
        "## Local Correlation by Reward Data",
        "| Reward Bin Centre | Pearson r | Count |",
        "|---|---|---|",
    ]
    for _, row in local_corr_by_reward_df.iterrows():
        or_lines.append(f"| {row['bin_mid']:.2f} | {row['pearson_r']:.4f} | {int(row['count'])} |")
    or_md = "\n".join(or_lines)

    # ─── Detrended Variance table ───────────────────────────────────────────
    vd_lines = [
        "## Detrended Reward vs Difficulty Variance",
        "| Reward bin centre | Difficulty variance |",
        "|---|---|",
    ]
    for _, row in var_detr_df.iterrows():
        vd_lines.append(f"| {row['bin_mid']:.2f} | {row['difficulty_variance']:.4f} |")
    vd_md = "\n".join(vd_lines)

    # ─── Within-Window Variance table ───────────────────────────────────────
    ww_lines = [
        "## Variance conditioned on Time and Reward",
        "| Iteration bin centre | Reward quantile centre | Difficulty variance |",
        "|---|---|---|",
    ]
    for reward_interval, series in within_window_df.iterrows():
        for iter_interval, val in series.items():
            ww_lines.append(f"| {iter_interval.mid:.2f} | {reward_interval.mid:.2f} | {val:.4f} |")
    ww_md = "\n".join(ww_lines)

    # ─── combine all sections ────────────────────────────────────────────────
    md = "\n\n".join([base_md, ot_md, or_md, vd_md, ww_md]).strip()

    print(md)
    (output_dir / "summary.md").write_text(md, encoding="utf-8")

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse relation between terrain difficulty and reward.")
    parser.add_argument("--source", choices=["wandb", "tfevents"], required=True)
    parser.add_argument("--entity", help="wandb team/user", default=None)
    parser.add_argument("--project", help="wandb project name", default=None)
    parser.add_argument("--tag", action="append", help="require run to contain this wandb tag; may repeat")
    parser.add_argument("--max_runs", type=int, help="limit number of wandb runs (useful for quick test)")
    parser.add_argument("--event_dir", help="Directory containing *.tfevents* files")
    parser.add_argument("--output_dir", required=True, help="Directory to write plots, CSV and summary")
    parser.add_argument("--keys", nargs=3, metavar=("ITER", "DIFF", "REWARD"),
                        default=("_step", "Curriculum/terrain_levels",
                                 "Episode_Reward/track_lin_vel_xy_exp"),
                        help=textwrap.dedent("""\
                        Scalar keys or TensorBoard tags for:
                          1) training iteration step
                          2) terrain difficulty
                          3) velocity tracking reward"""))
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Ignore the first N training iterations (per run) in every analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    out_dir = pathlib.Path(args.output_dir)

    if args.source == "wandb":
        if not (args.entity and args.project):
            sys.exit("⚠️  --entity and --project are required for wandb source.")
        df = load_wandb_runs(entity=args.entity, project=args.project, keys=tuple(args.keys), tag_filter=args.tag, max_runs=args.max_runs, warmup_iterations=args.warmup_iterations)
    else:  # tfevents
        if not args.event_dir:
            sys.exit("⚠️  --event_dir is required for tfevents source.")
        df = load_tfevents(event_dir=args.event_dir, scalars_map=tuple(args.keys), warmup_iterations=args.warmup_iterations)

    perform_all_analyses(df, out_dir)
    print(f"✅ Finished. Results written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
