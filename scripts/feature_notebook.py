"""Canonical feature-research notebook narrative and evidence displays."""


def cells(setup: str) -> list[tuple[str, str]]:
    return [
        (
            "md",
            "# Commodity forecasting | Domain feature research\n\n**Feature gate: OPEN.** A feature-rich model has to beat credible simple controls. This notebook preserves the earlier target-aware study, then develops 13 domain families and tests pooled residual forecasts with nested temporal calibration.\n\nAll comparisons use the corrected **535-date** development window. The original reservation contains a five-date permanent boundary buffer and **247 untouched final-test dates**. Earlier experiments remain intact. The domain section below presents the latest executed evidence.",
        ),
        ("code", setup),
        (
            "md",
            "## Correct the benchmark and the time boundary\n\nThe first shared-screen Ridge experiment omitted a strong control: each target's historical training mean. We now include frozen training means and expanding means of labels that have actually been released.\n\nA five-date terminal embargo removes development targets whose outcomes reach into the reserved interval. The initial models are rescored on these same 535 dates without refitting. Their old 540-date scores are historical records, not current comparisons. Because those overlapping outcomes were already inspected, origins 1709–1713 remain a permanent buffer; only origins 1714–1960 enter the eventual final test.",
        ),
        (
            "code",
            """comparison = pd.DataFrame(study["comparison"])
legacy = pd.DataFrame(study["initial_variants_rescored_on_common_dates"])
controls = comparison.loc[comparison.variant.isin(["historical_mean", "released_mean"]), ["variant", "official_metric", "fold_metrics"]]
display(controls.round(4))
display(legacy[["variant", "official_metric"]].sort_values("official_metric", ascending=False).round(4))
display(Markdown(f"**Same-window comparison:** {study['validation_dates']} validation dates; {study['terminal_embargo_dates']}-date terminal embargo; all 30 original fold experiments preserved."))""",
        ),
        (
            "md",
            "## Search broadly, route candidates to the prediction\n\nThe original nine families cover returns, momentum, volatility, reversion, liquidity, OHLC, missingness, market relationships, and target-pair spreads. Three added families test longer risk regimes, interactions, and released-label history.\n\nRegimes include 126/252-date trends, risk ratios, tail shape, drawdown, autocorrelation, and trailing percentiles. Interactions combine momentum with volatility, relative returns, and reversion. Historical labels are shifted by exactly horizon + 1 before expanding, rolling, or exponential aggregation. Future-value mutation tests cover every new family.",
        ),
        (
            "code",
            """counts = pd.Series(study["family_counts"]).sort_values().rename_axis("Family").reset_index(name="Candidates")
counts["Study"] = np.where(counts.Family.isin(["regime", "interactions", "release_history"]), "Added in follow-up", "Reused from initial study")
fig = px.bar(counts, x="Candidates", y="Family", color="Study", orientation="h", text="Candidates",
             color_discrete_map={"Reused from initial study": "#1F6C99", "Added in follow-up": "#27A394"},
             title=f"{study['candidate_count']:,} candidates; {study['new_candidate_count']:,} added candidates")
fig.update_traces(textposition="outside")
fig.update_layout(legend={"orientation": "h", "y": -0.14})
show_figure(fig, root, "feature_families", 690)""",
        ),
        (
            "md",
            "## Hold model complexity fixed while changing representation\n\nEach output can retain at most **24 features**, with training-only missingness, constant, duplicate, redundancy, and relevance checks. Stable screening requires the feature/target correlation to have the same sign in both training halves. Validation labels never enter the screener.\n\nAligned models see their own assets, exact pair, market context, and own released-label history. Structural models forecast asset returns first, then subtract components to preserve pair identities. Ridge alpha remains 100. A small fixed tree model checks nonlinear structure; automatic random early stopping is disabled.\n\nThe initial study used a shared 96-feature budget. Feature-attribution claims use matched controls inside this new 24-feature study.",
        ),
        (
            "code",
            """main = comparison.loc[~comparison.variant.str.startswith("drop_")].sort_values("official_metric")
fig = px.bar(main, x="official_metric", y="variant", orientation="h", text_auto=".3f",
             title="Do engineered representations beat credible simple controls?")
fig.update_traces(marker_color="#27A394", textposition="outside")
fig.update_xaxes(range=[min(0, float(main.official_metric.min())) - 0.04, max(0, float(main.official_metric.max())) + 0.05])
fig.update_layout(xaxis_title="Pooled official metric · 535 development dates", yaxis_title="Declared experiment")
show_figure(fig, root, "ablation_scores", 690)
display(main[["variant", "official_metric", "mean_daily_rank_correlation", "fold_metrics"]].sort_values("official_metric", ascending=False).round(4))""",
        ),
        (
            "md",
            "## Matched comparisons separate feature value from other changes\n\nThe official metric is mean daily cross-sectional Spearman correlation divided by its population standard deviation. Feature value is a difference from a matched reference, not merely a high standalone score.\n\nThe chart shows simultaneous 95% bounds over the study's declared comparisons using 20-date blocks and 2,000 paired resamples within folds. Reports also include conditional intervals and 10-/40-date sensitivity. These bounds do not include fresh feature searches, refitting uncertainty, or all prior adaptive decisions.",
        ),
        (
            "code",
            """bounds = pd.DataFrame(study["paired_intervals"])
labels = {
    ("target_all", "target_reference"): "Global families vs. returns",
    ("target_stable", "target_all"): "Stable vs. one-period screening",
    ("aligned_original", "aligned_reference"): "Aligned families vs. returns",
    ("aligned_extended", "aligned_original"): "Extended vs. original families",
    ("structural_extended", "structural_reference"): "Structural families vs. returns",
    ("hist_extended", "hist_reference"): "Trees: extended vs. returns",
}
matched = bounds.loc[(bounds.block_dates == 20) & bounds.apply(lambda r: (r.variant, r.reference) in labels, axis=1)].copy()
matched["Comparison"] = matched.apply(lambda r: labels[(r.variant, r.reference)], axis=1)
lo = matched.simultaneous_95_interval.map(lambda a: a[0])
hi = matched.simultaneous_95_interval.map(lambda a: a[1])
fig = go.Figure(go.Scatter(x=matched.delta, y=matched.Comparison, mode="markers", marker={"size": 11, "color": "#1F6C99"},
    error_x={"type": "data", "symmetric": False, "array": hi - matched.delta, "arrayminus": matched.delta - lo}))
fig.add_vline(x=0, line_dash="dash", line_color="#9EAFBF")
fig.update_layout(title="Which matched feature gains survive uncertainty?", xaxis_title="Metric change · simultaneous 95% bounds")
show_figure(fig, root, "ablation_uncertainty", 560)
display(matched[["Comparison", "delta", "conditional_95_interval", "simultaneous_95_interval"]])""",
        ),
        (
            "md",
            "## Conditional ablations: remove one family and refit\n\nEvery non-reference family is removed from the extended aligned representation while the same screening rule and estimator are refit. Positive deltas favor the full representation; negative deltas favor removal. An ablation measures the whole pipeline's dependence on a family, including replacement features selected after removal.",
        ),
        (
            "code",
            """drops = bounds.loc[(bounds.block_dates == 20) & (bounds.variant == "aligned_extended") & bounds.reference.str.startswith("drop_")].sort_values("delta").copy()
drops["Family"] = drops.reference.str.removeprefix("drop_")
lo = drops.conditional_95_interval.map(lambda a: a[0])
hi = drops.conditional_95_interval.map(lambda a: a[1])
fig = go.Figure(go.Scatter(x=drops.delta, y=drops.Family, mode="markers", marker={"size": 11, "color": "#27A394"},
    error_x={"type": "data", "symmetric": False, "array": hi - drops.delta, "arrayminus": drops.delta - lo}))
fig.add_vline(x=0, line_dash="dash", line_color="#9EAFBF")
fig.update_layout(title="Does each family help when the others remain?", xaxis_title="Full minus drop-family metric · conditional 95% interval")
show_figure(fig, root, "conditional_ablations", 650)""",
        ),
        (
            "md",
            "## Stability across time and horizons\n\nAll variants use the same expanding folds. The final validation block is five dates shorter to enforce the holdout boundary. A pooled gain that reverses in a period needs additional evidence. Horizon diagnostics report eligible dates separately; they do not replace the primary metric across all observed targets.",
        ),
        (
            "code",
            """focus = ["historical_mean", "released_mean", "target_stable", "aligned_extended", "structural_extended", "hist_extended"]
rows = [{"Variant": r["variant"], "Fold": f"Fold {r['fold'] + 1}", "Metric": r["official_metric"]}
        for r in study["results"] if r["variant"] in focus]
matrix = pd.DataFrame(rows).pivot(index="Variant", columns="Fold", values="Metric")
fig = px.imshow(matrix, text_auto=".3f", color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect="auto",
                title="A pooled score must also survive different periods", labels={"color": "Metric"})
show_figure(fig, root, "fold_stability", 520)
best_name = study["comparison"][0]["variant"]
horizon_rows = [{"Fold": r["fold"] + 1, "Horizon": h, **metrics} for r in study["results"] if r["variant"] == best_name for h, metrics in r["horizon_metrics"].items()]
display(Markdown(f"**Horizon diagnostics for the earlier study point-estimate leader:** {best_name}"))
display(pd.DataFrame(horizon_rows).round(4))""",
        ),
        (
            "md",
            "## Model reliance and selection stability\n\nJoint block permutation disrupts a family's date alignment while preserving within-date structure across targets. The model stays frozen. The chart averages changes in **fold-level** scores across five perturbations per fold. It is not a pooled official score or a causal importance estimate. Correlated groups can substitute for one another, and perturbations can leave the observed data distribution.",
        ),
        (
            "code",
            """permutation_rows = []
for fold, groups in study["group_permutation"].items():
    for family, stats in groups.items():
        permutation_rows.extend({"Fold": fold, "Family": family, "Metric drop": value} for value in stats["metric_drop_repetitions"])
importance = pd.DataFrame(permutation_rows).groupby("Family")["Metric drop"].mean().sort_values().reset_index()
fig = px.bar(importance, x="Metric drop", y="Family", orientation="h", title="Which families does the aligned model rely on?")
fig.update_traces(marker_color="#1F6C99")
fig.add_vline(x=0, line_dash="dash", line_color="#9EAFBF")
fig.update_layout(xaxis_title="Mean fold-metric decrease after joint block permutation")
show_figure(fig, root, "group_permutation", 620)""",
        ),
        (
            "code",
            """stability = pd.DataFrame([{"Variant": r["variant"], "Median Jaccard": r["selection_stability"]["median_jaccard"]}
    for r in study["comparison"] if r["variant"] in ["target_all", "target_stable", "aligned_original", "aligned_extended", "structural_extended"]])
fig = px.bar(stability, x="Median Jaccard", y="Variant", orientation="h", text_auto=".2f", title="Do adjacent folds retain the same features?")
fig.update_traces(marker_color="#27A394", textposition="outside")
fig.update_xaxes(range=[0, 1])
show_figure(fig, root, "selection_stability", 470)
last = next(r for r in study["results"] if r["variant"] == "aligned_extended" and r["fold"] == 2)
screening = last["screening"]
display(pd.Series({key: screening[key] for key in ["candidate_output_assignments", "retained_output_assignments", "rejected_output_assignments", "unique_retained_features", "retained_per_output_min", "retained_per_output_median", "retained_per_output_max"]}, name="Last-fold aligned screen").to_frame())
display(pd.Series(screening["rejection_reasons"], name="Rejected assignments").sort_values(ascending=False).to_frame())""",
        ),
        (
            "md",
            "Counts summed over target-specific screens are **feature/output assignments**. The same generated feature can enter several models. Unique candidates, unique retained features, and assignments are different quantities. A budget rejection does not prove absence of predictive signal.",
        ),
        (
            "code",
            """best = study["comparison"][0]
mean_control = next(r for r in study["comparison"] if r["variant"] == "historical_mean")
display(Markdown(f"**Earlier study leader:** {best['variant']} at **{best['official_metric']:.4f}**. Frozen training means score **{mean_control['official_metric']:.4f}** on the same dates.\\n\\nCompleted **{study['experiments_completed']} fold/representation experiments** using **{study['candidate_count']:,} candidates**, while preserving all 30 initial experiments. Feature gate: **open**."))""",
        ),
        (
            "md",
            "## What the earlier target-aware study established\n\nFrozen training means lead at **0.2177**; aligned return-only Ridge is close at **0.2087**. Its difference from the mean control is **−0.0091**, with a conditional 95% interval of **[−0.0580, +0.0366]**. This is evidence to keep a strong prior benchmark in every future experiment.\n\nThe three added families improve aligned Ridge by only **+0.0077**, with interval **[−0.0793, +0.0943]**. Released-label history has the largest positive conditional ablation estimate (**+0.0299**), but its interval includes zero. Pair removal helps under a conditional interval, but does not survive simultaneous comparison bounds. No positive matched feature gain is established by the simultaneous bounds.\n\nRicher aligned models reverse sign in the middle fold, and the extended structural model reverses in the last fold. These failures argue for investigating target priors, scale, and residual structure before broad tuning. More generated features are not evidence of more usable signal.",
        ),
        (
            "md",
            "## Transition to the domain study\n\nKeep the feature gate open. This study addresses credible mean controls, output-specific screening, economic routing, structural forecasts, longer risk regimes, released-label histories, conditional ablations, nonlinear controls, and block-size sensitivity. It does not establish diminishing returns for every plausible feature avenue.\n\nNext decisions should follow the measured weaknesses: distinguish cross-sectional prior information from useful time-varying features, assess target/horizon normalization and robust residual representations, and test surviving interactions with nested temporal selection. External data require a defensible calendar mapping, source vintages, and appropriate rights; date IDs must not receive guessed calendar dates. Final optimization and reserved-holdout evaluation remain gated.\n\n[Research protocol](../docs/research-protocol.md) · [Aggregate evidence](../reports/feature_study.json) · [Official metric](https://www.kaggle.com/code/metric/mitsui-co-commodity-prediction-metric).",
        ),
    ]
