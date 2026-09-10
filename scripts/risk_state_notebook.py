"""Show risk-state feature attribution and its limits in the canonical notebook."""


def cells() -> list[tuple[str, str]]:
    return [
        (
            "md",
            "## Do historical priors depend on the risk state?\n\nThe compact study left a concrete question: does a historical mean have different short-horizon relevance when risk is rising, falling, or unusually high? [Moreira and Muir (2017)](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12513) find that expected returns need not change proportionally with volatility in their studied portfolios. That motivates this conditional-feature hypothesis; it does not validate these formulas or turn this ranking metric into trading performance.\n\n**34 new templates:** six bounded risk states, four risk-normalized historical priors, and 24 state-by-prior products. States describe 63/252 and 126/252 risk ratios, five- and 21-date risk changes, 21-date risk instability, and a trailing 126-date percentile. Products keep their state and prior main effects. All inputs descend from the frozen, horizon+1-delayed prior features. Rolling windows end at the current prediction origin. The delay stress adds another date to those inputs before recomputing every derived feature.\n\n**Eight variants × three purged folds = 24 fits.** Frozen tree settings and matched admission controls isolate feature changes. Existing checkpoints are reused. This is exploratory research following the earlier 233 domain comparisons; the final 247 origins remain untouched.",
        ),
        (
            "code",
            """from scripts.notebook_support import checked_risk_state
risk_state = checked_risk_state(root)
display(Markdown(f"**Verified risk-state lineage:** `{risk_state['lineage'][:16]}` · {risk_state['new_fitted_models']} fitted models · {risk_state['checkpoint_count']} sealed stages · replay error {risk_state['maximum_prediction_replay_error']:.1e}"))
rows = [{"Variant": n, "Metric": s["official_metric"], **{f"Fold {i+1}": v for i,v in enumerate(s["fold_scores"])}} for n,s in risk_state["summaries"].items()]
state_scores = pd.DataFrame(rows).set_index("Variant")
display(state_scores.round(6))
fig = px.bar(state_scores.sort_values("Metric").reset_index(), x="Metric", y="Variant", orientation="h", text_auto=".3f", title="Risk-state hypotheses against frozen feature controls")
show_figure(fig, root, "risk_state_scores", 740)
counts = pd.DataFrame([{"Variant": r["variant"], "Fold": r["fold"]+1, "Candidates": r["selection"]["candidate_templates"], "Retained": r["selection"]["retained_templates"], "Rejected": r["selection"]["rejected_templates"], "Reasons": r["selection"]["rejection_reasons"], "Retained products": sum(n.startswith("state_prior_product__") for n in r["selection"]["selected_names"])} for r in risk_state["results"]])
display(counts)""",
        ),
        (
            "md",
            "### Attribute gains to features and inspect temporal failures\n\nStates, normalized priors, and their combination are compared with the admitted tail-risk control. Adding the 24 products is compared with the same state and prior main effects. Removing tail risk, adding freshness, screening to 64 columns, and adding an information delay test separate explanations. The same tree settings apply throughout. Per-fold results describe stability; they do not authorize choosing a special model for the weakest fold.",
        ),
        (
            "code",
            """folds_only = state_scores[["Fold 1","Fold 2","Fold 3"]]
fold_deltas = folds_only.sub(folds_only.loc["admitted_tail"],axis=1)
fig = px.imshow(fold_deltas, color_continuous_scale="RdBu", color_continuous_midpoint=0, text_auto=".3f", aspect="auto", title="Temporal metric differences from admitted tail-risk features", labels={"color":"Metric difference"})
show_figure(fig, root, "risk_state_folds", 680)
contrasts = pd.DataFrame([r for r in risk_state["comparisons"] if r["block_dates"] == 20 and r["reference"] != "historical_mean"])
contrasts["Contrast"] = contrasts.variant + " vs " + contrasts.reference
fig = go.Figure(go.Scatter(x=contrasts.delta, y=contrasts.Contrast, mode="markers", marker={"size":9,"color":"#1F6C99"}, error_x={"type":"data","symmetric":False,"array":[b[1]-d for b,d in zip(contrasts.simultaneous_95_interval,contrasts.delta)],"arrayminus":[d-b[0] for b,d in zip(contrasts.simultaneous_95_interval,contrasts.delta)]}))
fig.add_vline(x=0,line_dash="dash",line_color="#9EAFBF")
fig.update_layout(title="Risk-state attribution with simultaneous 95% intervals",xaxis_title="Official metric difference")
show_figure(fig,root,"risk_state_uncertainty",740)
display(contrasts[["Contrast","delta","conditional_95_interval","simultaneous_95_interval"]])
positive = {b:sum(r["simultaneous_95_interval"][0]>0 for r in risk_state["joint_comparisons"] if r["block_dates"]==b) for b in [10,20,40]}
display(Markdown(f"**{risk_state['joint_comparison_count']} joint comparisons** across five domain phases. Positive simultaneous lower bounds: **{positive}**. Bounds condition on fitted models and do not undo the adaptive research history."))""",
        ),
        (
            "md",
            "### Completion gate\n\nThese experiments turn another plausible mechanism into measurable evidence, including negative results. A feature-count increase or a better pooled point estimate is insufficient to close the gate. The middle period, multiple-comparison uncertainty, and the lack of independent confirmation remain explicit. True futures carry, inventories, macro surprises, seasonality, weather, and text still require verified calendar/instrument mappings and historical release vintages that have not been established for the supplied anonymous date panel. Final optimization and final-test evaluation remain gated.",
        ),
    ]
