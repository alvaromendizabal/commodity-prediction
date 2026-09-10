"""Domain hypotheses, controlled results, and remaining research in the canonical notebook."""


def cells() -> list[tuple[str, str]]:
    return [
        (
            "md",
            "## Domain study: separate target priors from changing market information\n\nThe preceding study motivated a stronger question: can economic features forecast the residual around each target's training mean? This follow-up keeps those earlier results intact and tests 13 domain families in a shared, risk-normalized date/target model.\n\n[Domain research ledger and primary sources](../docs/domain-feature-research.md) explains instrument conventions, mathematical definitions, and the data needed for remaining avenues. **Feature gate: open.**",
        ),
        (
            "code",
            """domain_rows = pd.DataFrame([{"Variant": name, "Metric": value["official_metric"], "Fold scores": value["fold_scores"], "Residual weights": value["residual_weights"]} for name, value in domain["summaries"].items()])
inventory = domain["inventory"]
display(pd.Series({"Pooled templates": inventory["templates"], "Instantiated source series": inventory["source_series_count"], "Target-template assignments": inventory["target_template_assignments"], "New fitted models (including inner fits)": domain["new_fitted_models"], "Outer evaluations (including controls)": domain["outer_evaluations"], "Validation dates": domain["validation_dates"]}, name="Domain study inventory").to_frame())
display(Markdown(inventory["counting_note"]))""",
        ),
        (
            "md",
            "### Translate mechanisms into observable features\n\nTrend shape, tail risk, activity, overnight versus intraday paths, and stale observations describe the individual market. Factor-relative innovations, macro-market links, FX currency strengths, contract/settlement differences, and pair dynamics describe relationships. Release-safe historical pools, horizon structure, and strictly trailing latent factors describe shared prediction structure.\n\nJPY prices are divided by USD/JPY before comparison with dollar movements. JPX/GLD and futures-contract differences remain **relative-price proxies**: missing delivery curves, units, timestamps, and transaction costs prevent carry or arbitrage claims. No guessed calendar, revised fundamentals, future holdings, or retrospective news enter predictors.",
        ),
        (
            "code",
            """family_counts = pd.Series(inventory["family_templates"]).sort_values().rename_axis("Family").reset_index(name="Templates")
fig = px.bar(family_counts, x="Templates", y="Family", orientation="h", text="Templates", title="Domain mechanisms expressed as 380 pooled input templates")
fig.update_traces(marker_color="#1F6C99", textposition="outside")
show_figure(fig, root, "domain_family_templates", 700)""",
        ),
        (
            "md",
            "### Fix model settings and put residual calibration inside training\n\nEvery date has equal total fitting weight. Each fit estimates target mean and risk, imputes and scales inputs, and selects at most 64 templates using only its training interval. A fixed Ridge penalty and fixed small histogram-tree configuration limit model-search confounding. Targets are standardized and clipped for fitting; evaluation uses unchanged observed targets.\n\nTwo purged 126-date inner validation blocks choose a residual weight from 0, .05, .1, .25, .5, or 1. A zero means the inner evidence chose the historical-mean fallback. **Raw variants always use weight 1**, exposing feature behavior that shrinkage might hide. All outer validation dates remain outside these decisions.\n\nThe 31 fitted variants include a reference, 13 single-family additions, all families, sign-stable screening, 13 removals, and two tree controls. Historical, graph-projected, and winsorized means test alternative priors.",
        ),
        (
            "code",
            """focus_names = ["historical_mean", "graph_mean", "winsorized_mean", "pooled_reference", "pooled_all", "pooled_stable", "tree_reference", "tree_all", "raw__pooled_reference", "raw__pooled_all", "raw__tree_all"]
focus = domain_rows.loc[domain_rows.Variant.isin(focus_names)].sort_values("Metric")
fig = px.bar(focus, x="Metric", y="Variant", orientation="h", text_auto=".3f", title="Can domain residual forecasts improve the strong prior?")
fig.update_traces(marker_color="#27A394", textposition="outside")
fig.update_xaxes(range=[min(0,focus.Metric.min())-.04,max(0,focus.Metric.max())+.05])
show_figure(fig, root, "domain_model_scores", 660)
display(focus.sort_values("Metric",ascending=False).round(5))""",
        ),
        (
            "md",
            "### Each family receives both an addition and a removal test\n\nAn addition compares reference-plus-family with reference alone. A removal compares all families with the representation lacking that family. Positive values favor including the family. Screening is refit in each candidate pool, so a removal can expose substitute features. The plot uses **raw weight-1 models** to isolate the unshrunk feature pipeline; the table includes the nested-calibrated versions.\n\nError bars are simultaneous 95% bounds across all declared comparisons in this study, using 20-date blocks and 2,000 paired resamples within folds. Ten- and forty-date sensitivity and conditional intervals remain in the aggregate report. Repeated development research prevents a confirmatory interpretation.",
        ),
        (
            "code",
            """domain_bounds = pd.DataFrame(domain["comparisons"])
matched = []
for r in domain["comparisons"]:
    if r["block_dates"] != 20:
        continue
    if r["variant"].startswith("raw__add_") and r["reference"] == "raw__pooled_reference":
        matched.append({**r,"Family":r["variant"].removeprefix("raw__add_"),"Test":"Add to reference"})
    elif r["variant"] == "raw__pooled_all" and r["reference"].startswith("raw__drop_"):
        matched.append({**r,"Family":r["reference"].removeprefix("raw__drop_"),"Test":"Remove from full"})
matched = pd.DataFrame(matched)
fig = go.Figure()
for label, color, symbol in [("Add to reference","#1F6C99","circle"),("Remove from full","#D58A29","diamond")]:
    group = matched.loc[matched.Test == label]
    lo = group.simultaneous_95_interval.map(lambda x:x[0]); hi = group.simultaneous_95_interval.map(lambda x:x[1])
    fig.add_trace(go.Scatter(x=group.delta,y=group.Family,mode="markers",name=label,marker={"color":color,"symbol":symbol,"size":10},error_x={"type":"data","symmetric":False,"array":hi-group.delta,"arrayminus":group.delta-lo}))
fig.add_vline(x=0,line_dash="dash",line_color="#9EAFBF")
fig.update_layout(title="Matched domain-family tests with simultaneous uncertainty",xaxis_title="Metric benefit from including the family",legend={"orientation":"h","y":-0.15})
show_figure(fig,root,"domain_matched_families",780)
display(domain_bounds.loc[(domain_bounds.block_dates == 20) & domain_bounds.variant.isin(["pooled_all","pooled_reference","tree_all","graph_mean"]) & domain_bounds.reference.isin(["historical_mean","pooled_reference","tree_reference"]),["variant","reference","delta","conditional_95_interval","simultaneous_95_interval"]])""",
        ),
        (
            "md",
            "### Check temporal and horizon stability, not just the pooled score\n\nA feature gain that changes sign across periods is not established as dependable. The heatmap shows unchanged outer folds; individual horizon metrics and 60-date blocks are preserved in the report. Selected residual weights show whether training evidence trusted each signal before outer evaluation.",
        ),
        (
            "code",
            """stability_rows = [{"Variant":name,"Fold":f"Fold {i+1}","Metric":metric} for name in focus_names for i,metric in enumerate(domain["summaries"][name]["fold_scores"])]
matrix = pd.DataFrame(stability_rows).pivot(index="Variant",columns="Fold",values="Metric")
fig = px.imshow(matrix,text_auto=".3f",color_continuous_scale="RdBu",color_continuous_midpoint=0,aspect="auto",title="Domain signals face three different validation periods")
show_figure(fig,root,"domain_fold_stability",650)
display(pd.DataFrame([{"Variant":name, "Weights chosen inside training":domain["summaries"][name]["residual_weights"], "Adjacent-fold template Jaccard":domain["summaries"][name]["adjacent_fold_selection_jaccard"]} for name in ["pooled_reference","pooled_all","pooled_stable","tree_all"]]))
display(pd.DataFrame([{"Variant":name,"Horizon":h,**m} for name in ["historical_mean","pooled_all","raw__pooled_all"] for h,m in domain["summaries"][name]["horizon_metrics"].items()]).round(4))""",
        ),
        (
            "md",
            "### Audit what the full model retained\n\nThese counts refer to a **single pooled model per fold**. They are not comparable with sums over 424 target-specific models in the earlier study. A rejected template may be constant, duplicated, highly correlated, sparse, or below the fixed feature budget; rejection does not prove an economic mechanism absent.",
        ),
        (
            "code",
            """display(pd.DataFrame([{k:r[k] for k in ["fold","candidate_templates","retained_templates","rejected_templates","rejection_reasons"]} for r in domain["screening"]]))
selected_rows = [{"Fold":f"Fold {r['fold']+1}","Family":family,"Retained":count} for r in domain["screening"] for family,count in r["retained_by_family"].items()]
fig = px.bar(pd.DataFrame(selected_rows),x="Family",y="Retained",color="Fold",barmode="group",title="Which domain templates survive training-only screening?")
fig.update_xaxes(tickangle=-30)
show_figure(fig,root,"domain_retained_families",560)
permuted = [{"Fold":r["fold"],"Family":family,"Metric drop":value} for r in domain["group_permutation"] for family,s in r["families"].items() for value in s["metric_drop_repetitions"]]
if permuted:
    importance = pd.DataFrame(permuted).groupby("Family")["Metric drop"].mean().sort_values().reset_index()
    fig = px.bar(importance,x="Metric drop",y="Family",orientation="h",title="Frozen raw model reliance under family block permutation")
    fig.update_traces(marker_color="#1F6C99")
    fig.add_vline(x=0,line_dash="dash",line_color="#9EAFBF")
    show_figure(fig,root,"domain_group_permutation",600)
display(Markdown("Permutation averages fold-level metric changes; it is neither a pooled score nor causal importance. Unselected families have no measured reliance in this particular frozen model."))""",
        ),
        (
            "code",
            """leader = domain_rows.sort_values("Metric",ascending=False).iloc[0]
mean_score = domain["summaries"]["historical_mean"]["official_metric"]
evidence20 = domain_bounds.loc[domain_bounds.block_dates == 20]
positive = evidence20.loc[evidence20.simultaneous_95_interval.map(lambda v:v[0]>0)]
matched_positive = positive.loc[(positive.variant.str.contains("add_")) | (positive.reference.str.contains("drop_")) | ((positive.reference.str.contains("reference")) & (positive.variant.str.contains("all")))]
display(Markdown(f"**Observed point-estimate leader:** `{leader.Variant}` at **{leader.Metric:.5f}**, versus historical means at **{mean_score:.5f}**. This ranking includes exploratory variants and is not a final-model selection.\\n\\n**{len(matched_positive)} matched feature comparisons** have positive simultaneous lower bounds at the 20-date block setting, among **{domain['declared_comparison_count']} declared comparisons**. Consult the 10/40-date sensitivity before interpreting them.\\n\\n**Completed:** {domain['new_fitted_models']} new fitted models, {domain['outer_evaluations']} outer evaluations, and preserved prior studies. Final test: **not evaluated**. Feature gate: **open**."))
display(domain_rows.sort_values("Metric",ascending=False).head(12).round(5))""",
        ),
        (
            "md",
            "### Research decision and remaining stones\n\nThis is evidence about the tested representations under fixed models, not a claim that every economic avenue is exhausted. Examine conditional target groups, nonlinear interactions, alternative normalization and robust priors only where the measured weaknesses justify them. Preserve failed hypotheses and matched comparisons; do not promote a family because of its name or column count.\n\nTrue futures carry needs dated delivery curves; inventories and positioning need publication vintages; seasonality and weather need a verified calendar; fundamentals need historical classifications; options and news need appropriately licensed as-of records. These prerequisites are explicit in the [domain ledger](../docs/domain-feature-research.md). Feature engineering remains open until major feasible avenues, stability, and diminishing returns have sufficient evidence. **No final optimization, final-test evaluation, or competition submissions were performed.**",
        ),
    ]


def tree_cells() -> list[tuple[str, str]]:
    return [
        (
            "md",
            "## Follow the nonlinear signal with matched family attribution\n\nThe parent tree improved its point estimate, while uncertainty remained wide. This explicitly exploratory follow-up tests **13 additions and 13 removals with the fixed tree** across the same three folds. All 78 fits reuse the existing feature panel; the reference, full, and historical-mean controls are preserved. Residual weight stays at one to assess unshrunk feature contributions. No hyperparameter search or additional validation dates are introduced.\n\nThe bounds now cover **177 comparisons jointly across both domain phases**. They still cannot turn adaptive development research into independent confirmation.",
        ),
        (
            "code",
            """tree_rows = pd.DataFrame([{"Variant":name,"Metric":m["official_metric"],"Fold scores":m["fold_scores"]} for name,m in attribution["summaries"].items()])
matched_tree = []
for r in attribution["comparisons"]:
    if r["block_dates"] != 20:
        continue
    if r["variant"].startswith("add_") and r["reference"] == "reference_control":
        matched_tree.append({**r,"Family":r["variant"].removeprefix("add_"),"Test":"Add to reference"})
    elif r["variant"] == "all_control" and r["reference"].startswith("drop_"):
        matched_tree.append({**r,"Family":r["reference"].removeprefix("drop_"),"Test":"Remove from full"})
matched_tree = pd.DataFrame(matched_tree)
fig = go.Figure()
for label,color,symbol in [("Add to reference","#1F6C99","circle"),("Remove from full","#D58A29","diamond")]:
    group = matched_tree.loc[matched_tree.Test == label]
    lo=group.simultaneous_95_interval.map(lambda v:v[0]); hi=group.simultaneous_95_interval.map(lambda v:v[1])
    fig.add_trace(go.Scatter(x=group.delta,y=group.Family,mode="markers",name=label,marker={"color":color,"symbol":symbol,"size":10},error_x={"type":"data","symmetric":False,"array":hi-group.delta,"arrayminus":group.delta-lo}))
fig.add_vline(x=0,line_dash="dash",line_color="#9EAFBF")
fig.update_layout(title="Which families explain the nonlinear point-estimate gain?",xaxis_title="Metric benefit from inclusion · joint simultaneous 95% bounds",legend={"orientation":"h","y":-0.15})
show_figure(fig,root,"tree_matched_families",780)
display(tree_rows.sort_values("Metric",ascending=False).head(15).round(5))""",
        ),
        (
            "md",
            "### Test reliance in the model that showed the gain\n\nThe original domain permutation diagnosed the pooled linear model. This follow-up perturbs each selected family in the **saved full tree** using shared date blocks across targets, without refitting. Addition, removal, and permutation address different questions; agreement across them is more useful than a single importance ranking.",
        ),
        (
            "code",
            """tree_permuted = [{"Family":family,"Metric drop":v} for record in attribution["group_permutation"] for family,m in record["families"].items() for v in m["metric_drop_repetitions"]]
importance = pd.DataFrame(tree_permuted).groupby("Family")["Metric drop"].mean().sort_values().reset_index()
fig = px.bar(importance,x="Metric drop",y="Family",orientation="h",title="Frozen full-tree reliance on domain families")
fig.update_traces(marker_color="#27A394")
fig.add_vline(x=0,line_dash="dash",line_color="#9EAFBF")
show_figure(fig,root,"tree_group_permutation",620)
display(matched_tree[["Family","Test","delta","conditional_95_interval","simultaneous_95_interval"]].sort_values(["Test","delta"],ascending=[True,False]))""",
        ),
        (
            "code",
            """tree_best = tree_rows.sort_values("Metric",ascending=False).iloc[0]
positive_tree = matched_tree.loc[matched_tree.simultaneous_95_interval.map(lambda v:v[0]>0)]
joint_positive = [r for r in attribution["joint_comparisons"] if r["block_dates"]==20 and r["simultaneous_95_interval"][0]>0]
display(Markdown(f"**Tree attribution point-estimate leader:** `{tree_best.Variant}` at **{tree_best.Metric:.5f}**. **{len(positive_tree)} of 26 matched tree-family comparisons** have positive simultaneous lower bounds at the 20-date setting. Across both domain phases, **{len(joint_positive)} of {attribution['joint_comparison_count']} comparisons** have positive lower bounds.\\n\\nThe study completed **{domain['new_fitted_models'] + attribution['new_fitted_models']} new fitted models** across both domain phases, plus nine mean-control evaluations. All earlier 99 fold experiments are preserved. Feature engineering remains **open**; final evaluation has not run."))""",
        ),
        (
            "md",
            "### Decision after the matched nonlinear tests\n\nUse this evidence to identify conditional mechanisms worth further scrutiny, not to pick a winner from a leaderboard of development scores. A positive pooled estimate needs adequate uncertainty bounds, consistent temporal behavior, and robustness to alternative representations before promotion. No family is established merely by a high point estimate or a low permutation score.\n\nRemaining research includes defensible conditional interactions and target-group behavior, plus data-dependent carry, physical supply, positioning, calendar, option, and event features. The domain ledger states the prerequisites for each. The final test and final optimization remain gated. [Complete tree evidence](../reports/tree_attribution.json) · [Domain ledger](../docs/domain-feature-research.md).",
        ),
    ]
