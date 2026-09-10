"""Explain additional fixed-model feature sensitivities in the canonical notebook."""


def cells() -> list[tuple[str, str]]:
    return [
        (
            "md",
            "## Feature robustness: screening, timing, and conditional mechanisms\n\nThe preceding experiments leave specific alternatives unresolved. A 64-template relevance screen may hide complementary nonlinear information. Released priors may depend on very recent outcomes. Pooling can hide differences between horizons and markets. This exploratory follow-up tests those possibilities with **36 additional fits**, holding the unshrunk histogram-tree settings and the 535-date evaluation fixed. It reuses all parent controls and preserves their lineages.\n\nThere are **56 prior-by-group interactions** (14 released-prior summaries × four explicit metadata gates) and **12 bounded return/trend-by-shock/activity/freshness products**. These add 68 distinct templates to the research inventory, not 68 independent information sources. Market gates use the fraction of target legs in the market; the horizon gate identifies horizons 3 and 4. Structural absence is zero, applicable missingness remains missing.\n\nThe extra 1/5-date prior delays occur after the original horizon+1 release delay. The one-date market delay leaves static target metadata and already released priors unchanged. Increasing or removing the selection budget does not increase the tree's depth or iterations.",
        ),
        (
            "code",
            """from scripts.notebook_support import checked_robustness
robustness = checked_robustness(root)
display(Markdown(f"**Verified follow-up:** `{robustness['lineage'][:16]}` · {robustness['new_fitted_models']} fits · {robustness['checkpoint_count']} sealed stages · replay error {robustness['maximum_prediction_replay_error']:.1e}"))
rows = [{"Variant":n,"Metric":s["official_metric"],"Fold 1":s["fold_scores"][0],"Fold 2":s["fold_scores"][1],"Fold 3":s["fold_scores"][2]} for n,s in robustness["summaries"].items()]
display(pd.DataFrame(rows).set_index("Variant").round(6))
fig = px.bar(pd.DataFrame(rows).sort_values("Metric"),x="Metric",y="Variant",orientation="h",text_auto=".3f",title="Fixed-model feature sensitivities on the same validation dates")
fig.update_layout(margin={"l":220})
show_figure(fig,root,"robustness_scores",730)""",
        ),
        (
            "md",
            "### Matched effects and representation size\n\nComparisons against the frozen full tree isolate the changed feature representation. Compact conditional, factor, and risk/freshness representations also compare with the frozen reference-plus-priors tree. The simultaneous interval family expands to include all declared comparisons from the three domain phases. These exploratory intervals condition on fitted models; they cannot restore independence after repeated development research. An interval crossing zero is inconclusive, rather than proof of equivalence.",
        ),
        (
            "code",
            """matched = pd.DataFrame([r for r in robustness["comparisons"] if r["block_dates"] == 20 and r["reference"] == "all_control"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=matched.delta,y=matched.variant,mode="markers",marker={"size":10,"color":"#1F6C99"},error_x={"type":"data","symmetric":False,"array":[r[1]-d for r,d in zip(matched.simultaneous_95_interval,matched.delta)],"arrayminus":[d-r[0] for r,d in zip(matched.simultaneous_95_interval,matched.delta)]}))
fig.add_vline(x=0,line_dash="dash",line_color="#9EAFBF")
fig.update_layout(title="Sensitivity benefits relative to the frozen full tree",xaxis_title="Metric difference · joint simultaneous 95% bounds",margin={"l":220})
show_figure(fig,root,"robustness_uncertainty",660)
screen = pd.DataFrame([{"Variant":r["variant"],"Fold":r["fold"]+1,"Candidates":r["selection"]["candidate_templates"],"Retained":r["selection"]["retained_templates"],"Rejected":r["selection"]["rejected_templates"],"Reasons":r["selection"]["rejection_reasons"]} for r in robustness["results"]])
display(screen)
counts = {b:sum(r["simultaneous_95_interval"][0]>0 for r in robustness["joint_comparisons"] if r["block_dates"]==b) for b in [10,20,40]}
display(Markdown(f"Joint comparison count: **{robustness['joint_comparison_count']}**. Positive simultaneous lower bounds by block length: **{counts}**. Feature gate remains **open**."))""",
        ),
        (
            "md",
            "### Subgroup diagnostics and unresolved mechanisms\n\nMarket-involved target groups overlap. Single/pair and within-market rankings change which targets are compared, so these descriptive scores are not substitutes for the global metric and are not independently significant findings. Compare a variant with its control within the same subgroup. Dates with fewer than two observed subgroup targets are excluded by a truth-only rule, and coverage is reported. Undefined subgroup scores remain null. The single-asset group has one target per horizon, so its individual horizon ranking scores are undefined. The global metric continues to use all 535 dates.\n\nTrue delivery-curve carry, physical inventories, positioning, macro surprises, filings, options, and text still require verified dated inputs. Current website access does not establish historical data availability. The [domain research ledger](../docs/domain-feature-research.md) records source-specific timing, applicability, and the remaining evidence needed before feature-gate closure.",
        ),
        (
            "code",
            """groups = []
for n,s in robustness["summaries"].items():
    for group,g in s["subgroups"].items():
        groups.append({"Variant":n,"Group":group,"Targets":g["targets"],"Eligible dates":g["eligible_dates"],"Excluded dates":g["excluded_dates"],"Undefined":g["undefined_reason"],"Metric":g["official_metric"]})
display(pd.DataFrame(groups).query("Variant == 'all_control'")[["Group","Targets","Eligible dates","Excluded dates","Undefined"]].set_index("Group"))
group_table = pd.DataFrame(groups).pivot(index="Variant",columns="Group",values="Metric")
display(group_table.round(4))
control_groups = group_table.loc["all_control"]
delta_groups = group_table.drop(index=["all_control","historical_mean","compact_control"]).sub(control_groups,axis=1)
fig = px.imshow(delta_groups,color_continuous_scale="RdBu",color_continuous_midpoint=0,text_auto=".2f",aspect="auto",title="Descriptive subgroup changes relative to the full tree",labels={"color":"Metric change"})
show_figure(fig,root,"robustness_subgroups",680)
display(Markdown("**Decision:** retain the complete evidence, including weak or negative hypotheses. Do not promote an exploratory point-estimate winner or touch the final test. A feature is not established by its name, count, or a single favorable fold."))""",
        ),
    ]
