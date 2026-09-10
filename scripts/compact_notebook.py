"""Present compact mechanism tests in the existing research notebook."""


def cells() -> list[tuple[str, str]]:
    return [
        (
            "md",
            "## Compact feature mechanisms: separating evidence from screening\n\nThe preceding compact risk/freshness lead raises three different questions: which block helps, does it survive information delays, and did supervised screening hide interactions? This follow-up completes **36 fixed-tree fits** over the same three folds. It tests the reference-plus-priors base, the 30 tail-risk templates, the 12 freshness templates, and their joint representation. Both the original 64-column screen and a matched admission policy are reported.\n\n**Admission policy:** keep every training-usable nonduplicate column. Missingness and constant checks remain; relevance ranking and near-correlation exclusions cannot remove a usable candidate. This tests the representation under the same 64-iteration tree capacity. It does not guarantee the model will learn every nonlinear relationship. The 12 previously screened-out products are reused exactly. A further 42 state-by-FX-leg-share templates test whether target representations need market-specific conditioning. They are not isolated FX-leg returns or new external information.",
        ),
        (
            "code",
            """from scripts.notebook_support import checked_compact
compact = checked_compact(root)
display(Markdown(f"**Verified compact study:** `{compact['lineage'][:16]}` · {compact['new_fitted_models']} fits · {compact['checkpoint_count']} sealed stages · exact replay error {compact['maximum_prediction_replay_error']:.1e}"))
rows = [{"Variant":n,"Metric":s["official_metric"],"Fold 1":s["fold_scores"][0],"Fold 2":s["fold_scores"][1],"Fold 3":s["fold_scores"][2]} for n,s in compact["summaries"].items()]
scores = pd.DataFrame(rows)
display(scores.set_index("Variant").round(6))
fig = px.bar(scores.sort_values("Metric"),x="Metric",y="Variant",orientation="h",text_auto=".3f",title="Compact feature mechanisms on 535 matched dates")
show_figure(fig,root,"compact_scores",770)
screening = pd.DataFrame([{"Variant":r["variant"],"Fold":r["fold"]+1,"Candidates":r["selection"]["candidate_templates"],"Retained":r["selection"]["retained_templates"],"Rejected":r["selection"]["rejected_templates"],"Reasons":r["selection"]["rejection_reasons"],"Product slots":sum(n.startswith("mechanism_products__") for n in r["selection"]["selected_names"]),"FX state slots":sum(n.startswith("fx_states__") for n in r["selection"]["selected_names"])} for r in compact["results"]])
display(screening)""",
        ),
        (
            "md",
            "### Matched contrasts and uncertainty\n\nThe two screened component tests compare against the frozen reference-plus-priors control; removing each component from the earlier joint lead reveals conditional contribution. The admitted four-way design compares the base, tail, freshness, and joint panels without a relevance-budget bottleneck. Product and FX-state additions compare against the admitted joint panel. Extra prior delays of one/five dates start **after** the original horizon+1 label release; market delays affect dynamic market features only. The compound one-date test shifts each dynamic family exactly once and preserves static metadata.\n\nAll new contrasts join the existing 204 domain comparisons, with 2,000 paired block bootstrap replications at 10, 20, and 40 dates. Simultaneous bounds control the declared comparison family conditional on these fitted models. They do not account for every adaptive research decision or create independent confirmation.",
        ),
        (
            "code",
            """matched = pd.DataFrame([r for r in compact["comparisons"] if r["block_dates"] == 20 and r["reference"] != "historical_mean"])
matched["Contrast"] = matched.variant + " vs " + matched.reference
fig = go.Figure(go.Scatter(x=matched.delta,y=matched.Contrast,mode="markers",marker={"size":9,"color":"#1F6C99"},error_x={"type":"data","symmetric":False,"array":[b[1]-d for b,d in zip(matched.simultaneous_95_interval,matched.delta)],"arrayminus":[d-b[0] for b,d in zip(matched.simultaneous_95_interval,matched.delta)]}))
fig.add_vline(x=0,line_dash="dash",line_color="#9EAFBF")
fig.update_layout(title="Matched feature changes with joint uncertainty",xaxis_title="Official metric difference · simultaneous 95% interval")
show_figure(fig,root,"compact_uncertainty",850)
display(matched[["Contrast","delta","conditional_95_interval","simultaneous_95_interval"]])
positive = {b:sum(r["simultaneous_95_interval"][0]>0 for r in compact["joint_comparisons"] if r["block_dates"]==b) for b in [10,20,40]}
display(Markdown(f"**{compact['joint_comparison_count']} joint comparisons.** Positive simultaneous lower bounds by block length: **{positive}**. These are exploratory development results; the final test remains untouched."))""",
        ),
        (
            "md",
            "### Economic interpretation and remaining limitations\n\nTail-risk inputs describe recent shocks, downside variation, and path extremes; freshness inputs describe missing/stale observations and resumption. Their interaction is plausible because a large recorded move can reflect either new information or accumulated price adjustment. This is a hypothesis about this panel, not a claim that missing observations measure tradable liquidity.\n\nFX conditioning follows a concrete market distinction: [BIS research on fragmented FX execution](https://www.bis.org/publications/qr-201912/fx-trade-execution-complex-and-highly-fragmented) describes dispersed venues, dealer internalisation, and incomplete visibility of activity. Those observations motivate allowing different state responses. They do not validate this feature formula. No order-book depth, true FX volume, physical inventory, or historical macro vintage is created by a metadata gate.\n\nDescriptive group changes below use the same truth-only eligibility rule as the preceding analysis. Market groups overlap; one-target horizon rankings remain undefined. The research ledger records results, failures, timing assumptions, and the remaining feature-gate requirements.",
        ),
        (
            "code",
            """groups = []
for n in ["admitted_base","admitted_tail","admitted_freshness","admitted_joint","admitted_products","admitted_fx_states"]:
    for group,s in compact["summaries"][n]["subgroups"].items():
        groups.append({"Variant":n,"Group":group,"Metric":s["official_metric"],"Eligible dates":s["eligible_dates"],"Excluded dates":s["excluded_dates"]})
group_table = pd.DataFrame(groups).pivot(index="Variant",columns="Group",values="Metric")
display(group_table.round(4))
deltas = group_table.sub(group_table.loc["admitted_joint"],axis=1).drop(index="admitted_joint")
fig = px.imshow(deltas,color_continuous_scale="RdBu",color_continuous_midpoint=0,text_auto=".2f",aspect="auto",title="Descriptive subgroup changes from the admitted joint panel",labels={"color":"Metric change"})
show_figure(fig,root,"compact_subgroups",520)
display(Markdown("**Feature gate: open.** Preserve failed hypotheses and valid checkpoints. Evaluate fold stability and matched uncertainty before proposing any feature promotion. More columns, an improved aggregate score, or one favorable subgroup does not establish completion."))""",
        ),
    ]
