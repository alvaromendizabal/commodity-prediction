"""Interpret short-history and related-target feature ablations in the canonical notebook."""


def cells() -> list[tuple[str, str]]:
    return [
        (
            "md",
            "## Can short released histories add information?\n\nThe previous risk-state expansion left long-window priors largely unchanged. Fifth-, sixth- and eighth-place solutions motivate short context and shared target information; see the [competitive source audit](../docs/competitive-research.md). Their scores use another evaluation and are not reproduced here. Our earlier routed representation already included a latest own-target value. This experiment tests richer short history under the current fixed pooled learner.\n\n**24 candidate templates:** twelve own-history snapshots/summaries and twelve peer summaries. Same-pair peers align reversed spreads and exclude the own target. Shared-asset peers use fixed signed long/short exposures and exclude the same-pair group. Divide each released peer label by its horizon, aggregate available peers, and multiply by the receiving horizon. These are noisy contexts from differently timed observations, not reconstructed contemporaneous returns.\n\nEvery constituent is delayed by its own **horizon + 1 before aggregation**. Missing histories stay missing; coverage is explicit. No validation correlation learns the graph. A first-fold probe precedes a cumulative five-minute, nine-fit study. Fixed settings and the frozen admitted-tail control isolate the two feature families.",
        ),
        (
            "code",
            """from scripts.notebook_support import checked_released_context
context = checked_released_context(root)
display(Markdown(f"**Verified context lineage:** `{context['lineage'][:16]}` · {context['new_fitted_models']} model checkpoints · replay error {context['maximum_prediction_replay_error']:.1e}"))
context_names = ["historical_mean", "admitted_tail", "screened_tail", "tail_states_and_priors", "short_own", "short_peers", "short_joint"]
context_scores = pd.DataFrame([{"Variant": n, "Metric": context["summaries"][n]["official_metric"], **{f"Fold {i+1}": s for i,s in enumerate(context["summaries"][n]["fold_scores"])}} for n in context_names]).set_index("Variant")
display(context_scores.round(6))
fig = px.bar(context_scores.reset_index(), x="Metric", y="Variant", orientation="h", text_auto=".3f", title="Short released context under a fixed learner")
show_figure(fig, root, "released_context_scores", 560)
inventory = pd.DataFrame([{"Variant": r["variant"], "Fold": r["fold"]+1, "Candidates": r["selection"]["candidate_templates"], "Retained": r["selection"]["retained_templates"], "Rejected": r["selection"]["rejected_templates"], "Reasons": r["selection"]["rejection_reasons"]} for r in context["results"]])
display(inventory)""",
        ),
        (
            "md",
            "### Separate feature contribution from favorable periods\n\nThe own-only and peer-only additions share an unchanged control. The joint panel permits each component to be removed while the other remains. All three chronological periods are reported; no variant is chosen from the first-fold probe. Candidate counts are pooled templates, not independent signals. Latest own-target history is an existing information source, so feature novelty and representation contribution are separate questions.",
        ),
        (
            "code",
            """delta_folds = context_scores.loc[["short_own","short_peers","short_joint"], ["Fold 1","Fold 2","Fold 3"]].sub(context_scores.loc["admitted_tail",["Fold 1","Fold 2","Fold 3"]],axis=1)
fig = px.imshow(delta_folds.astype(float), text_auto=".3f", color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect="auto", title="Short-context changes across the three validation periods", labels={"color":"Metric change"})
show_figure(fig, root, "released_context_folds", 430)
context_bounds = pd.DataFrame([r for r in context["comparisons"] if r["block_dates"]==20])
context_bounds["Contrast"] = context_bounds.variant + " vs " + context_bounds.reference
fig = go.Figure(go.Scatter(x=context_bounds.delta, y=context_bounds.Contrast, mode="markers", error_x={"type":"data","symmetric":False,"array":[b[1]-d for b,d in zip(context_bounds.simultaneous_95_interval,context_bounds.delta)],"arrayminus":[d-b[0] for b,d in zip(context_bounds.simultaneous_95_interval,context_bounds.delta)]}))
fig.add_vline(x=0,line_dash="dash")
fig.update_layout(title={"text":"Short-context attribution with simultaneous uncertainty","xref":"container","xanchor":"left"},xaxis_title="Official metric difference")
show_figure(fig,root,"released_context_uncertainty",540)
display(context_bounds[["Contrast","delta","conditional_95_interval","simultaneous_95_interval"]])
display(Markdown(f"The comparison history now contains **{context['joint_comparison_count']} contrasts**. These intervals condition on fitted models and do not undo adaptive research. The final **247 origins remain untouched**."))""",
        ),
        (
            "md",
            "### Research decision\n\nOwn history scores **0.308890**, only +0.003554 over its matched control, with conditional 95% interval **[-0.008937, +0.017018]**. Its middle fold remains below historical means. Peer context scores **0.243727** and is worse in every fold; the combined panel scores **0.249300**. All 81/81/93 candidate columns are admitted. **Stop expansion of this peer formulation; promote neither family as proven.** The previous **0.309186** best remains unchanged. The [study review](../docs/released-context-research.md) records the measured results and decision. Neural joint representation, short raw sequences and unresolved external-data prerequisites remain separate research questions; this small study does not exhaust them.",
        ),
    ]
