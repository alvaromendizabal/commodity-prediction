# Session-pair first-fold ablation protocol

## Completed evidence, not a new result

The submitted normalization report records 535 development dates. Current-market
control: 0.3097087232124053; normalized price: 0.3057333590409221;
volume confirmation: 0.3069106073242419; union: 0.29697041754207887.
All three pooled deltas are negative. The six continuation fits reused the three
first-fold fits and did not refit controls. Preserve this negative experiment.

Notebook 06 constructed 41 templates. Thirty-nine are numerical and two are
support indicators. Eleven numerical candidates had directionally consistent
training-half association; the other 28 are not proved useless, merely not
advanced by this screen. The selected names are frozen in the JSON declaration.
Only 89 of 424 target definitions have complete-leg OHLC: 87 pairs and 2 singles.
A narrow session representation cannot substitute for studying the other markets.

## Four predefined fits

1. Saved current_market + two complete-bar support indicators: new mask control.
2. Same base + indicators + six shortlisted intraday features.
3. Same base + indicators + five overnight/risk features (including risk share).
4. Same base + indicators + the exact eleven-feature union.

The original current_market model is replayed exactly, never refitted. Masks are
identical across new panels. Covariance-aware session features are computed for
the signed pair before normalization; all rolling reference moments stop at t-1.
Invalid observed values stay missing; unsupported targets use explicit structural
zeros and indicators. Supplied-row alignment does not establish synchronized
international closing times. Do not add rejected normalization features here.

## Information contract and budget

Training prefix: 0 through 1163. Model warmup: 252. Validation origins: 1169 through
1348. Label h is released at t+h+1; the existing five-date purge is unchanged.
Raw CSV numerical reads stop at 1348, with no later-fold or final labels loaded.
File hashing reads raw bytes for integrity, not outcome values for selection.
The existing cached feature panel is immutable; its first-fold prefix is used.
The 1164-row candidate prefix must replay exactly against the saved notebook-06
checkpoint. Current-market controls and parent hashes must verify before fitting.

Keep the project histogram settings fixed. Audit exact duplicates against the
current base and within candidates; report all training screening and numeric
admission. Any candidate selection is training-only. The previous screen covered
new features and normalization features, not the entire historical feature catalog.

One externally supervised worker has 240 seconds, including preflight, replay,
feature construction and the four fits. Fifteen-second progress heartbeats.
Independent sealed checkpoints survive later failure. Failed/incomplete runs stop
for diagnosis; completed results reopen without training. No automatic next folds,
GPU, dependency changes, GitHub/AWS writes, final model promotion or submission.
Notebook display, HTML export and ZIP backup are separate from the fitting worker.

## Review gate, not significance or promotion

A numerical panel earns review only if it beats BOTH the saved current_market and
the support-only control by at least 0.002. All exact replays and integrity gates
must pass. The test evaluates an already inspected development fold; do not call
it a fresh holdout. Seven paired contrasts use 10/20/40-date block resampling.
Conditional and simultaneous bounds are within-family; neither corrects all
prior adaptive experiments or screening. Record every negative panel.

If the mask-only panel wins, do not attribute that to numeric session information.
Inspect supported and other-target diagnostic effects separately: a pooled model
can change predictions for unsupported targets after refitting. The subgroup
metric uses at least three usable targets/date and is not the global metric.

## Domain basis and remaining research

Blanc, Chicheportiche & Bouchaud (2013), overnight/intraday volatility feedback:
https://arxiv.org/abs/1309.5806
This supports session decomposition for stock volatility; commodity return
predictability is an unproven transfer hypothesis, not an implication.

Gorton, Hayashi & Rouwenhorst, commodity futures fundamentals:
https://www.nber.org/papers/w13249
Inventories and basis inform futures risk-premium research. They are not
interchangeable with anonymous target row IDs or necessarily short-horizon alpha.
Actual instrument, timestamp and licensing contracts precede external-data joins.

After this test, consider close-only pair innovation, relative market structure
and horizon-correct released-target context for the other markets, after checking
existing return, volatility, graph and macro-relative coverage. Keep feature versus
learner/objective comparisons distinct. Research is not exhausted and a historical
leaderboard score on other dates is not comparable to this first-fold result.

## Artifacts and manual operations

Notebook: notebooks/07_session_feature_ablation.ipynb
Declaration: configs/manual_session_ablation.json
Code: scripts/commodity_session_ablation.py
Report/dashboard/ZIP: logs/manual_session_ablation/
New checkpoints: artifacts/session_pair_ablation/<child-lineage>/fold_0/
Save the executed notebook. Keep private models/predictions out of public GitHub.
Download the ZIP for an off-disk copy, then Stop space (never Delete space).
No space shutdown or remote GitHub write is performed by this helper.
