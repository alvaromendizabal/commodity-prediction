"""Execute the domain feature study with sealed, independently resumable stages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from commodity_prediction.data import Fold, load_data
from commodity_prediction.runtime import (
    RunLog,
    atomic_json,
    digest,
    fingerprint,
    seal_checkpoint,
    verify_checkpoint,
)
from commodity_prediction.studies.evaluation import compare_predictions, evaluate
from commodity_prediction.studies.run import study_folds, study_lineage

from .catalog import Panel
from .diagnostics import group_permutations, temporal_diagnostics
from .experiment import (
    Experiment,
    choose_weight,
    control_model,
    declared_comparisons,
    experiment_plan,
    inner_folds,
)
from .features import build_panel
from .model import fit, prepare, select


def domain_lineage(root: Path, config: dict) -> tuple[str, dict]:
    parent_config = json.loads((root / "configs/feature_study.json").read_text())
    parent, _ = study_lineage(root, parent_config)
    if parent != config["parent_lineage"]:
        raise ValueError("Earlier study lineage changed; do not reuse its evidence")
    files = sorted((root / "src/commodity_prediction/domain").glob("*.py"))
    evidence = {
        "parent_lineage": parent,
        "config": config,
        "files": {str(p.relative_to(root)): digest(p) for p in files},
        "final_evaluation_sha256": digest(root / "configs/final_evaluation.json"),
        "experiments": [asdict(e) for e in experiment_plan()],
    }
    return fingerprint(evidence), evidence


def load_panel(stage: Path) -> Panel:
    inventory = json.loads((stage / "inventory.json").read_text())
    with np.load(stage / "panel.npz") as archive:
        panel = Panel(
            archive["values"],
            inventory["dates"],
            inventory["targets"],
            inventory["names"],
            inventory["source_series"],
        )
    panel.validate()
    return panel


def run_experiments(
    panel: Panel,
    y: pd.DataFrame,
    pairs: pd.DataFrame,
    folds: list[Fold],
    config: dict,
    directory: Path,
    lineage: str,
    log: RunLog,
    checkpoint: Callable[[Path], None],
    plan: list[Experiment] | None = None,
) -> list[dict]:
    plan = experiment_plan() if plan is None else plan
    results = []
    for outer in folds:
        if outer.train_stop - 1 + 5 >= outer.validation_start:
            raise ValueError("Unreleased training labels at outer boundary")
        inner = inner_folds(outer.train_stop, config)
        # All experiments share training statistics, but each pool gets independent screening.
        schedule = [(f"inner_{f.number}", f) for f in inner] + [("outer", outer)]
        for scope, fold in schedule:
            base = directory / f"fold_{outer.number}" / scope
            pending = [e for e in plan if not verify_checkpoint(base / e.name, lineage)]
            stats = None
            if pending:
                with log.stage(f"fold_{outer.number}/{scope}/training_statistics"):
                    stats = prepare(panel, y, fold.train_stop, config)
            truth = y.iloc[fold.validation_start : fold.validation_stop]
            for experiment in plan:
                stage = base / experiment.name
                with log.stage(f"fold_{outer.number}/{scope}/{experiment.name}"):
                    if verify_checkpoint(stage, lineage):
                        record = json.loads((stage / "result.json").read_text())
                        log.event("checkpoint_reused", stage=str(stage.relative_to(directory)))
                    else:
                        if stats is None:
                            raise ValueError("Missing training statistics")
                        selected, audit = select(
                            stats, experiment.candidates(panel.names), config, experiment.stable
                        )
                        model = fit(stats, selected, panel, y, experiment.algorithm, config)
                        calibration = None
                        if scope == "outer":
                            records = []
                            for inside in inner:
                                path = (
                                    directory
                                    / f"fold_{outer.number}/inner_{inside.number}"
                                    / experiment.name
                                )
                                verify_checkpoint(path, lineage)
                                bundle = joblib.load(path / "model.joblib")
                                records.append(
                                    (
                                        y.iloc[inside.validation_start : inside.validation_stop],
                                        pd.read_parquet(path / "raw_predictions.parquet"),
                                        bundle.target_mean,
                                    )
                                )
                            calibration = choose_weight(records, config)
                            model.weight = calibration["selected_weight"]
                        raw = model.predict(
                            panel, fold.validation_start, fold.validation_stop, weight=1
                        )
                        prediction = model.predict(
                            panel, fold.validation_start, fold.validation_stop
                        )
                        raw.index.name = prediction.index.name = truth.index.name
                        record = {
                            "variant": experiment.name,
                            "fold": outer.number,
                            "scope": scope,
                            "train_stop": fold.train_stop,
                            "validation_start": fold.validation_start,
                            "validation_stop": fold.validation_stop,
                            "selected_weight": model.weight,
                            "calibration": calibration,
                            "selection": audit,
                            "metrics": evaluate(truth, prediction, pairs),
                            "raw_metrics": evaluate(truth, raw, pairs),
                        }
                        stage.mkdir(parents=True, exist_ok=True)
                        joblib.dump(model, stage / "model.joblib", compress=3)
                        replay = joblib.load(stage / "model.joblib").predict(
                            panel, fold.validation_start, fold.validation_stop
                        )
                        np.testing.assert_array_equal(replay.to_numpy(), prediction.to_numpy())
                        prediction.to_parquet(stage / "predictions.parquet")
                        raw.to_parquet(stage / "raw_predictions.parquet")
                        atomic_json(stage / "result.json", record)
                        seal_checkpoint(
                            stage,
                            lineage,
                            [
                                "model.joblib",
                                "predictions.parquet",
                                "raw_predictions.parquet",
                                "result.json",
                            ],
                        )
                    checkpoint(stage)
                    if scope == "outer":
                        results.append(record)
            del stats
        for name in ["historical_mean", "graph_mean", "winsorized_mean"]:
            stage = directory / f"fold_{outer.number}/outer" / name
            with log.stage(f"fold_{outer.number}/outer/{name}"):
                if verify_checkpoint(stage, lineage):
                    record = json.loads((stage / "result.json").read_text())
                    log.event("checkpoint_reused", stage=str(stage.relative_to(directory)))
                else:
                    model = control_model(name, y, pairs, outer.train_stop, config)
                    truth = y.iloc[outer.validation_start : outer.validation_stop]
                    prediction = model.predict(panel, outer.validation_start, outer.validation_stop)
                    prediction.index.name = truth.index.name
                    record = {
                        "variant": name,
                        "fold": outer.number,
                        "scope": "outer",
                        "train_stop": outer.train_stop,
                        "validation_start": outer.validation_start,
                        "validation_stop": outer.validation_stop,
                        "selected_weight": 0,
                        "selection": None,
                        "calibration": None,
                        "metrics": evaluate(truth, prediction, pairs),
                    }
                    stage.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, stage / "model.joblib", compress=3)
                    prediction.to_parquet(stage / "predictions.parquet")
                    atomic_json(stage / "result.json", record)
                    seal_checkpoint(
                        stage, lineage, ["model.joblib", "predictions.parquet", "result.json"]
                    )
                checkpoint(stage)
                results.append(record)
    return results


def summarize(
    panel: Panel,
    y: pd.DataFrame,
    pairs: pd.DataFrame,
    folds: list[Fold],
    results: list[dict],
    directory: Path,
    lineage: str,
    config: dict,
) -> dict:
    summaries = {}
    daily = {}
    lengths = [f.validation_stop - f.validation_start for f in folds]
    for variant in dict.fromkeys(r["variant"] for r in results):
        records = [r for r in results if r["variant"] == variant]
        for raw in [False, True] if records[0]["selection"] is not None else [False]:
            name = "raw__" + variant if raw else variant
            filename = "raw_predictions.parquet" if raw else "predictions.parquet"
            predictions = pd.concat(
                [
                    pd.read_parquet(directory / f"fold_{f.number}/outer" / variant / filename)
                    for f in folds
                ]
            )
            metrics = evaluate(y.loc[predictions.index], predictions, pairs)
            daily[name] = np.asarray(metrics["daily_rank_correlations"])
            selection = [
                r["selection"]["selected_names"] for r in records if r["selection"] is not None
            ]
            summaries[name] = {
                **metrics,
                "fold_scores": [
                    r["raw_metrics" if raw else "metrics"]["official_metric"] for r in records
                ],
                "residual_weights": [1.0 if raw else r["selected_weight"] for r in records],
                **temporal_diagnostics(daily[name], lengths, selection),
            }
    comparisons = declared_comparisons(list(daily))
    intervals = compare_predictions(daily, lengths, comparisons, config)
    selected = [r for r in results if r["variant"] == "pooled_all"]
    screening = []
    for r in selected:
        audit = r["selection"]
        screening.append(
            {
                "fold": r["fold"],
                **audit,
                "retained_by_family": dict(
                    Counter(n.split("__", 1)[0] for n in audit["selected_names"])
                ),
            }
        )
    return {
        "lineage": lineage,
        "parent_lineage": config["parent_lineage"],
        "feature_gate": "open",
        "holdout_evaluated": False,
        "validation_dates": sum(lengths),
        "final_test_start_date_id": 1714,
        "untouched_final_test_dates": 247,
        "fitted_variants": len(experiment_plan()),
        "controls": 3,
        "outer_evaluations": len(results),
        "new_fitted_models": len(experiment_plan()) * len(folds) * (config["inner_folds"] + 1),
        "templates": len(panel.names),
        "inventory": panel.inventory(),
        "summaries": summaries,
        "comparisons": intervals,
        "declared_comparison_count": len(comparisons),
        "screening": screening,
        "results": results,
        "limitations": [
            "Repeated development-window research makes all intervals conditional and exploratory; no pristine confirmation set was evaluated.",
            "Simultaneous bounds cover the declared comparisons within this study, not the entire adaptive research history.",
            "Fixed pooled models and a 64-template budget can miss useful nonlinear, interaction, or target-specific effects.",
            "Price relationships are proxies, not verified carry, cointegration, causal effects, or executable arbitrage.",
            "Date IDs do not identify verified calendar dates or synchronized market closes.",
            "True futures curves, inventories, positioning, release-vintage fundamentals, weather, options, and news require additional lawful point-in-time data.",
            "247 final-test origins (1714–1960) remain untouched; 1709–1713 is a permanent buffer, not a pristine final test.",
        ],
    }


def run_study(root: Path, sync: bool = False) -> dict:
    config = json.loads((root / "configs/domain_study.json").read_text())
    if config["feature_gate"] != "open":
        raise ValueError("Feature research gate must remain open")
    lineage, evidence = domain_lineage(root, config)
    log = RunLog(root / "logs/domain_study.jsonl", config["heartbeat_seconds"])
    storage = None
    if sync:
        from commodity_prediction.cloud import restore_run
        from commodity_prediction.studies.storage import StudyStorage

        with log.stage("restore_domain_checkpoints"):
            restore_run(root, lineage, log)
        storage = StudyStorage(root, log)
    directory = root / "artifacts" / lineage
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "lineage.json", evidence)

    def checkpoint(stage: Path) -> None:
        if storage is not None:
            storage.upload_checkpoint(stage)

    summary_stage = directory / "summary"
    if verify_checkpoint(summary_stage, lineage):
        # Full integrity verification remains mandatory, without rebuilding or refitting.
        with log.stage("verify_complete_domain_resume"):
            count = 0
            for manifest in sorted(directory.rglob("manifest.json")):
                verify_checkpoint(manifest.parent, lineage)
                checkpoint(manifest.parent)
                count += 1
            summary = json.loads((summary_stage / "summary.json").read_text())
            log.event("complete_run_reused", verified_checkpoints=count)
        atomic_json(root / "reports/domain_study.json", summary)
        atomic_json(root / "reports/domain_lineage.json", evidence)
        return summary
    with log.stage("domain_data_contracts"):
        x, y, pairs = load_data(root)
        parent_config = json.loads((root / "configs/research.json").read_text())
        study_config = json.loads((root / "configs/feature_study.json").read_text())
        folds, stop = study_folds(len(x), parent_config, study_config)
        x, y = x.iloc[:stop], y.iloc[:stop]
    with threadpool_limits(limits=config["threads"]):
        stage = directory / "features"
        with log.stage("domain_feature_panel"):
            if verify_checkpoint(stage, lineage):
                panel = load_panel(stage)
                log.event("checkpoint_reused", stage="domain_feature_panel")
            else:
                panel = build_panel(x, y, pairs, log)
                panel.save(stage)
                atomic_json(stage / "inventory.json", panel.inventory())
                seal_checkpoint(stage, lineage, ["panel.npz", "inventory.json"])
            checkpoint(stage)
            log.event(
                "domain_candidates",
                templates=len(panel.names),
                source_series=sum(panel.source_series.values()),
            )
        results = run_experiments(
            panel, y, pairs, folds, config, directory, lineage, log, checkpoint
        )
        permutations = []
        for fold in folds:
            stage = directory / f"fold_{fold.number}/permutation"
            with log.stage(f"fold_{fold.number}/domain_group_permutation"):
                if not verify_checkpoint(stage, lineage):
                    model = joblib.load(
                        directory / f"fold_{fold.number}/outer/pooled_all/model.joblib"
                    )
                    result = group_permutations(
                        model,
                        panel,
                        y.iloc[fold.validation_start : fold.validation_stop],
                        fold.validation_start,
                        fold.validation_stop,
                        config,
                    )
                    atomic_json(stage / "result.json", result)
                    seal_checkpoint(stage, lineage, ["result.json"])
                checkpoint(stage)
                permutations.append(
                    {
                        "fold": fold.number,
                        "families": json.loads((stage / "result.json").read_text()),
                    }
                )
        with log.stage("domain_matched_comparisons"):
            summary = summarize(panel, y, pairs, folds, results, directory, lineage, config)
            summary["group_permutation"] = permutations
            # The strongest established control must agree exactly, without refitting earlier models.
            parent = json.loads((root / "reports/feature_study.json").read_text())
            prior = next(
                r["official_metric"]
                for r in parent["comparison"]
                if r["variant"] == "historical_mean"
            )
            current = summary["summaries"]["historical_mean"]["official_metric"]
            if abs(prior - current) > 1e-12:
                raise ValueError(
                    "Historical mean differs from the preserved matched-window control"
                )
            atomic_json(summary_stage / "summary.json", summary)
            seal_checkpoint(summary_stage, lineage, ["summary.json"])
            checkpoint(summary_stage)
    atomic_json(root / "reports/domain_study.json", summary)
    atomic_json(root / "reports/domain_lineage.json", evidence)
    log.event(
        "domain_study_completed",
        lineage=lineage,
        new_fitted_models=summary["new_fitted_models"],
        feature_gate="open",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--sync-s3", action="store_true")
    args = parser.parse_args()
    run_study(args.root.resolve(), args.sync_s3)


if __name__ == "__main__":
    main()
