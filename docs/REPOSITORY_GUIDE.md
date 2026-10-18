# Repository guide

The public review has three entry points: [Forecast Lab](../public-demo/index.html) for the working demonstration, [the case study](PORTFOLIO_CASE_STUDY.md) for the research decisions, and [the reproduction guide](REPRODUCIBILITY.md) for executable checks. Start with those before entering individual historical studies.

| Directory | Maintained purpose |
|---|---|
| `public-demo/` | Authored data generator, actual rolling ridge fits, release guards, error diagnostics and browser controls |
| `examples/` | Small Python example of delayed-label access |
| `src/commodity_prediction/` | Public research implementations, feature contracts, evaluation and checkpoint storage |
| `tests/`, `tools/` | Research regression checks, browser behavior, export replay and publication checks |
| `reports/` | Aggregate scientific evidence, figures and source identities |
| `notebooks/` | Executed analyses tied to retained research evidence |
| `configs/`, `scripts/` | Frozen study settings, reproducible entry points and publication tools |
| `docs/history/` | Historical source required by the byte-level publication lineage map |
| `review/` | One reproducible technical brief and its source-bound verification runner |

## Closeout cleanup

The obsolete `feature-research-status.md` was removed because it described open gates and pending experiments superseded by the completed closeout. A workflow step applicable only to already-completed PR 7 was also removed. The current quality, evidence, notebook and publication gates remain active.

Historical source snapshots are retained where [the source map](../reports/manual_research/publication_source_map.json) and publication checks use their bytes. A dated experiment is evidence, not automatically dead code. Git history retains the deleted status and one-off workflow block.

Private datasets, fitted research weights, prediction matrices and current research recipes are outside this public review. [Publication scope](PUBLICATION_SCOPE.md) records the boundary and historical source credits.
