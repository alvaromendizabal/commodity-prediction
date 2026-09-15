# Publication copies and historical execution identity

The public repository includes the research implementation and recorded notebook evidence.
The September 14 publication repairs formatting and explicit exception/import/closure
contracts in the manual helpers and tests. The UP022 correction replaces only a
reviewed subprocess.run stdout/stderr PIPE pair with capture_output=True. The PR
message is a bytes literal with unchanged contents. Full configured safe fixes
are applied only to the listed manual publication candidates in scratch. No
blanket unsafe fixes, rule suppression, or weakened quality configuration are used. It does not change frozen `src/` model/feature
modules, model parameters, statistical study declarations, raw data, or saved predictions.
The historical AWS research checkout remains at its recorded source pin.

The pre-publication manual Python bytes are preserved as UTF-8 text under
`docs/history/manual_execution_sources/`, with their original relative paths and hashes in
`reports/manual_research/publication_source_map.json`. The active public Python files are
readable, linted review copies. Historical checkpoint gates intentionally depend on the
original executable bytes: do not overwrite expected hashes to reuse old artifacts.
Use the original source pin plus the archived original helper/test bytes in a SEPARATE
reproduction checkout when reproducing those saved runs; required licensed data and private
checkpoint files are not shipped here. Copy only the archive paths recorded in the map,
verify SHA-256 before copying, and keep the active research environment unchanged.

Publication lint/test success is not a new scientific result. The recorded numerical
results belong to their original execution-source identities, not a claim of private-data
reproduction under reformatted copies. Full quality checks and existing publication
verifiers are retained. No lint-ignore rule or expected historical checksum was relaxed.

The original review notebook outputs were preserved where their source matched the
approved overlay. If the aggregate inputs changed, only `portfolio_overview.ipynb` was
refreshed; research notebooks and trained models were not re-executed. Publication reports
link the original evidence hashes to the approved review bytes.
