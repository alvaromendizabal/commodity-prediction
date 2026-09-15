# Manual research results

Scores remain grouped by their own evaluation period; no cross-period ranking is implied.

## 03 — Runtime and feature readiness
Status: **NOTEBOOK_AND_FEATURE_AUDIT_READY**. Evaluation origins: **0**.

Decision: `not recorded`. Source report SHA-256: `2c16ec7dca306e01b7af33c0c90169d290bb19e8456f8f1551fb93d170d8a54c`.

## 04 — Normalization first-period screen
Status: **NOTEBOOK_AND_FIRST_FOLD_REVIEW_READY**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| normalized_price | 0.407167 | +0.003786 |
| volume_confirmation | 0.404538 | +0.001157 |
| normalized_joint | 0.385727 | -0.017654 |

Decision: `REVIEW_BEFORE_REMAINING_FOLDS`. Source report SHA-256: `19e0c7c14d3bdf01c244ccd573e94b240f95eb1673dba16b631cfc9775ae1815`.

## 05 — Normalization temporal validation
Status: **NOTEBOOK_AND_VALIDATION_REVIEW_READY**. Evaluation origins: **535**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| normalized_price | 0.305733 | -0.003975 |
| volume_confirmation | 0.306911 | -0.002798 |
| normalized_joint | 0.296970 | -0.012738 |

Decision: `NO_STABLE_GAIN_IN_THIS_FAMILY`. Source report SHA-256: `7c8a7984f87c7ba8a48472194128a6f0173d58903f25c03f6b10c8dfa72dd14a`.

## 06 — Session candidate laboratory
Status: **NOTEBOOK_AND_SESSION_FEATURES_READY**. Evaluation origins: **0**.

Decision: `not recorded`. Source report SHA-256: `b6185ed15a0db01efa15aa70ae1d04e40ad6a6cdd7efb3a26f5550f8b391f50d`.

## 07 — Session feature ablations
Status: **NOTEBOOK_AND_SESSION_ABLATION_READY**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| session_support_only | 0.403381 | +0.000000 |
| session_intraday | 0.397112 | -0.006269 |
| session_overnight_risk | 0.401594 | -0.001787 |
| session_joint | 0.375454 | -0.027927 |

Decision: `STOP_FITTED_SESSION_FORMULATION`. Source report SHA-256: `e56c8ca7c1da5f77faa6a7544ed1fe26f25814173e423853e79538f2f12ceeff`.

## 08 — Cross-asset peer features
Status: **NOTEBOOK_AND_CLOSE_NETWORK_READY**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| network_availability | 0.372697 | -0.030685 |
| network_synchronous | 0.374725 | -0.028656 |
| network_directed | 0.363208 | -0.040173 |
| network_joint | 0.363892 | -0.039489 |

Decision: `STOP_THIS_NETWORK_FORMULATION`. Source report SHA-256: `3131f1a96e6a7a9f3ca90ad0de0aa7b6b3f6ee685588af327f8ead9346fa0bb1`.

## 09 — Network removal and rank laboratory
Status: **NOTEBOOK_AND_DIAGNOSIS_READY**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| numeric_synchronous | 0.377545 | -0.025836 |
| numeric_directed | 0.399654 | -0.003727 |

Decision: `STOP_TESTED_NETWORK_VARIANTS`. Source report SHA-256: `7003517459b1fbb71c1dfe6e131de47ce1d7c36243d324a0c33166f6cb727c48`.

## 10 — Released historical-rank states
Status: **NOTEBOOK_AND_RANK_STATE_READY**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| rank_global | 0.390577 | -0.012804 |
| rank_horizon_relative | 0.399073 | -0.004308 |
| rank_joint | 0.388567 | -0.014814 |

Decision: `STOP_TESTED_RANK_STATE_PANELS`. Source report SHA-256: `9eae64662af41f6fb9409cd96faf2414ca6ae219f75e2d46556263e5dc356fec`.

## 11 — Instrument context
Status: **NOTEBOOK_AND_TARGET_CONTEXT_READY**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| context_identity | 0.153349 | -0.013693 |
| context_shocks | 0.133649 | -0.033393 |
| context_joint | 0.128888 | -0.038154 |

Decision: `STOP_TESTED_TARGET_CONTEXT_PANELS`. Source report SHA-256: `7a577e40c4c0c87de13da8de573e529214220fd5f5b46bae9ee8c42950f33bf5`.

## 12 — Delayed response encoding
Status: **NOTEBOOK_AND_RESPONSE_READY**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| response_drivers | 0.154426 | -0.012616 |
| response_encoded | 0.165238 | -0.001804 |
| response_joint | 0.158861 | -0.008181 |

Decision: `STOP_TESTED_RESPONSE_PANELS`. Source report SHA-256: `7e5fc8b133129149d2d2332e351889bd9c1d2a45b39c0a8670408fd72aea77af`.

## 13 — Saved-model information audit
Status: **NOTEBOOK_AND_INFORMATION_AUDIT_READY**. Evaluation origins: **535**.

Decision: `REVIEW_RELIANCE_AND_CANDIDATES_BEFORE_FITS`. Source report SHA-256: `01f5cef137cc0527b4fd59b9e56b1c31e0258913d85534f3d318f4bfc3778bee`.

## 14 — Ordinary and robust innovations
Status: **NOTEBOOK_AND_FEATURE_ROUND_READY**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| audit_shortlist | 0.161291 | -0.005750 |
| ordinary_innovations | 0.147183 | -0.019858 |
| robust_innovations | 0.146112 | -0.020930 |
| innovation_joint | 0.125402 | -0.041640 |

Decision: `STOP_TESTED_PANELS`. Source report SHA-256: `8c3ef8a7a0989dab4cab6253c4ad9e8f141201fe3710e03b70abc3e6accc1fd6`.

## 15 — Prior dynamics screening
Status: **NOTEBOOK_AND_FEATURE_ROUND_READY**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| risk_dynamics_only | 0.157692 | -0.009349 |
| own_history_dynamics | 0.163633 | -0.003409 |
| pool_and_risk_dynamics | 0.144366 | -0.022675 |
| prior_dynamics_joint | 0.170736 | +0.003694 |

Decision: `REVIEW_ON_OTHER_PERIODS`. Source report SHA-256: `d62a3d88b1c336d40127011755236d094c75dbe2d4a077389a276fc947fb0a03`.

## 16 — Prior dynamics temporal replication
Status: **NOTEBOOK_COMPLETE**. Evaluation origins: **535**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| risk_dynamics_only | 0.300334 | -0.009375 |
| own_history_dynamics | 0.294642 | -0.015067 |
| pool_and_risk_dynamics | 0.296007 | -0.013702 |
| prior_dynamics_joint | 0.303614 | -0.006095 |

Decision: `NO_STABLE_PRIOR_DYNAMICS_GAIN`. Source report SHA-256: `48faf0c54e4f95a6e4f6f2c7c0416edc72795a80a4b9dccb0a12c52126768364`.

## 17 — Observed-event histories
Status: **NOTEBOOK_COMPLETE**. Evaluation origins: **180**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| event_clock_only | 0.378182 | -0.025199 |
| event_location | 0.322297 | -0.081085 |
| event_shape | 0.359661 | -0.043720 |
| event_joint | 0.307131 | -0.096250 |

Decision: `STOP_TESTED_EVENT_HISTORY_PANELS`. Source report SHA-256: `32485001c39595e87ede0151cbb597b15d2e96dd2ec0b0d1db059411419b575b`.

## 18 — Released sequence structure
Status: **NOTEBOOK_COMPLETE**. Evaluation origins: **175**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| sequence_clock | 0.389913 | -0.000706 |
| sequence_magnitude | 0.391716 | +0.001097 |
| sequence_transitions | 0.389913 | -0.000706 |
| sequence_joint | 0.395621 | +0.005002 |

Decision: `REVIEW_ON_OTHER_PERIODS`. Source report SHA-256: `96dc161cbd35a8d3633dd22802c7c2bb6365954b05de30b20f6f47e2294dcc21`.

## 19 — As-of prior-error memory
Status: **NOTEBOOK_COMPLETE**. Evaluation origins: **175**.

| Representation | Metric | Matched baseline change |
|---|---:|---:|
| error_clock | 0.387301 | -0.003318 |
| error_bias | 0.396322 | +0.005703 |
| error_dispersion | 0.396463 | +0.005844 |
| error_joint | 0.377076 | -0.013543 |

Decision: `REVIEW_ON_OTHER_PERIODS`. Source report SHA-256: `8de2d6013733f51042556a8c3843c375643478d05c5982819c06e027b131cc17`.
