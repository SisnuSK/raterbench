Baselines on the eval role (point estimates; bootstrap CIs are added in Phase 4).

| dataset | model | seeds | n | accuracy | macro-F1 | Cohen's kappa | Krippendorff's alpha |
|---|---|---|---|---|---|---|---|
| dreaddit | majority | - | 715 | 0.516 | 0.340 | 0.000 | -0.318 |
| dreaddit | roberta-base-ft | 13,14,15 | 715 | 0.811 ± 0.009 | 0.810 ± 0.009 | 0.621 ± 0.018 | 0.621 ± 0.018 |
| tweet_eval_stance | majority | - | 1000 | 0.573 | 0.243 | 0.000 | -0.192 |
| tweet_eval_stance | roberta-base-ft | 13,14,15 | 1000 | 0.687 ± 0.010 | 0.658 ± 0.006 | 0.485 ± 0.011 | 0.482 ± 0.012 |

Encoder rows show mean ± sample SD across seeds.
