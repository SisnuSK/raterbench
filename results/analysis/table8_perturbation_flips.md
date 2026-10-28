# Label flip rates under meaning-preserving perturbations (T=0)

flip_rate = share of items whose label changes after the edit [95% bootstrap CI]. Read alongside the manual spot check (data/perturbations/spot_check.md).

| dataset | model | condition | perturbation | n | flip_rate | acc_original | acc_perturbed | invalid_perturbed |
|---|---|---|---|---|---|---|---|---|
| dreaddit | llama3.1-8b | codebook | gender_swap | 120 | 0.017 [0.000, 0.042] | 0.692 | 0.692 | 0 |
| dreaddit | llama3.1-8b | codebook | paraphrase | 203 | 0.084 [0.049, 0.123] | 0.739 | 0.773 | 0 |
| dreaddit | llama3.1-8b | codebook | typo | 206 | 0.053 [0.024, 0.083] | 0.743 | 0.709 | 0 |
| dreaddit | llama3.1-8b | zero_shot | gender_swap | 120 | 0.033 [0.008, 0.067] | 0.725 | 0.692 | 0 |
| dreaddit | llama3.1-8b | zero_shot | paraphrase | 203 | 0.084 [0.049, 0.128] | 0.773 | 0.798 | 0 |
| dreaddit | llama3.1-8b | zero_shot | typo | 206 | 0.058 [0.029, 0.092] | 0.777 | 0.738 | 0 |
| dreaddit | qwen2.5-7b | codebook | gender_swap | 120 | 0.025 [0.000, 0.058] | 0.825 | 0.833 | 0 |
| dreaddit | qwen2.5-7b | codebook | paraphrase | 203 | 0.069 [0.039, 0.103] | 0.818 | 0.808 | 0 |
| dreaddit | qwen2.5-7b | codebook | typo | 206 | 0.029 [0.010, 0.053] | 0.820 | 0.820 | 0 |
| dreaddit | qwen2.5-7b | zero_shot | gender_swap | 120 | 0.033 [0.008, 0.067] | 0.733 | 0.750 | 0 |
| dreaddit | qwen2.5-7b | zero_shot | paraphrase | 203 | 0.079 [0.044, 0.118] | 0.749 | 0.768 | 0 |
| dreaddit | qwen2.5-7b | zero_shot | typo | 206 | 0.039 [0.015, 0.068] | 0.752 | 0.743 | 0 |
| tweet_eval_stance | llama3.1-8b | codebook | paraphrase | 200 | 0.085 [0.050, 0.125] | 0.755 | 0.745 | 0 |
| tweet_eval_stance | llama3.1-8b | codebook | typo | 200 | 0.065 [0.035, 0.100] | 0.755 | 0.755 | 0 |
| tweet_eval_stance | llama3.1-8b | zero_shot | paraphrase | 200 | 0.080 [0.045, 0.115] | 0.595 | 0.580 | 0 |
| tweet_eval_stance | llama3.1-8b | zero_shot | typo | 200 | 0.080 [0.045, 0.120] | 0.595 | 0.590 | 0 |
| tweet_eval_stance | qwen2.5-7b | codebook | paraphrase | 200 | 0.075 [0.040, 0.115] | 0.715 | 0.700 | 0 |
| tweet_eval_stance | qwen2.5-7b | codebook | typo | 200 | 0.055 [0.025, 0.090] | 0.715 | 0.715 | 0 |
| tweet_eval_stance | qwen2.5-7b | zero_shot | paraphrase | 200 | 0.090 [0.055, 0.130] | 0.610 | 0.580 | 0 |
| tweet_eval_stance | qwen2.5-7b | zero_shot | typo | 200 | 0.050 [0.025, 0.085] | 0.610 | 0.625 | 0 |
