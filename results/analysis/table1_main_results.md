# Main results (eval role)

Point estimate [95% percentile bootstrap CI, 1000 resamples over items]. Invalid outputs count as errors.

| dataset | model | condition | n | invalid | macro_f1 | cohen_kappa | krippendorff_alpha | accuracy |
|---|---|---|---|---|---|---|---|---|
| dreaddit | majority | baseline | 715 | 0 | 0.340 [0.324, 0.356] | 0.000 [0.000, 0.000] | -0.318 [-0.351, -0.286] | 0.516 |
| dreaddit | roberta-base-ft (seed 13; 3-seed kappa 0.621±0.018) | supervised | 715 | 0 | 0.820 [0.791, 0.847] | 0.641 [0.582, 0.695] | 0.641 [0.582, 0.695] | 0.821 |
| dreaddit | llama3.1-8b | zero_shot | 715 | 0 | 0.734 [0.701, 0.766] | 0.482 [0.423, 0.541] | 0.467 [0.402, 0.532] | 0.744 |
| dreaddit | llama3.1-8b | few_shot | 715 | 0 | 0.718 [0.682, 0.751] | 0.463 [0.403, 0.524] | 0.437 [0.364, 0.503] | 0.736 |
| dreaddit | llama3.1-8b | codebook | 715 | 0 | 0.715 [0.682, 0.748] | 0.460 [0.406, 0.518] | 0.431 [0.364, 0.496] | 0.734 |
| dreaddit | llama3.1-8b | codebook_reasoning | 715 | 0 | 0.743 [0.709, 0.774] | 0.499 [0.436, 0.558] | 0.486 [0.419, 0.549] | 0.752 |
| dreaddit | qwen2.5-7b | zero_shot | 715 | 0 | 0.760 [0.728, 0.791] | 0.531 [0.470, 0.588] | 0.521 [0.457, 0.583] | 0.768 |
| dreaddit | qwen2.5-7b | few_shot | 715 | 0 | 0.774 [0.742, 0.803] | 0.556 [0.497, 0.612] | 0.548 [0.484, 0.605] | 0.780 |
| dreaddit | qwen2.5-7b | codebook | 715 | 0 | 0.818 [0.789, 0.843] | 0.637 [0.581, 0.687] | 0.636 [0.579, 0.687] | 0.818 |
| dreaddit | qwen2.5-7b | codebook_reasoning | 715 | 0 | 0.806 [0.776, 0.833] | 0.612 [0.553, 0.667] | 0.611 [0.552, 0.667] | 0.806 |
| tweet_eval_stance | majority | baseline | 1000 | 0 | 0.243 [0.234, 0.251] | 0.000 [0.000, 0.000] | -0.192 [-0.210, -0.175] | 0.573 |
| tweet_eval_stance | roberta-base-ft (seed 13; 3-seed kappa 0.485±0.011) | supervised | 1000 | 0 | 0.658 [0.623, 0.688] | 0.487 [0.436, 0.532] | 0.486 [0.434, 0.531] | 0.694 |
| tweet_eval_stance | llama3.1-8b | zero_shot | 1000 | 0 | 0.538 [0.505, 0.569] | 0.344 [0.306, 0.382] | 0.296 [0.247, 0.343] | 0.556 |
| tweet_eval_stance | llama3.1-8b | few_shot | 1000 | 0 | 0.658 [0.628, 0.687] | 0.490 [0.445, 0.531] | 0.479 [0.433, 0.522] | 0.672 |
| tweet_eval_stance | llama3.1-8b | codebook | 1000 | 0 | 0.706 [0.675, 0.736] | 0.563 [0.518, 0.606] | 0.562 [0.517, 0.606] | 0.739 |
| tweet_eval_stance | llama3.1-8b | codebook_reasoning | 1000 | 1 | 0.645 [0.610, 0.679] | 0.506 [0.456, 0.548] | 0.500 [0.448, 0.544] | 0.708 |
| tweet_eval_stance | qwen2.5-7b | zero_shot | 1000 | 0 | 0.530 [0.498, 0.561] | 0.336 [0.296, 0.375] | 0.299 [0.252, 0.345] | 0.567 |
| tweet_eval_stance | qwen2.5-7b | few_shot | 1000 | 0 | 0.629 [0.596, 0.658] | 0.441 [0.395, 0.484] | 0.426 [0.376, 0.472] | 0.642 |
| tweet_eval_stance | qwen2.5-7b | codebook | 1000 | 0 | 0.637 [0.604, 0.667] | 0.459 [0.414, 0.501] | 0.455 [0.409, 0.497] | 0.679 |
| tweet_eval_stance | qwen2.5-7b | codebook_reasoning | 1000 | 0 | 0.663 [0.630, 0.692] | 0.494 [0.447, 0.537] | 0.494 [0.446, 0.536] | 0.697 |
