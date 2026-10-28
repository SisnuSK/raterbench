# Subgroup analysis (eval role)

Per-subgroup macro-F1 [95% bootstrap CI]. chi2_p tests whether error rates differ across subgroups. Subgroups differ in size and label mix, so gaps are not purely model bias.

| dataset | view | model | condition | subgroup | n | macro_f1 | error_rate | max_f1_gap | chi2_p_error_rates |
|---|---|---|---|---|---|---|---|---|---|
| dreaddit | domain | llama3.1-8b | codebook | abuse | 142 | 0.622 [0.535, 0.711] | 0.317 | 0.176 | 0.001 |
| dreaddit | domain | llama3.1-8b | codebook | anxiety | 161 | 0.777 [0.699, 0.844] | 0.199 | 0.176 | 0.001 |
| dreaddit | domain | llama3.1-8b | codebook | financial | 143 | 0.713 [0.636, 0.783] | 0.287 | 0.176 | 0.001 |
| dreaddit | domain | llama3.1-8b | codebook | ptsd | 127 | 0.798 [0.715, 0.877] | 0.165 | 0.176 | 0.001 |
| dreaddit | domain | llama3.1-8b | codebook | social | 142 | 0.634 [0.556, 0.710] | 0.359 | 0.176 | 0.001 |
| dreaddit | domain | qwen2.5-7b | codebook | abuse | 142 | 0.796 [0.723, 0.859] | 0.204 | 0.156 | 0.020 |
| dreaddit | domain | qwen2.5-7b | codebook | anxiety | 161 | 0.835 [0.776, 0.887] | 0.161 | 0.156 | 0.020 |
| dreaddit | domain | qwen2.5-7b | codebook | financial | 143 | 0.860 [0.796, 0.915] | 0.133 | 0.156 | 0.020 |
| dreaddit | domain | qwen2.5-7b | codebook | ptsd | 127 | 0.848 [0.778, 0.907] | 0.142 | 0.156 | 0.020 |
| dreaddit | domain | qwen2.5-7b | codebook | social | 142 | 0.704 [0.624, 0.778] | 0.268 | 0.156 | 0.020 |
| dreaddit | domain | roberta-base-ft | supervised | abuse | 142 | 0.799 [0.726, 0.865] | 0.197 | 0.117 | 0.069 |
| dreaddit | domain | roberta-base-ft | supervised | anxiety | 161 | 0.855 [0.796, 0.910] | 0.137 | 0.117 | 0.069 |
| dreaddit | domain | roberta-base-ft | supervised | financial | 143 | 0.831 [0.765, 0.894] | 0.161 | 0.117 | 0.069 |
| dreaddit | domain | roberta-base-ft | supervised | ptsd | 127 | 0.836 [0.758, 0.903] | 0.150 | 0.117 | 0.069 |
| dreaddit | domain | roberta-base-ft | supervised | social | 142 | 0.738 [0.660, 0.809] | 0.254 | 0.117 | 0.069 |
| dreaddit | subreddit | llama3.1-8b | codebook | almosthomeless | 19 | 0.808 [0.552, 1.000] | 0.158 | 0.315 | 0.000 |
| dreaddit | subreddit | llama3.1-8b | codebook | anxiety | 147 | 0.781 [0.703, 0.849] | 0.197 | 0.315 | 0.000 |
| dreaddit | subreddit | llama3.1-8b | codebook | assistance | 66 | 0.707 [0.587, 0.812] | 0.288 | 0.315 | 0.000 |
| dreaddit | subreddit | llama3.1-8b | codebook | domesticviolence | 72 | 0.741 [0.607, 0.852] | 0.194 | 0.315 | 0.000 |
| dreaddit | subreddit | llama3.1-8b | codebook | food_pantry | 6 | 0.829 [0.400, 1.000] | 0.167 | 0.315 | 0.000 |
| dreaddit | subreddit | llama3.1-8b | codebook | homeless | 52 | 0.653 [0.532, 0.785] | 0.346 | 0.315 | 0.000 |
| dreaddit | subreddit | llama3.1-8b | codebook | ptsd | 127 | 0.798 [0.715, 0.877] | 0.165 | 0.315 | 0.000 |
| dreaddit | subreddit | llama3.1-8b | codebook | relationships | 142 | 0.634 [0.556, 0.710] | 0.359 | 0.315 | 0.000 |
| dreaddit | subreddit | llama3.1-8b | codebook | stress | 14 | 0.714 [0.391, 1.000] | 0.214 | 0.315 | 0.000 |
| dreaddit | subreddit | llama3.1-8b | codebook | survivorsofabuse | 70 | 0.513 [0.394, 0.626] | 0.443 | 0.315 | 0.000 |
| dreaddit | subreddit | qwen2.5-7b | codebook | almosthomeless | 19 | 0.825 [0.593, 1.000] | 0.158 | 0.211 | 0.124 |
| dreaddit | subreddit | qwen2.5-7b | codebook | anxiety | 147 | 0.834 [0.770, 0.896] | 0.163 | 0.211 | 0.124 |
| dreaddit | subreddit | qwen2.5-7b | codebook | assistance | 66 | 0.810 [0.693, 0.903] | 0.167 | 0.211 | 0.124 |
| dreaddit | subreddit | qwen2.5-7b | codebook | domesticviolence | 72 | 0.811 [0.714, 0.896] | 0.181 | 0.211 | 0.124 |
| dreaddit | subreddit | qwen2.5-7b | codebook | food_pantry | 6 | 0.829 [0.400, 1.000] | 0.167 | 0.211 | 0.124 |
| dreaddit | subreddit | qwen2.5-7b | codebook | homeless | 52 | 0.915 [0.830, 0.980] | 0.077 | 0.211 | 0.124 |
| dreaddit | subreddit | qwen2.5-7b | codebook | ptsd | 127 | 0.848 [0.778, 0.907] | 0.142 | 0.211 | 0.124 |
| dreaddit | subreddit | qwen2.5-7b | codebook | relationships | 142 | 0.704 [0.624, 0.778] | 0.268 | 0.211 | 0.124 |
| dreaddit | subreddit | qwen2.5-7b | codebook | stress | 14 | 0.825 [0.476, 1.000] | 0.143 | 0.211 | 0.124 |
| dreaddit | subreddit | qwen2.5-7b | codebook | survivorsofabuse | 70 | 0.762 [0.656, 0.863] | 0.229 | 0.211 | 0.124 |
| dreaddit | subreddit | roberta-base-ft | supervised | almosthomeless | 19 | 0.676 [0.424, 0.842] | 0.316 | 0.219 | 0.083 |
| dreaddit | subreddit | roberta-base-ft | supervised | anxiety | 147 | 0.857 [0.795, 0.911] | 0.136 | 0.219 | 0.083 |
| dreaddit | subreddit | roberta-base-ft | supervised | assistance | 66 | 0.895 [0.802, 0.964] | 0.091 | 0.219 | 0.083 |
| dreaddit | subreddit | roberta-base-ft | supervised | domesticviolence | 72 | 0.816 [0.717, 0.904] | 0.167 | 0.219 | 0.083 |
| dreaddit | subreddit | roberta-base-ft | supervised | food_pantry | 6 | 0.829 [0.400, 1.000] | 0.167 | 0.219 | 0.083 |
| dreaddit | subreddit | roberta-base-ft | supervised | homeless | 52 | 0.797 [0.677, 0.902] | 0.192 | 0.219 | 0.083 |
| dreaddit | subreddit | roberta-base-ft | supervised | ptsd | 127 | 0.836 [0.758, 0.903] | 0.150 | 0.219 | 0.083 |
| dreaddit | subreddit | roberta-base-ft | supervised | relationships | 142 | 0.738 [0.660, 0.809] | 0.254 | 0.219 | 0.083 |
| dreaddit | subreddit | roberta-base-ft | supervised | stress | 14 | 0.825 [0.476, 1.000] | 0.143 | 0.219 | 0.083 |
| dreaddit | subreddit | roberta-base-ft | supervised | survivorsofabuse | 70 | 0.771 [0.663, 0.857] | 0.229 | 0.219 | 0.083 |
| tweet_eval_stance | target | llama3.1-8b | codebook | abortion | 224 | 0.652 [0.578, 0.713] | 0.317 | 0.137 | 0.111 |
| tweet_eval_stance | target | llama3.1-8b | codebook | atheism | 176 | 0.627 [0.527, 0.710] | 0.261 | 0.137 | 0.111 |
| tweet_eval_stance | target | llama3.1-8b | codebook | climate | 136 | 0.665 [0.562, 0.748] | 0.279 | 0.137 | 0.111 |
| tweet_eval_stance | target | llama3.1-8b | codebook | feminist | 228 | 0.708 [0.640, 0.767] | 0.250 | 0.137 | 0.111 |
| tweet_eval_stance | target | llama3.1-8b | codebook | hillary | 236 | 0.764 [0.703, 0.821] | 0.208 | 0.137 | 0.111 |
| tweet_eval_stance | target | qwen2.5-7b | codebook | abortion | 224 | 0.622 [0.541, 0.695] | 0.304 | 0.227 | 0.050 |
| tweet_eval_stance | target | qwen2.5-7b | codebook | atheism | 176 | 0.469 [0.381, 0.551] | 0.386 | 0.227 | 0.050 |
| tweet_eval_stance | target | qwen2.5-7b | codebook | climate | 136 | 0.575 [0.471, 0.661] | 0.382 | 0.227 | 0.050 |
| tweet_eval_stance | target | qwen2.5-7b | codebook | feminist | 228 | 0.632 [0.557, 0.696] | 0.307 | 0.227 | 0.050 |
| tweet_eval_stance | target | qwen2.5-7b | codebook | hillary | 236 | 0.696 [0.626, 0.755] | 0.267 | 0.227 | 0.050 |
| tweet_eval_stance | target | roberta-base-ft | supervised | abortion | 224 | 0.611 [0.534, 0.679] | 0.317 | 0.093 | 0.000 |
| tweet_eval_stance | target | roberta-base-ft | supervised | atheism | 176 | 0.521 [0.427, 0.599] | 0.307 | 0.093 | 0.000 |
| tweet_eval_stance | target | roberta-base-ft | supervised | climate | 136 | 0.598 [0.501, 0.708] | 0.162 | 0.093 | 0.000 |
| tweet_eval_stance | target | roberta-base-ft | supervised | feminist | 228 | 0.591 [0.517, 0.656] | 0.386 | 0.093 | 0.000 |
| tweet_eval_stance | target | roberta-base-ft | supervised | hillary | 236 | 0.614 [0.542, 0.681] | 0.301 | 0.093 | 0.000 |
