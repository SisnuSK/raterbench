# Calibration of confidence signals (eval role)

ECE (10 equal-width bins) and AUROC with 95% bootstrap CIs; Brier score of confidence as a forecast of correctness; AURC = area under the risk-coverage curve (lower is better). Self-consistency = share of the 5 T=0.7 samples that agree with the T=0 label.

| dataset | model | condition | signal | n_with_signal | n_distinct_values | ece | brier | auroc | aurc |
|---|---|---|---|---|---|---|---|---|---|
| dreaddit | llama3.1-8b | zero_shot | verbalized | 715 | 4 | 0.132 [0.106, 0.161] | 0.185 | 0.774 [0.738, 0.805] | 0.117 |
| dreaddit | llama3.1-8b | zero_shot | logprob | 715 | 653 | 0.194 [0.164, 0.222] | 0.210 | 0.762 [0.725, 0.797] | 0.114 |
| dreaddit | llama3.1-8b | few_shot | verbalized | 715 | 5 | 0.135 [0.105, 0.167] | 0.193 | 0.723 [0.688, 0.757] | 0.141 |
| dreaddit | llama3.1-8b | few_shot | logprob | 715 | 653 | 0.211 [0.182, 0.244] | 0.228 | 0.724 [0.683, 0.763] | 0.130 |
| dreaddit | llama3.1-8b | codebook | verbalized | 715 | 4 | 0.165 [0.134, 0.194] | 0.202 | 0.769 [0.733, 0.801] | 0.125 |
| dreaddit | llama3.1-8b | codebook | logprob | 715 | 663 | 0.221 [0.192, 0.253] | 0.233 | 0.752 [0.716, 0.788] | 0.122 |
| dreaddit | llama3.1-8b | codebook | self_consistency | 715 | 4 | 0.239 [0.207, 0.269] | 0.246 | 0.562 [0.532, 0.591] | 0.243 |
| dreaddit | llama3.1-8b | codebook_reasoning | verbalized | 715 | 5 | 0.138 [0.108, 0.168] | 0.198 | 0.632 [0.596, 0.666] | 0.174 |
| dreaddit | llama3.1-8b | codebook_reasoning | logprob | 715 | 421 | 0.245 [0.215, 0.277] | 0.244 | 0.694 [0.650, 0.739] | 0.152 |
| dreaddit | qwen2.5-7b | zero_shot | verbalized | 715 | 4 | 0.086 [0.058, 0.116] | 0.176 | 0.643 [0.607, 0.678] | 0.155 |
| dreaddit | qwen2.5-7b | zero_shot | logprob | 715 | 401 | 0.211 [0.185, 0.244] | 0.216 | 0.720 [0.675, 0.762] | 0.125 |
| dreaddit | qwen2.5-7b | few_shot | verbalized | 715 | 5 | 0.078 [0.051, 0.108] | 0.175 | 0.554 [0.527, 0.581] | 0.186 |
| dreaddit | qwen2.5-7b | few_shot | logprob | 715 | 402 | 0.203 [0.176, 0.233] | 0.208 | 0.702 [0.657, 0.745] | 0.122 |
| dreaddit | qwen2.5-7b | codebook | verbalized | 715 | 3 | 0.048 [0.022, 0.076] | 0.148 | 0.550 [0.538, 0.563] | 0.148 |
| dreaddit | qwen2.5-7b | codebook | logprob | 715 | 468 | 0.159 [0.137, 0.187] | 0.166 | 0.801 [0.761, 0.840] | 0.068 |
| dreaddit | qwen2.5-7b | codebook | self_consistency | 715 | 4 | 0.165 [0.140, 0.191] | 0.169 | 0.565 [0.533, 0.599] | 0.169 |
| dreaddit | qwen2.5-7b | codebook_reasoning | verbalized | 715 | 6 | 0.081 [0.053, 0.110] | 0.155 | 0.645 [0.618, 0.671] | 0.121 |
| dreaddit | qwen2.5-7b | codebook_reasoning | logprob | 715 | 88 | 0.195 [0.166, 0.224] | 0.195 | 0.673 [0.625, 0.720] | 0.124 |
| dreaddit | roberta-base-ft | supervised | softmax | 715 | 310 | 0.139 [0.115, 0.168] | 0.156 | 0.759 [0.715, 0.798] | 0.072 |
| tweet_eval_stance | llama3.1-8b | zero_shot | verbalized | 1000 | 5 | 0.330 [0.299, 0.363] | 0.369 | 0.480 [0.446, 0.512] | 0.513 |
| tweet_eval_stance | llama3.1-8b | zero_shot | logprob | 1000 | 981 | 0.353 [0.323, 0.385] | 0.367 | 0.619 [0.582, 0.654] | 0.359 |
| tweet_eval_stance | llama3.1-8b | few_shot | verbalized | 1000 | 4 | 0.160 [0.131, 0.188] | 0.237 | 0.615 [0.590, 0.639] | 0.265 |
| tweet_eval_stance | llama3.1-8b | few_shot | logprob | 1000 | 993 | 0.199 [0.175, 0.228] | 0.243 | 0.698 [0.664, 0.731] | 0.196 |
| tweet_eval_stance | llama3.1-8b | codebook | verbalized | 1000 | 6 | 0.110 [0.082, 0.137] | 0.208 | 0.522 [0.484, 0.560] | 0.285 |
| tweet_eval_stance | llama3.1-8b | codebook | logprob | 1000 | 991 | 0.171 [0.150, 0.200] | 0.212 | 0.694 [0.658, 0.729] | 0.143 |
| tweet_eval_stance | llama3.1-8b | codebook | self_consistency | 1000 | 5 | 0.216 [0.190, 0.242] | 0.232 | 0.582 [0.553, 0.615] | 0.272 |
| tweet_eval_stance | llama3.1-8b | codebook_reasoning | verbalized | 999 | 6 | 0.164 [0.139, 0.194] | 0.222 | 0.657 [0.624, 0.689] | 0.229 |
| tweet_eval_stance | llama3.1-8b | codebook_reasoning | logprob | 999 | 575 | 0.283 [0.256, 0.314] | 0.282 | 0.740 [0.704, 0.773] | 0.180 |
| tweet_eval_stance | qwen2.5-7b | zero_shot | verbalized | 1000 | 5 | 0.282 [0.254, 0.313] | 0.325 | 0.506 [0.491, 0.521] | 0.459 |
| tweet_eval_stance | qwen2.5-7b | zero_shot | logprob | 1000 | 602 | 0.400 [0.369, 0.429] | 0.401 | 0.608 [0.574, 0.644] | 0.375 |
| tweet_eval_stance | qwen2.5-7b | few_shot | verbalized | 1000 | 5 | 0.209 [0.181, 0.240] | 0.273 | 0.512 [0.497, 0.526] | 0.405 |
| tweet_eval_stance | qwen2.5-7b | few_shot | logprob | 1000 | 644 | 0.329 [0.303, 0.359] | 0.331 | 0.692 [0.657, 0.722] | 0.248 |
| tweet_eval_stance | qwen2.5-7b | codebook | verbalized | 1000 | 3 | 0.178 [0.152, 0.207] | 0.248 | 0.525 [0.511, 0.539] | 0.301 |
| tweet_eval_stance | qwen2.5-7b | codebook | logprob | 1000 | 662 | 0.289 [0.261, 0.318] | 0.295 | 0.713 [0.679, 0.743] | 0.185 |
| tweet_eval_stance | qwen2.5-7b | codebook | self_consistency | 1000 | 5 | 0.298 [0.271, 0.326] | 0.303 | 0.542 [0.521, 0.564] | 0.321 |
| tweet_eval_stance | qwen2.5-7b | codebook_reasoning | verbalized | 1000 | 9 | 0.179 [0.153, 0.208] | 0.237 | 0.590 [0.563, 0.617] | 0.254 |
| tweet_eval_stance | qwen2.5-7b | codebook_reasoning | logprob | 1000 | 203 | 0.299 [0.272, 0.329] | 0.299 | 0.568 [0.526, 0.611] | 0.295 |
| tweet_eval_stance | roberta-base-ft | supervised | softmax | 1000 | 995 | 0.167 [0.144, 0.199] | 0.222 | 0.724 [0.694, 0.754] | 0.153 |
