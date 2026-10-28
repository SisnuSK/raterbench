# Paired significance tests (exact McNemar, eval role)

Holm-Bonferroni correction within each dataset's family of tests.

| dataset | a | b | acc_a | acc_b | only_a_correct | only_b_correct | p | p_holm | significant_0.05 |
|---|---|---|---|---|---|---|---|---|---|
| dreaddit | llama3.1-8b/zero_shot | llama3.1-8b/few_shot | 0.744 | 0.736 | 47 | 41 | 0.594 | 1.000 | False |
| dreaddit | llama3.1-8b/zero_shot | llama3.1-8b/codebook | 0.744 | 0.734 | 44 | 37 | 0.505 | 1.000 | False |
| dreaddit | llama3.1-8b/zero_shot | llama3.1-8b/codebook_reasoning | 0.744 | 0.752 | 48 | 54 | 0.621 | 1.000 | False |
| dreaddit | llama3.1-8b/few_shot | llama3.1-8b/codebook | 0.736 | 0.734 | 33 | 32 | 1.000 | 1.000 | False |
| dreaddit | llama3.1-8b/few_shot | llama3.1-8b/codebook_reasoning | 0.736 | 0.752 | 45 | 57 | 0.276 | 1.000 | False |
| dreaddit | llama3.1-8b/codebook | llama3.1-8b/codebook_reasoning | 0.734 | 0.752 | 36 | 49 | 0.193 | 1.000 | False |
| dreaddit | qwen2.5-7b/zero_shot | qwen2.5-7b/few_shot | 0.768 | 0.780 | 40 | 49 | 0.397 | 1.000 | False |
| dreaddit | qwen2.5-7b/zero_shot | qwen2.5-7b/codebook | 0.768 | 0.818 | 55 | 91 | 0.004 | 0.062 | False |
| dreaddit | qwen2.5-7b/zero_shot | qwen2.5-7b/codebook_reasoning | 0.768 | 0.806 | 63 | 90 | 0.035 | 0.460 | False |
| dreaddit | qwen2.5-7b/few_shot | qwen2.5-7b/codebook | 0.780 | 0.818 | 61 | 88 | 0.033 | 0.460 | False |
| dreaddit | qwen2.5-7b/few_shot | qwen2.5-7b/codebook_reasoning | 0.780 | 0.806 | 72 | 90 | 0.181 | 1.000 | False |
| dreaddit | qwen2.5-7b/codebook | qwen2.5-7b/codebook_reasoning | 0.818 | 0.806 | 30 | 21 | 0.262 | 1.000 | False |
| dreaddit | llama3.1-8b/zero_shot | qwen2.5-7b/zero_shot | 0.744 | 0.768 | 51 | 68 | 0.142 | 1.000 | False |
| dreaddit | llama3.1-8b/few_shot | qwen2.5-7b/few_shot | 0.736 | 0.780 | 29 | 61 | 0.001 | 0.018 | True |
| dreaddit | llama3.1-8b/codebook | qwen2.5-7b/codebook | 0.734 | 0.818 | 74 | 134 | 0.000 | 0.001 | True |
| dreaddit | llama3.1-8b/codebook_reasoning | qwen2.5-7b/codebook_reasoning | 0.752 | 0.806 | 72 | 110 | 0.006 | 0.095 | False |
| dreaddit | llama3.1-8b/zero_shot | roberta-base-ft/supervised | 0.744 | 0.821 | 49 | 104 | 0.000 | 0.000 | True |
| dreaddit | llama3.1-8b/few_shot | roberta-base-ft/supervised | 0.736 | 0.821 | 53 | 114 | 0.000 | 0.000 | True |
| dreaddit | llama3.1-8b/codebook | roberta-base-ft/supervised | 0.734 | 0.821 | 56 | 118 | 0.000 | 0.000 | True |
| dreaddit | llama3.1-8b/codebook_reasoning | roberta-base-ft/supervised | 0.752 | 0.821 | 58 | 107 | 0.000 | 0.003 | True |
| dreaddit | qwen2.5-7b/zero_shot | roberta-base-ft/supervised | 0.768 | 0.821 | 57 | 95 | 0.003 | 0.046 | True |
| dreaddit | qwen2.5-7b/few_shot | roberta-base-ft/supervised | 0.780 | 0.821 | 61 | 90 | 0.022 | 0.336 | False |
| dreaddit | qwen2.5-7b/codebook | roberta-base-ft/supervised | 0.818 | 0.821 | 64 | 66 | 0.930 | 1.000 | False |
| dreaddit | qwen2.5-7b/codebook_reasoning | roberta-base-ft/supervised | 0.806 | 0.821 | 65 | 76 | 0.400 | 1.000 | False |
| tweet_eval_stance | llama3.1-8b/zero_shot | llama3.1-8b/few_shot | 0.556 | 0.672 | 47 | 163 | 0.000 | 0.000 | True |
| tweet_eval_stance | llama3.1-8b/zero_shot | llama3.1-8b/codebook | 0.556 | 0.739 | 59 | 242 | 0.000 | 0.000 | True |
| tweet_eval_stance | llama3.1-8b/zero_shot | llama3.1-8b/codebook_reasoning | 0.556 | 0.708 | 134 | 286 | 0.000 | 0.000 | True |
| tweet_eval_stance | llama3.1-8b/few_shot | llama3.1-8b/codebook | 0.672 | 0.739 | 66 | 133 | 0.000 | 0.000 | True |
| tweet_eval_stance | llama3.1-8b/few_shot | llama3.1-8b/codebook_reasoning | 0.672 | 0.708 | 138 | 174 | 0.047 | 0.379 | False |
| tweet_eval_stance | llama3.1-8b/codebook | llama3.1-8b/codebook_reasoning | 0.739 | 0.708 | 123 | 92 | 0.041 | 0.371 | False |
| tweet_eval_stance | qwen2.5-7b/zero_shot | qwen2.5-7b/few_shot | 0.567 | 0.642 | 42 | 117 | 0.000 | 0.000 | True |
| tweet_eval_stance | qwen2.5-7b/zero_shot | qwen2.5-7b/codebook | 0.567 | 0.679 | 47 | 159 | 0.000 | 0.000 | True |
| tweet_eval_stance | qwen2.5-7b/zero_shot | qwen2.5-7b/codebook_reasoning | 0.567 | 0.697 | 82 | 212 | 0.000 | 0.000 | True |
| tweet_eval_stance | qwen2.5-7b/few_shot | qwen2.5-7b/codebook | 0.642 | 0.679 | 68 | 105 | 0.006 | 0.078 | False |
| tweet_eval_stance | qwen2.5-7b/few_shot | qwen2.5-7b/codebook_reasoning | 0.642 | 0.697 | 89 | 144 | 0.000 | 0.005 | True |
| tweet_eval_stance | qwen2.5-7b/codebook | qwen2.5-7b/codebook_reasoning | 0.679 | 0.697 | 72 | 90 | 0.181 | 1.000 | False |
| tweet_eval_stance | llama3.1-8b/zero_shot | qwen2.5-7b/zero_shot | 0.556 | 0.567 | 80 | 91 | 0.445 | 1.000 | False |
| tweet_eval_stance | llama3.1-8b/few_shot | qwen2.5-7b/few_shot | 0.672 | 0.642 | 112 | 82 | 0.037 | 0.371 | False |
| tweet_eval_stance | llama3.1-8b/codebook | qwen2.5-7b/codebook | 0.739 | 0.679 | 133 | 73 | 0.000 | 0.001 | True |
| tweet_eval_stance | llama3.1-8b/codebook_reasoning | qwen2.5-7b/codebook_reasoning | 0.708 | 0.697 | 128 | 117 | 0.523 | 1.000 | False |
| tweet_eval_stance | llama3.1-8b/zero_shot | roberta-base-ft/supervised | 0.556 | 0.694 | 148 | 286 | 0.000 | 0.000 | True |
| tweet_eval_stance | llama3.1-8b/few_shot | roberta-base-ft/supervised | 0.672 | 0.694 | 174 | 196 | 0.275 | 1.000 | False |
| tweet_eval_stance | llama3.1-8b/codebook | roberta-base-ft/supervised | 0.739 | 0.694 | 178 | 133 | 0.012 | 0.137 | False |
| tweet_eval_stance | llama3.1-8b/codebook_reasoning | roberta-base-ft/supervised | 0.708 | 0.694 | 159 | 145 | 0.456 | 1.000 | False |
| tweet_eval_stance | qwen2.5-7b/zero_shot | roberta-base-ft/supervised | 0.567 | 0.694 | 159 | 286 | 0.000 | 0.000 | True |
| tweet_eval_stance | qwen2.5-7b/few_shot | roberta-base-ft/supervised | 0.642 | 0.694 | 166 | 218 | 0.009 | 0.110 | False |
| tweet_eval_stance | qwen2.5-7b/codebook | roberta-base-ft/supervised | 0.679 | 0.694 | 168 | 183 | 0.455 | 1.000 | False |
| tweet_eval_stance | qwen2.5-7b/codebook_reasoning | roberta-base-ft/supervised | 0.697 | 0.694 | 172 | 169 | 0.914 | 1.000 | False |
