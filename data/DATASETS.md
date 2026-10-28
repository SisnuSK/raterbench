# Datasets

The raw data isn't in this repo. `raterbench data download` fetches it into `data/raw/` (which is
git-ignored) and checks every file against a SHA-256 pinned in `src/raterbench/data/download.py`.
What is committed is `data/samples/*.jsonl`: the item IDs, which role each item plays, its
subgroup and a hash of its text. No text or labels are redistributed.

Every count below comes from `raterbench data stats` on the pinned files, and that command fails
if any count disagrees with what the original source reports. I last checked them on 2026-10-27.

---

## Dreaddit (stress)

| | |
|---|---|
| Paper | Turcan & McKeown (2019), *Dreaddit: A Reddit Dataset for Stress Analysis in Social Media*, LOUHI workshop at EMNLP. [ACL Anthology D19-6213](https://aclanthology.org/D19-6213/) |
| Download | `http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip`, the URL given in the paper (1,348,791 bytes, last modified 2019-12-12) |
| SHA-256 | `6c7d8859764231cca47410d6995d8941ac45b08a446a561923fd73b34f3f61cb` |
| Contents | `dreaddit-train.csv` and `dreaddit-test.csv`, 116 columns each, including LIWC and other lexical features. The archive has no README or license file. |
| License | None stated, either in the paper, on the download page or in the archive. The text is from public Reddit posts, so Reddit's User Agreement applies too. I use it for research only and don't redistribute it. |
| What's labeled | Five-sentence excerpts of Reddit posts, labeled by Amazon Mechanical Turk workers (majority vote). |
| Labels | `0` is `not_stress`, `1` is `stress` |
| Subgroups | 10 subreddits, which the paper groups into 5 domains (abuse, anxiety, financial, PTSD, social). The domain is stored as `meta.domain`. |
| Grouping | Several excerpts can come from one post (3,553 excerpts from 2,929 posts), so sampling always keeps a whole post in one role. |
| Label quality | The paper reports Fleiss' κ = 0.47 over all labeled data. The `confidence` column is the share of annotators who agreed with the majority label (kept as `meta.annotator_agreement`). |

### Splits

The paper splits by post, putting 10% of posts in the test set (Section 5). Table 1 of the paper
gives the totals per subreddit, and the files match them:

| Subreddit | Domain | Train | Test | Total (= Table 1) |
|---|---|---|---|---|
| domesticviolence | abuse | 316 | 72 | 388 |
| survivorsofabuse | abuse | 245 | 70 | 315 |
| anxiety | anxiety | 503 | 147 | 650 |
| stress | anxiety | 64 | 14 | 78 |
| almosthomeless | financial | 80 | 19 | 99 |
| assistance | financial | 289 | 66 | 355 |
| food_pantry | financial | 37 | 6 | 43 |
| homeless | financial | 168 | 52 | 220 |
| ptsd | ptsd | 584 | 127 | 711 |
| relationships | social | 552 | 142 | 694 |
| **Total** | | **2,838** | **715** | **3,553** |

Train has 1,488 stress and 1,350 not-stress excerpts (52.4% stress); test has 369 and 346 (51.6%).
That's 1,857 stress labels overall, which matches the paper.

### Things worth knowing

- The paper says train is 51.6% stressful and test 52.4%. The files say the reverse. Since the
  totals match exactly, the paper most likely swapped the two numbers.
- 98 rows in the test file have `confidence = 0.0` (65 not stress, 33 stress), which shouldn't be
  possible for an agreement share. The paper doesn't explain it. I keep those rows and never
  filter on `confidence`.
- `confidence` is formatted differently in the two files (`1` in one, `1.0` in the other).
- 3 excerpt texts appear in both train and test, under different IDs and posts.
- Some subreddits are tiny (6 test items for food_pantry, 14 for stress, 19 for almosthomeless),
  so per-subreddit numbers are very noisy. The subgroup analysis therefore uses the 5 domains,
  with subreddits as a secondary view.

---

## TweetEval stance

| | |
|---|---|
| Paper | Barbieri et al. (2020), *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*, Findings of EMNLP. The stance data comes from Mohammad et al. (2016), SemEval-2016 Task 6; cite both. |
| Hugging Face | [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval), configs `stance_{abortion,atheism,climate,feminist,hillary}` |
| Source used | The official TweetEval GitHub repo, which holds the same files the Hugging Face configs are built from: `https://raw.githubusercontent.com/cardiffnlp/tweeteval/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66/datasets/stance/` (commit `4fbd22c`, 2022-07-08). These are plain-text files, pinned by commit and by per-file SHA-256, so loading them doesn't need the `datasets` library. The split sizes match the Hugging Face card exactly. |
| License | The Hugging Face card lists the license as "unknown" overall and "Undefined" for stance. The TweetEval README says TweetEval itself has no restrictions, but individual tasks and Twitter's terms may add some, and users must follow Twitter's Terms of Service and API terms. I don't redistribute any tweets. |
| Labels | `0` is `none`, `1` is `against`, `2` is `favor` (from `mapping.txt`) |
| Subgroups | the 5 stance targets |
| IDs | The files have no tweet IDs, so each item's ID is `{target}-{split}-{line index:04d}` in the pinned files. |

### Splits (matching the Hugging Face card)

| Target | Train | Val | Test | Test none / against / favor |
|---|---|---|---|---|
| abortion | 587 | 66 | 280 | 45 / 189 / 46 |
| atheism | 461 | 52 | 220 | 28 / 160 / 32 |
| climate | 355 | 40 | 169 | 35 / 11 / 123 |
| feminist | 597 | 67 | 285 | 44 / 183 / 58 |
| hillary | 620 | 69 | 295 | 78 / 172 / 45 |
| **Total** | **2,620** | **294** | **1,249** | 230 / 715 / 304 |

### Things worth knowing

- The labels are very unbalanced within targets. Climate, for example, has only 11 `against`
  test tweets, so per-class numbers there have wide intervals.
- Almost every tweet ends with `#SemST`, a leftover from how SemEval collected the data (4,063
  occurrences across 4,163 tweets). The stance codebook tells the model to ignore it; the text is
  otherwise passed through unchanged.

---

## How the benchmark samples are drawn

`raterbench data sample` builds the fixed samples (seed 13; settings in `configs/datasets/`). It
stratifies by subgroup and label, allocating each stratum its proportional share with
largest-remainder rounding.

| Dataset | Evaluation | Calibration | Train (few-shot and baseline pool) | Validation |
|---|---|---|---|---|
| Dreaddit | 715 (the whole test split) | 305, held out from train (aiming for 300; whole posts are kept together, so it overshoots slightly) | 2,533 | none published |
| TweetEval stance | 1,000 of the 1,249 test tweets | the other 249 | 2,620 | 294 |

- Evaluation, calibration and training items never overlap, and few-shot examples only come from
  the training items.
- Dreaddit's test split has 715 items, so its evaluation set can't reach the 1,000 used for
  TweetEval. I use all 715.
- For TweetEval, the calibration set is the rest of the test split, so it comes from exactly the
  same distribution as the evaluation set, which is what conformal prediction wants.
- For Dreaddit, the test split was too small to divide, so calibration items come from the
  training split instead. The original train/test split picked posts at random, so the two should
  look alike, but the conformal guarantee is a little weaker here.
- Neither dataset has an explicit license. I use both for research only, publish only IDs and
  text hashes, and follow the original sources' terms.
