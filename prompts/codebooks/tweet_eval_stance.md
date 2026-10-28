# Codebook: Stance toward a target in tweets

(Written for RaterBench, following the task framing of SemEval-2016 Task 6, Mohammad et al., 2016. It is
not the original annotator questionnaire.)

## Unit of annotation
One tweet and one **target** (named in the task description). Decide the author's stance toward that
target, based on the tweet alone.

## Labels

### `favor`
From reading the tweet, the author most likely **supports** the target. This holds:
- when support is stated directly, or
- when it can be **inferred** even though the tweet does not mention the target. Examples: supporting a
  position or group aligned with the target, or opposing something the target opposes.

### `against`
From reading the tweet, the author most likely **opposes** the target. This holds when opposition is
stated directly or can be inferred: supporting something opposed to the target, or attacking the target
or its supporters.

### `none`
No stance toward the target can reasonably be inferred. Examples: neutral news or questions, tweets
unrelated to the target, or tweets whose stance toward the target is truly undeterminable.

## Inference rules
1. **The target need not be mentioned.** Most tweets express stance indirectly. Infer the author's
   likely position.
2. **Opposing the opposition implies support.** Criticizing people or groups who oppose the target
   implies `favor`. Praising them implies `against`.
3. **Target-specific cues:**
   - *Legalization of Abortion:* pro-life or anti-abortion messages imply `against`. Pro-choice or
     reproductive-rights messages imply `favor`.
   - *Atheism:* expressions of religious faith, prayer, or praise of God imply `against`. Criticism of
     religion or religious belief implies `favor`.
   - *Climate Change is a Real Concern:* concern about environmental damage or support for climate
     action implies `favor`. Denial or mockery of climate concern implies `against`.
   - *Feminist Movement:* support for gender equality or feminist causes implies `favor`. Attacks on
     feminism or feminists, or sexist content, imply `against`.
   - *Hillary Clinton:* support for her or her campaign implies `favor`. Criticism of her, or support
     for her political opponents framed against her, implies `against`.
4. **Hashtags** often carry the stance (for example #prolife, #FeelTheBern). Read them as part of the
   message. Ignore `#SemST`, which is a dataset artifact with no meaning.
5. **Sarcasm and irony.** Label the intended meaning, not the literal one.
6. **Mixed or unclear signals.** Choose the stance with the strongest evidence. Use `none` only when no
   stance can reasonably be inferred, and give low confidence when evidence is weak.
