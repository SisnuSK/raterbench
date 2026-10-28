# Codebook: Stress in Reddit post excerpts

(Written for RaterBench. These are not the original Dreaddit annotator instructions, which are in the
appendix of Turcan & McKeown, 2019.)

## Unit of annotation
One excerpt (about five sentences) from a longer Reddit post. Judge only what is in the excerpt. Do not
guess what the rest of the post says.

## Definition
**Stress** is a state of mental or emotional strain or tension that comes from adverse or very
demanding circumstances. Label the **writer's own state at the time of writing**, as the excerpt
expresses it.

## Labels

### `stress`
Use when the excerpt expresses that the writer is currently under strain. Indicators include:
- Explicit statements of being stressed, anxious, overwhelmed, scared, panicking, desperate, or unable
  to cope ("I can't take this anymore", "I'm freaking out").
- A description of ongoing difficult circumstances together with a negative emotional reaction to them
  (worry, fear, frustration, hopelessness).
- Current symptoms tied to distress: trouble sleeping, nightmares, flashbacks, panic attacks, crying,
  feeling on edge.
- Pleas for help or advice that come from the writer's distress, even if no emotion word is used, when
  the situation is plainly pressing for them (for example facing eviction tomorrow with nowhere to go).

### `not_stress`
Use when the excerpt does not express that the writer is currently under strain. This includes:
- Neutral, factual, or informational content (resource listings, logistics, background facts).
- Advice, support, or encouragement written to someone else, with no sign of the writer's own strain.
- Calm, positive, grateful, or hopeful tone, including reports of improvement or recovery.
- Past difficulties described as resolved or recounted from a settled present ("Back then I was a
  wreck, but things are good now").
- Other people's stress described without the writer expressing their own.

## Edge cases
1. **Past trauma with present effects.** If past events are described *and* the excerpt shows present
   distress (current symptoms, current fear), label `stress`. If past events are narrated calmly with
   no present strain, label `not_stress`.
2. **Hard circumstances, flat tone.** A difficult situation stated matter-of-factly is not enough on its
   own. Ask whether the excerpt conveys strain for the writer. Label `stress` only if strain is
   expressed or unmistakably implied by an urgent, personal predicament.
3. **Requests for help.** Routine questions ("Does anyone know a food bank open on Sundays?") are
   `not_stress`. Requests framed by worry or desperation are `stress`.
4. **Anger and frustration.** Count these as stress when they reflect strain from the writer's
   situation (for example a conflict they feel trapped in). Do not count them when they are detached
   commentary.
5. **Mixed excerpts.** If any part clearly expresses the writer's current strain, label `stress`.
6. **Humor or sarcasm.** Label the underlying state. Joking about a situation the writer is clearly
   struggling with is still `stress`.
7. **Uncertainty.** If the excerpt gives no clear evidence either way, label `not_stress` and give low
   confidence.
