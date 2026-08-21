# Pilot v2 findings: does selective refusal survive translation?

**In one sentence:** on the same questions, translated rather than resampled, one model refuses
answerable questions roughly twice as often in Polish, Russian, Mandarin and Cantonese as it does
in English, and that result holds up against every artefact explanation we could test.

**Who this is for.** The research team, including people who have not read the RefusalBench paper.
Where the paper is relevant it is summarised here, with a section number so it can be checked.
Every statistical term used is defined in section 0.

**Scope.** One model, Llama 3.1 70B. 900 evaluated items. This is a pilot: it is enough to decide
whether the direction is worth funding, and not enough to publish. Section 5 lists what it does
*not* show, and it is the section to read before quoting any number elsewhere.

**How to read this.** Sections 1 and 2 are the argument. Section 3 is context about the model,
useful but not the point. Sections 4 and 5 are the honesty sections. Sections 6 and 7 are for
deciding what to do next. Every number comes from one of the three notebooks in this directory,
and each section says which one.

---

## 0. Terms used in this report

Skip this if the terms are familiar. Nothing here is specific to this pilot.

| Term | What it means |
|---|---|
| **Paired comparison** | The same items are measured under both conditions, so each item acts as its own control. Here the same 180 questions appear in all five languages, so "English versus Polish" compares a question to itself. |
| **Unpaired comparison** | The two conditions are treated as two separate samples of items. It throws away the pairing, and it needs far more items to detect the same effect. |
| **Discordant item** | In a paired comparison, an item where the two conditions disagree. Items that both conditions get right, or both get wrong, carry no information about which condition is harder, so they drop out of the test. The number of discordant items is the real sample size. |
| **McNemar's test** | The standard test for paired yes/no outcomes. It looks only at the discordant items and asks whether the split between them is more lopsided than chance would give. |
| **p value** | The probability of seeing a result at least this lopsided if the two conditions were really identical. Small means "hard to explain as luck". **It says nothing about how big the effect is.** A large effect on few items and a small effect on many items can give the same p. Effect size has to be read off the counts, separately. |
| **Holm correction** | Testing four languages means four chances to get a small p by luck. The Holm procedure adjusts each p value upward to account for that. Where both are shown, the adjusted one is the one to read. |
| **Cohen's κ / Fleiss' κ** | Agreement between raters on a label, corrected for how much they would agree by chance alone given how often each label is used. Cohen's is for two raters, Fleiss' for more. 1.0 is perfect, and above roughly 0.8 is usually called very strong. |
| **Spearman's ρ** | Agreement on *ranking* rather than on exact values. If one rater scores an item higher, do the others also score it higher? Used here for the 1-to-5 quality score. 1.0 is perfect. |
| **Permutation test** | Instead of assuming a formula, shuffle the group labels many thousands of times and see how often chance alone produces a difference as large as the observed one. Used in the leakage analysis. |
| **Placebo arm** | A condition deliberately built to have no effect, included so that whatever changes in it tells you how much change to expect from nothing at all. Section 4.2 uses one. |
| **Noise floor** | The amount a measurement moves when nothing has actually changed, for example when the same model is re-run or the same responses are re-judged. A result only means something if it is larger than the floor. |

---

## 1. What was tested

### The capability

A model is given a question plus a supporting context. It should answer when the context supports
an answer, and refuse when it does not. RefusalBench calls this **selective refusal**. One of the
paper's findings about models (§4.2, *Refusal Comprises Two Distinct Skills*) is that refusing well
is really two separate abilities:

- **detection**, noticing that something is wrong,
- **categorisation**, naming *what* is wrong.

The paper's own headline contribution is different, and it is the generative methodology: build
test items on demand so that a benchmark cannot be exhausted or contaminated (§3.1).

For the categorisation part the paper defines six categories of informational uncertainty (§3.2),
each with one expected refusal code, such as `REFUSE_MISSING` when the context is silent on the
matter or `REFUSE_FALSE_PREMISE` when the question assumes something the context contradicts.

Two implementation details are worth flagging before any number below is read.

**The code names in this repository are longer than the paper's.** `REFUSE_MISSING` in the paper is
`REFUSE_INFO_MISSING_IN_CONTEXT` in
[`aggregate.py`](../refusalbench/multilingual/aggregate.py), and so on for the rest. They are the
same six categories under different spellings.

**Scoring here is seven-way, not six-way.** `VALID_REFUSAL_CODES` adds a seventh label,
`REFUSE_OTHER`, which the paper's taxonomy does not define. It is never the ground truth for any
item, and the model never emits it, so it costs nothing here. It does mean that "exact seven-way
code match" describes our scoring code rather than the paper.

### The data

180 base items, each translated into five languages (English, Polish, Russian, Mandarin,
Cantonese), giving **900 evaluated items**. Every item belongs to one of six perturbation classes
and one of three intensity levels, with 10 items per language, class and intensity cell.

**Every item is perturbed, including the answerable ones.** This is easy to get wrong and it
changes how the results read. The paper (§3.3, *Intensity Progression*) defines the ladder as:

- **LOW.** A subtle defect that a competent model should resolve and answer correctly. The point is
  to test for *over-sensitive* refusal. All 300 LOW items are **answerable**.
- **MEDIUM.** A clear informational deficit that requires refusal.
- **HIGH.** A severe defect, often a logical paradox. All 600 MEDIUM and HIGH items are
  **unanswerable**.

So an answerable item here is a *baited* question, not a clean one. A LOW FalsePremise item asks
for a fact the context does supply, inside a query that presupposes something false. For example,
*"Who is the CEO of InnoTech, the Google subsidiary?"* where the context says only that InnoTech's
CEO is John Doe and never mentions Google. The benchmark's ground truth is to answer the answerable
part. Keep this in mind at section 3.3.

Each perturbation class maps to exactly one correct refusal code, one to one.

### The model under test

Llama 3.1 70B, recorded as `llama3.1:70b` in the `response_metadata` field of every inference
record. It is the same model across all four inference run directories and all five languages,
which [`rendered/pilot_v2_leakage_subsets.ipynb`](rendered/pilot_v2_leakage_subsets.ipynb) checks rather than
assumes. Nothing in this pilot evaluates any other model.

This is not an off-benchmark choice. `Llama-3.1-70B-Instruct` is one of the 30+ models the paper
itself evaluates (§4.1, *Models Evaluated*, and the model table in Appendix J). We cannot quote a
paper score for it as an anchor, because the only per-model statements in the text are relative
ones: on RefusalBench-NQ, Llama's answer accuracy falls slightly from 8B to 70B while its refusal
accuracy improves 3.1 times (§4.2 RQ3, *Effect of Scale*). The absolute numbers live in the
paper's figures.

### The scoring

Three LLM judges (`mistral-large-3`, `nemotron-3-super`, `sonnet4_5`) read each response and
record whether the model answered or refused, which refusal code it used, and an answer quality
score from 1 to 5. Reported metrics use the majority verdict across the three.

The paper used **one** judge for RefusalBench-NQ, Claude-4-Sonnet (§4.1, *Evaluation Protocol*).
Using three lets us measure how much the judge itself contributes, which turns out to matter
(section 3.5) and which is a finding about the benchmark rather than about Llama.

### The design feature that makes this work

The same 180 items appear in every language. They were **translated, not resampled**. That makes
cross-language comparison *paired*, and section 2.1 shows this is the difference between finding
something and finding nothing.

### Why this pilot exists

The paper's Limitations section (§6, *Linguistic and Modal Scope*) states that the implementation
and the released benchmarks are exclusively English, that informational defects can manifest in
highly language-specific ways, and that extending the framework to other languages is future work.
This pilot is a first probe at a gap the authors flagged themselves.

---

## 2. The main result: translation makes the model refuse answerable questions

*Source: [`rendered/pilot_v2_eval_exploration.ipynb`](rendered/pilot_v2_eval_exploration.ipynb) section 5.*

### 2.1 Read as five separate samples, nothing is there

This is worth showing first, because it is the reading most people reach for and it finds nothing.

Treating the five languages as five independent sets of items, English leads on the pooled
calibrated score (0.483 against 0.383 to 0.433) and every other language sits below it. But every
non-English confidence interval overlaps the English one. On answer accuracy English spans 0.541
to 0.773 while Polish, the weakest, spans 0.377 to 0.623. The largest gap between languages is
0.100, and section 3.5 shows that simply choosing a different judge moves the same number by
0.058.

Those intervals are wide because the 60 items differ enormously among themselves. Some questions
are just harder than others. The unpaired reading has to account for that variation, and at 60
items per language it swamps everything.

**The pairing is what removes it.** Because it is the same question in both languages, item
difficulty cancels out.

### 2.2 Read as paired, the effect is clear

On answerable questions the model wrongly refuses **15%** of the time in English. Translate the
identical question and that rises to between **28% and 35%**.

Because the items are paired, this can be counted directly rather than inferred. Out of the 60
answerable items in each language:

| language | refused only after translation | refused only in English | Holm-adjusted p |
|---|---|---|---|
| Polish | 14 | 2 | 0.013 |
| Cantonese | 13 | 1 | 0.007 |
| Mandarin | 10 | 2 | 0.043 |
| Russian | 9 | 1 | 0.043 |

![Paired disagreement against English](figures/04-paired-language-disagreement.png)

*Bars to the left are items only English gets wrong. Bars to the right are items only the other
language gets wrong. A balanced bar would mean translation changed nothing.*

**How to read one row.** Take Polish. Of the 60 answerable items, most are handled the same way in
both languages and drop out of the test. 16 are handled differently. Of those 16, English answered
and Polish refused on 14, and Polish answered and English refused on 2. If translation made no
difference, that split should be near 8 to 8. The Holm-adjusted p of 0.013 says a 14 to 2 split is
hard to get by chance, even after allowing for the fact that we tested four languages.

**Be honest about the sample.** The Polish result rests on those **16 discordant items**, not on 60
and certainly not on 180. Every language points the same way and all four clear correction, but
these are small numbers and the *size* of the effect is not established (section 5.1).

### 2.3 Four independent lines of support

The result was stress-tested in several ways. Each row below is a separate re-analysis, not a
restatement of the same one.

| Check | What was done | Result |
|---|---|---|
| **Judge choice** | Re-ran the paired test under each judge alone, with no majority pooling | Reproduces in all **12** judge and language combinations, worst p 0.039 |
| **Judge instrument** | Re-judged all 900 responses with each judge instructed in the language it was reading, then recomputed the paired test from scratch | Reproduces: 14 to 2, 10 to 1, 10 to 2, 12 to 1. All four still clear Holm correction (section 4.2) |
| **Input contamination** | Restricted to three progressively cleaner subsets of items with less untranslated English left in the prompt | Every gap stays positive, ten of the twelve grow, and zero counter-examples appear on the one-sided disagreement counts (section 4.1) |
| **Model sampling noise** | The subset directories re-ran the same model on the same prompts | 2 flips in 155 answerable re-runs, and resampling noise is symmetric, so it cannot create a one-sided split (section 4.3) |

One further point about what the result does *not* depend on. Whether a response is a refusal is a
pure classification call, and the judges agree on those at κ 0.99. It does not touch the 1-to-5
quality score, which is the one thing the judges genuinely disagree about (section 3.5). The
headline is therefore insulated from the noisiest part of the measurement.

### 2.4 Answer accuracy moves the same way, but the evidence is thinner

Answer accuracy falls from 0.667 in English to between 0.500 and 0.550 elsewhere, and the paired
disagreement points the same way in every language (13 against 3 in Polish, down to 9 against 2 in
Cantonese). But three of the four languages are individually significant and Cantonese is not
(p 0.065), and after Holm correction only Mandarin survives (0.047).

The reason is section 3.5: `answer_accuracy` is gated on the quality score, which is where the
judges disagree. **We report the direction, not the size.**

### 2.5 Translation does not change *how* the model names its refusals

This is the clean negative that sharpens the headline, and it is easy to overlook.

Paired refusal accuracy is flat across all four languages. Counting discordant items as
*English-only correct* against *other-language-only correct*, the split is near even everywhere:
10 against 6 in Polish, 7 against 11 in Russian, 6 against 8 in Mandarin, 8 against 10 in
Cantonese. Three of the four lean towards the translated language rather than towards English.
Every p value is above 0.45.

So translation changes **willingness to answer**, not refusal reasoning. When the model does
decline, it picks its reason the same way in every language. That fits section 3.1: the choice of
code is dominated by a collapse onto one label, which is a property of the model rather than of
the language it is reading.

### 2.6 What this result does not say

- It does not rank the four languages. See section 5.1.
- It does not measure how large the effect is. See section 5.1.
- It does not separate "in Polish" from "in translated text". See section 5.2.
- It says nothing about any other model. See section 5.6.

---

## 3. Supporting findings about the model

These are about Llama 3.1 70B rather than about language. They matter mainly because they set
limits on which metrics can be trusted in section 2.

### 3.1 The model collapses its refusal vocabulary onto one code

*Source: [`rendered/pilot_v2_eval_exploration.ipynb`](rendered/pilot_v2_eval_exploration.ipynb) section 4.*

The model is good at noticing that something is wrong and poor at saying what. On the 600
unanswerable items it declines on **500** but names the correct reason on only **184**.

The cause is a near-total collapse onto one label. `REFUSE_INFO_MISSING_IN_CONTEXT` accounts for
**61%** of all predictions on unanswerable items, against a true prevalence of **17%**, and for
**73%** of the refusals the model actually issued.

![Refusal code collapse](figures/03-refusal-code-collapse.png)

*Each row is the true reason, each column is what the model said, rows sum to 1. A model that told
the reasons apart would show a strong diagonal. This shows a vertical stripe.*

- `REFUSE_GRANULARITY_MISMATCH` is **never emitted once in 900 items**, and it is the correct
  answer for 100 of them. Those items score zero by construction.
- False-premise items are called "information missing" 86% of the time. Granularity items, 91%.
- A model that refused everything with the single label "information missing" would score 0.167.
  The observed score is 0.307, better than a one-word vocabulary, but not by the margin a seven-way
  task should give.

**This is a known pattern, and it is worse here than in the paper.** RefusalBench §4.2
(*Systematic Misclassification of Refusal Types*) reports that models default to
`REFUSE_INFO_MISSING` as a catch-all taking **25%** of all predictions on RefusalBench-NQ, with
granularity and ambiguity items frequently misclassified as missing information. The paper also
calls `REFUSE_GRANULARITY` "nearly unsolvable" (§4.2, and Appendix F.4, where the best model
reaches 31.1%).

**Match the denominators before quoting the comparison.** The paper's 25% is a share of *all*
predictions, averaged across all models it evaluates. The comparable figure for our model is
**49%** of all 900 predictions. The 61% above is the share restricted to unanswerable items, which
is the more diagnostic slice but not the paper's. On either denominator this model defaults to the
catch-all far harder than the paper's average, and it never emits the granularity code at all. The
shape is general. The severity is not.

**Why this matters for everything downstream.** `refusal_accuracy` requires an exact code match,
so under this collapse it is not really measuring refusal skill. It mostly measures how close a
slice's correct codes sit to the one code the model actually uses. Every downstream
`refusal_accuracy` number inherits that, which is why section 5.3 rejects one apparent finding
entirely.

### 3.2 Detection and categorisation really are separate skills

*Source: [`rendered/pilot_v2_eval_exploration.ipynb`](rendered/pilot_v2_eval_exploration.ipynb) section 7.*

Comparing MEDIUM against HIGH, both entirely unanswerable and both carrying identical
distributions of correct codes, the model detects HIGH more readily: 261 declines against 239, and
105 right codes against 79, out of 300 each. That is what the design intends, since HIGH items
carry more severe defects, and the paper reports the same monotonic rise in refusal rates from LOW
to HIGH on RefusalBench-NQ (§4.2, *Perturbation Type and Intensity Drive Performance Patterns*).

The interesting part is what does *not* move. `REFUSE_INFO_MISSING_IN_CONTEXT` is emitted
**exactly 182 times at each intensity level**. All 22 of the extra refusals at HIGH went to some
other code. Making the defect more obvious changes *whether* the model refuses and does not shift
the code it reaches for.

**This corroborates the paper's claim about refusal from a different direction.** The paper
establishes the detection and categorisation split by comparing models against each other (§4.2,
*Refusal Comprises Two Distinct Skills*). We see the same split *inside a single model*, along an
axis the paper does not use for this purpose.

### 3.3 False-premise framing is associated with heavy over-refusal

*Source: [`rendered/pilot_v2_eval_exploration.ipynb`](rendered/pilot_v2_eval_exploration.ipynb) section 6.*

![Perturbation class](figures/05-perturbation-class.png)

Answerable items in the **FalsePremise** class score **0.220** answer accuracy against **0.740 to
0.760** for Contradiction and Ambiguity, driven by a false refusal rate of **0.660**.

**Two caveats have to travel with that number.**

*On what the items are.* These are LOW FalsePremise items, and as section 1 described, the query
really does contain a false presupposition. The ground truth is to answer the answerable part
anyway. So the correct reading is not "the model refuses clean questions". It is that *when a query
contains a false presupposition alongside an answerable request, the model refuses two thirds of
the time instead of answering the answerable part*. Whether that should count as a failure is a
fair question about the benchmark's ground truth, not only about the model. It remains a real
over-refusal signal, because the other five classes carry LOW-level defects too and refuse far
less often (0.10 to 0.34).

*On the sample.* The 0.220 and 0.660 are computed over 50 rows, but those 50 rows are **10 base
items seen in five languages**, not 50 independent items. Section 5.4 warns that 10 items per cell
separates nothing. Treat the size as unmeasured and the direction as worth following up.

For context, the paper reports GPT-4o incorrectly refusing over 60% of answerable questions on
RefusalBench-NQ (§4.2, *Refusal Comprises Two Distinct Skills*), so heavy over-refusal on
answerable items is not unique to this model.

### 3.4 Where all 900 items land

*Source: [`rendered/pilot_v2_eval_exploration.ipynb`](rendered/pilot_v2_eval_exploration.ipynb) section 3.*

![All 900 items](figures/02-all-900-items.png)

- **Answerable (300).** 165 answered well, 50 answered but below the quality bar, 85 refused
  outright.
- **Unanswerable (600).** 184 refused with the right code, 316 refused with the wrong code, 100
  answered anyway.

Wrong-code refusals are the single largest bucket in the whole pilot.

The 85 false refusals are 28% of the answerable items, but that figure **pools all five
languages**. It is not a single number about the model: English alone is 15% and the other four
run 28% to 35%. That spread is the subject of section 2.

### 3.5 The judges are reliable for classification and unreliable for quality

*Source: [`rendered/pilot_v2_eval_exploration.ipynb`](rendered/pilot_v2_eval_exploration.ipynb) section 2.*

![Judge agreement](figures/01-judge-agreement.png)

- **Refusal code classification is effectively solved.** Cohen's κ 0.984 to 0.993, Fleiss' κ
  0.990, raw agreement above 99%.
- **Answer quality is not.** Spearman's ρ 0.67 to 0.70, with judges differing by about a third of
  a point on the 1-to-5 scale.

That split has a practical consequence. `answer_accuracy` counts an item as correct only if the
quality score reaches 4, so it carries the disagreement. Simply choosing which of the three judges
to trust moves the calibrated score by up to **0.058**, which is comparable to the largest gap
between languages. Any language claim resting on `answer_accuracy` has to clear that bar, and
section 2.2 is trustworthy precisely because it does not rest on it.

**On whether the ensemble is worth its cost:** a single judge would give practically the same
refusal codes. The three-judge ensemble is only doing work on the quality gate.

**This is also a finding about RefusalBench, not only about Llama.** The paper scored
RefusalBench-NQ with a single judge (§4.1). It measured agreement between models on the *verifier*
side of its generation pipeline, where Cohen's κ ran as low as 0.061 (§4.2 RQ1, and Appendix E.1),
and it validated generated items against a human expert at a 93.1% pass rate (§4.1, *Human
Validation*). Neither of those is a measurement of the evaluation judge. Our result, that
evaluation judges agree almost perfectly on classification and only moderately on quality,
suggests the quality gate is the fragile part of the protocol.

---

## 4. Three artefact explanations were tested

Each of these would produce the section 2 pattern without the model behaving any differently in
other languages. None of them explains the result. Each subsection also says how strong that
conclusion is, because they are not equally strong.

### 4.1 "The translations still contain English, so you are measuring the mess"

*Source: [`rendered/pilot_v2_leakage_subsets.ipynb`](rendered/pilot_v2_leakage_subsets.ipynb).*

If a supposedly Polish prompt still carries chunks of untranslated English, the model is being
scored on a code-switched input rather than on Polish.

Three progressively stricter filtered subsets already existed, keeping 18, 30 and 55 of the 180
items. Restricting to them does not shrink the effect. The English-versus-other gap stays positive
in all twelve language and subset cells, and it grows in ten of them. The two exceptions are
Russian in the two larger subsets, where the gap narrows slightly (0.133 on the full run against
0.125 and 0.118), which is one item either way at these sample sizes. Polish and Cantonese roughly
double.

The disagreements also become entirely one-sided. Across all twelve cells there is **not one item**
where English refuses and the translated version does not, against six such items on the full run.

![Leakage subsets](figures/06-leakage-subsets.png)

Two qualifications matter:

- **The subsets on their own cannot settle it.** They leave 6, 8 and 17 answerable items. The
  smallest cannot reach significance at any effect size, because with 6 items and 3 disagreements
  the smallest attainable exact p is 0.25, and one perturbation class disappears from it entirely,
  so any raw comparison of full against filtered would be measuring composition rather than
  leakage.
- **Measured properly, the direction holds but the size does not.** Scoring leakage on all 180
  items and splitting them five different ways, cleaner items show the larger gap under four of
  the five, monotonically across tertiles on both leakage measures, with both rank correlations
  negative. But the permutation p ranges from 0.034 to 0.90 depending on how "clean" is defined.
  Treat this as **leakage failing to explain the effect, and probably diluting it**, rather than as
  a measured dilution of a given size.

### 4.2 "The judge was instructed in English, so it is harsher on other languages"

*Source: [`rendered/pilot_v2_judge_prompt_language.ipynb`](rendered/pilot_v2_judge_prompt_language.ipynb).*

Every verdict in section 2 came from a judge reading English instructions, whatever the language
of the response. A judge that was harsher on unfamiliar text would produce exactly the observed
pattern.

The whole evaluation was rerun with each judge instructed in the language it was reading.

**This design has a built-in placebo arm.** For English, the "localised" prompt *is* the canonical
prompt, because translating English into English changes nothing. So the English items were
re-judged with an identical instrument, and anything that moves in the English slice cannot be an
effect of localisation. It can only be the judges being non-deterministic. That gives a noise floor
to compare the other languages against, measured before any result is looked at.

![Judge prompt language](figures/07-judge-prompt-language.png)

- **The floor.** On a same-prompt re-run, English labels changed on 0.4% of judge rows and quality
  scores on 1.7%.
- **Classification sits at the floor.** Non-English label flips run 0.0% to 0.9%. Polish, on a
  fully localised prompt, changed no label at all, which is *below* the placebo.
- Consensus classification labels are identical on **896 of 900 items**.
- No consensus metric on any language moves by more than **0.017**, against the 0.058 that
  swapping judges moves.
- The false refusal rates change by at most **one item in 60**, and all four languages still clear
  Holm correction.
- **No language moves towards the English baseline**, which is exactly what a language-biased judge
  would have produced.
- The refusal-code collapse from 3.1 is unchanged, so it is a property of the model and not of
  English-instructed judging.

Localising the judge prompt also buys no improvement. Inter-judge classification κ moves by at
most 0.005, and quality agreement moves up for two judge pairs and down for the third. That is
noise redistributed, not reduced.

*One caveat the notebook states and this report should carry:* the floor is itself estimated from
a single re-run, so it is only known to roughly a percentage point. It supports "non-English sits
at or near the floor". It does not support a fine ranking of the four languages against each
other.

### 4.3 "The model is just noisy, and you sampled English once"

*Source: [`rendered/pilot_v2_leakage_subsets.ipynb`](rendered/pilot_v2_leakage_subsets.ipynb) section 1.*

Section 4.2 bounds the noise in the *judge*. The equivalent question on the *model* side is
whether re-running Llama on the same English prompts would itself produce a different set of
refusals, in which case some of the 14-against-2 split could be resampling noise.

The subset directories answer part of this by accident, because they re-ran the same model on the
same prompts. Restricted to the answerable items, there are **155 re-run pairs across the three
subsets and all five languages, and 2 changed between answering and refusing** (1.3%).

Both of those were English, and both moved towards refusing. So on English answerable items
specifically the flip rate is 2 out of 31, not 2 out of 155, and it is fair to quote the less
flattering number. Two things still follow, and neither depends on the exact rate:

- **The direction of the noise helps rather than hurts.** Both flips made English refuse *more*.
  A re-run would therefore have raised the English baseline and *narrowed* the gap, which makes
  the section 2.2 estimate conservative rather than inflated.
- **Resampling noise is symmetric.** It flips items in both directions, so it pushes a paired
  comparison towards balance, not away from it. It cannot manufacture a one-sided 14-against-2
  split, however large the flip rate turns out to be.

This is a partial answer, not a complete one. 31 English pairs on filtered items is not the same
as a proper English-versus-English replication of all 60 answerable items. That replication is
recommendation 2 in section 7, and it is the weakest of the three checks in this section.

---

## 5. What the pilot does not show

Each of these is easy to report by mistake.

### 5.1 A ranking between the four languages, or the size of the effect

The paired test establishes the direction and that it is real. It does not pin the size. The gaps
between languages in calibrated score run 0.050 to 0.100, and judge choice alone moves the same
number by 0.058. The most that should be said is: **"translation degrades selective refusal,
driven by over-refusal."**

### 5.2 Whether the effect is about the *language* or about *translation*

This is the most important open question and the design cannot answer it. A translated item is not
a natively authored item. "The model over-refuses in Polish" and "the model over-refuses on
translated text" are different claims with different implications, and nothing here separates
them. Back-translated English controls, or natively authored items in each language, would.

Related: the four languages are **not four independent replications**. They are four translations
of the *same* 60 answerable items, produced by one pipeline. That pipeline's provenance is not
recorded anywhere in this repository, and `refusalbench/multilingual` explicitly does not build
translations, so the translation method is currently an undocumented shared input to all four
results.

### 5.3 A difficulty ranking across perturbation classes on the refusal side

The right-hand panel of the perturbation figure looks decisive: MissingInfo scores 0.84 and
Granularity scores 0.00, with non-overlapping intervals. It is an artefact. MissingInfo scores
well because its correct code happens to be the one code the model defaults to, and Granularity
scores zero because its code is never emitted at all.

**That panel shows which codes the model uses, not which perturbations are hard.** The honest
statement is that the model has no granularity refusal behaviour whatsoever.

### 5.4 Anything about intensity, and anything at the finest slice

Answerability is perfectly confounded with intensity. LOW carries every answerable item and MEDIUM
and HIGH carry every unanswerable one, so LOW is scored on a different metric and a different
population. The three numbers are not on a common scale and must not be ranked. The one legitimate
comparison, MEDIUM against HIGH, is section 3.2.

**This confound is inherited from the parent benchmark, not introduced here.** Paper §3.3
specifies exactly this ladder: answer at LOW, refuse at MEDIUM and HIGH. Removing it means
deliberately departing from RefusalBench's design, a choice worth making, but a design
contribution rather than a cleanup task.

`metrics_by_stratum.csv` has 90 rows of 10 items each. At n=10 a 95% interval is about half the
scale wide, and all 90 rows are pure, so none carries a calibrated score at all. Use that file to
check design balance and for nothing else.

### 5.5 That the pooled calibrated score means anything

`calibrated_refusal_score` averages `answer_accuracy` (dominated by judge noise, section 3.5) with
`refusal_accuracy` (dominated by label collapse, section 3.1). The paper defines it the same way,
as the arithmetic mean of the two (Appendix D). A movement in the average cannot be attributed to
either component. Sections 2.2 and 2.5 are a concrete case: the two components move independently
and the average hides it.

### 5.6 Anything about models other than Llama 3.1 70B

Only one model was evaluated. Nothing here is a claim about language models in general. Section 6
separates what is likely general from what is likely specific to this model.

---

## 6. How this sits against the RefusalBench paper

| Paper finding | Where | What this pilot saw | Verdict |
|---|---|---|---|
| Models default to `REFUSE_INFO_MISSING` as a catch-all taking 25% of all predictions on NQ, averaged across models, with granularity and ambiguity frequently misclassified as missing information | §4.2, *Systematic Misclassification* | Same failure at **49%** of all predictions, **61%** of predictions on unanswerable items, with granularity items misfiled as missing information 91% of the time | **Replicates**, more severely |
| `REFUSE_GRANULARITY` is "nearly unsolvable", best model reaches 31.1% | §4.2 and Appendix F.4 | Never emitted once in 900 items, 0.00 accuracy on its 100 items | **Replicates**, more severely |
| Refusal comprises separable detection and categorisation skills | §4.2, *Two Distinct Skills* | Corroborated from a new direction: raising defect severity moves detection while `INFO_MISSING` stays at exactly 182 emissions per level | **Replicates**, independently |
| Refusal rates rise monotonically with intensity on NQ | §4.2, *Perturbation Type and Intensity* | 239 declines at MEDIUM against 261 at HIGH, out of 300 each | **Replicates** on the one intensity comparison this design allows |
| Over-refusal on answerable items is a real failure mode (GPT-4o refuses more than 60%) | §4.2, *Two Distinct Skills* | 0.66 false refusal on LOW FalsePremise items, though on only 10 base items | **Consistent**, weak evidence |
| The implementation and released benchmarks are English-only, and other languages are future work | §6, *Linguistic and Modal Scope* | Translation roughly doubles false refusal on identical items, in four languages | **New ground** |
| RefusalBench-NQ was scored by a single LLM judge | §4.1, *Evaluation Protocol* | Three judges: κ≈0.99 on classification, ρ≈0.67 on quality, 0.058 swing on the pooled score | **New**, a methodological finding about the benchmark |

Two things follow for how to describe this work. The cross-language result is aimed at a gap the
paper's authors flagged themselves, which is the strongest argument for the direction. And the
three-judge measurement is an under-claimed contribution, because it says something about
RefusalBench's evaluation protocol rather than only about Llama.

What we should *not* claim as ours: the catch-all collapse, the granularity failure, the
detection-versus-categorisation split, and the intensity ordering are all the paper's findings.
This pilot reproduces them on a model and in languages the paper did not cover, which is worth
reporting as replication and nothing more.

---

## 7. What to change before scaling up

Ranked. The first two are cheaper than adding a fifth language and would strengthen the headline
more.

**Highest value**

1. **Run a second model.** Everything in section 2 is an existence proof on one model. A second
   model is what turns it from "Llama does this" into "this happens". Nothing else changes the
   status of the finding as much.
2. **Run English against itself.** Re-sample the target model on all 60 English answerable items
   and run the same paired test. Section 4.3 gives a partial floor from re-run subsets, but a
   proper English-versus-English arm would bound generator-side noise directly, the same way
   section 4.2's placebo bounds judge noise. Cheap, and it closes the last *measurement* artefact
   question. What it does not touch is translation quality, which is item 3.
3. **Get the translations reviewed by a speaker of each language, and record how they were made.**
   This is the check none of the three notebooks can perform, and it is now the main outstanding
   risk to the headline result. The translation pipeline's provenance should be written down
   alongside the data (section 5.2).

**On the design**

4. **Keep the paired design and analyse it as paired.** Translating the same items rather than
   sampling fresh ones per language is what makes section 2.2 reachable at 60 items. An unpaired
   reading of the same data finds nothing. Any scaled-up run should preserve the shared item set,
   and the metrics files should carry the item ID so paired tests remain possible downstream.
5. **Add natively authored or back-translated control items.** This is what would separate "in
   Polish" from "on translated text" (section 5.2). It is the difference between a result about
   four languages and a result about translation.
6. **Break the answerability and intensity confound.** Put answerable controls at every intensity,
   or stop calling the answerable stratum an intensity level. As built, no intensity conclusion is
   reachable. Note this means departing from the parent benchmark deliberately (section 5.4).
7. **Spend extra items on answer quality rather than on more languages.** False refusal already
   resolves at 60 items per language. Answer accuracy does not, because it is gated on the one
   judgment the judges disagree about.

**On the metrics**

8. **Compute the paper's own refusal metrics, which our pipeline does not.** Appendix D already
   defines Refusal Detection F1, Category Accuracy (the reason given a correct decision to refuse)
   and the Hierarchical Refusal Score that combines them.
   [`aggregate.py`](../refusalbench/multilingual/aggregate.py) emits none of the three, so an
   exact code match currently scores a correctly detected refusal as a total failure whenever the
   code is wrong, which is 316 of 500 detections here. Adding them costs a few lines and untangles
   the two skills that section 3.2 shows move independently.
9. **Report `answer_accuracy` and `refusal_accuracy` separately and retire the pooled calibrated
   score,** or at least never publish it without both components (section 5.5).

**On the pipeline**

10. **Keep `english` as the default judge prompt.** It is cheaper, it needs no translated judge
    segments kept in sync with the rubric, and the localised run shows it costs nothing. Keep
    `--judge-prompt-language target` as a robustness check for when the rubric or the language set
    changes.
11. **Do not use the three filtered subset directories as an evaluation filter.** Score leakage
    over all items and use it as a covariate instead. That keeps the design balanced and keeps the
    sample at 60 rather than 6.

---

## 8. Where to look for more

| Question | Notebook |
|---|---|
| What do the statistical terms mean? | Section 0 above |
| How were the metrics defined, and what does the design allow? | [`rendered/pilot_v2_eval_exploration.ipynb`](rendered/pilot_v2_eval_exploration.ipynb) sections 1 and 2 |
| The full picture of all 900 items, and the label collapse | sections 3 and 4 |
| The language result, paired and unpaired, and why paired is right | section 5 |
| Perturbation classes and intensity | sections 6 and 7 |
| Does English left in the prompts explain the result? | [`rendered/pilot_v2_leakage_subsets.ipynb`](rendered/pilot_v2_leakage_subsets.ipynb) |
| Does the judge's own prompt language explain it? | [`rendered/pilot_v2_judge_prompt_language.ipynb`](rendered/pilot_v2_judge_prompt_language.ipynb) |
| The RefusalBench paper itself | [`refusalbench.md`](refusalbench.md) |
| How to regenerate the data and run the notebooks | [`README.md`](README.md) |

The marimo notebooks (`*.py`) are the source of truth. The `.ipynb` copies and the figures in this
report are both regenerated from them by [`export_figures.py`](export_figures.py), which executes
each notebook, so a figure here can never disagree with the analysis it came from:

```bash
uv run python notebooks/export_figures.py
```
