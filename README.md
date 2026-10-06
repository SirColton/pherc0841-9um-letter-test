# How much of the 2.4 µm text does a 9 µm reader recover? A pre-registered letter-level test on PHerc0841, and a failed same-scroll distillation

**Author:** Colton Edward Boyer
**Public repository:** https://github.com/SirColton/pherc0841-9um-letter-test
**Round:** ScrollPrize progress prizes, October 2026. Written 2026-10-06.

Every number below is in [`results/`](results/), and its source is named next to it. A line marked [read] is my interpretation, not a measurement.

## Summary

- **Pixel level: a replication.** I tested existing 9 µm ink models on one held-out PHerc0841 segment. The checkpoint my pre-registered selection rule picked, TAUIL-Abd-Elilah's ft6k, scores AUCsup 0.810 against the human ink labels. ScrollPrize's 2.4 µm ink map of the same text scores 0.869.
  - TAUIL's own public README already reports about 0.80 on PHerc0841, and says the letters are not legible at 9 µm.
  - My number replicates that result under a pre-registered, hash-frozen protocol. It is not an advance.
- **Letter level: the new part.** Blind readers named 3 of 8 reference letters from the 9 µm map and 6 of 8 from the 2.4 µm map. Both reads used the same tiles, the same matching rule and the same kind of phase-scrambled null tiles.
  - The pre-registered bar was 6, so the verdict is NO-GO.
  - The released `ink_9um` checkpoint scored 0 of 8.
- **A pre-registered negative result.** I distilled the 2.4 µm map into a 9 µm student on the same scroll.
  - AUCsup rose to 0.825, but the gain cannot be separated from zero, and the depth-shuffle controls failed.
  - In the student's letter read, readers again named 3 of 8 letters. They also called letters on 17 of 25 phase-scrambled tiles, with the same median confidence as their calls on real tiles.
- **A second pre-registered student** (section D), penalised for reading depth-shuffled input, reached AUCsup 0.832 on a stricter second-look bar and failed the same two gates: the gain over v1 cannot be separated from zero, and structure from depth-shuffled input persists.
- **No letters are claimed on any prize-eligible scroll.**

## What this report claims, and what it does not

- **No First Letters claim.**
  - PHerc0841, the test scroll, is not on ScrollPrize's `prizeEligibility.json` (re-read 2026-10-05).
  - The 9 µm letter read ended NO-GO, and so did the distilled student.
  - The PHerc1203 work in section F is a scout at 2.4 µm. A 2.4 µm result is not First Letters evidence (villa issue #1739).
- **The claim is a measured gap.** The test text is held out, and ScrollPrize's 2.4 µm map shows it. On that text, a 9 µm map that does well on pixel-level ink detection lets blind readers name half as many letters as the 2.4 µm map does.
- **What follows from it:**
  - A pixel AUC near 0.81 does not by itself mean legible letters at 9 µm.
  - A 9 µm null result on an eligible scroll should be read against this gap.
- **The reference is not ground truth.** It is a blind AI reading of ScrollPrize's 2.4 µm map (section B). "6 of 8" therefore measures agreement with the letters that map shows, not with a papyrologist's transcription.

## Form field 5: what the contribution is

### (1) Which scroll data was used

- **PHerc0841**, three segments, all public:
  - `20260220213127-w00` ("w00"). A 995 × 1245 px window of it, called "w00big" below, was my selection frame.
  - `auto_grown_20260220144552896` ("552896"), used for student training only.
  - `auto_grown_20260220174252405` ("252405"), the held-out test segment.
- **The native 9.366 µm scan**, volume `20250821151531`. Every 9 µm input in this work was rendered from it.
- **The 2.403 µm scan**, volume `20260319124803`. I used it only for registration and through ScrollPrize's published 2.4 µm ink maps, which served as the letter reference, the ceiling and the student's teacher.
- **Labels:** the public ink-labels dataset, collection `841/` (HF v2 ink labels, supervision masks and validation masks).
- **PHerc1203 at 2.403 µm:** a scout only (section F).

### (2) How it raises the chance of reading these scrolls or others

- A 9 µm null on an eligible scroll means little unless we know how much text a 9 µm reader recovers where the text is known. This work measures that at letter level on held-out PHerc0841 text: 3 of 8 letters against 6 of 8, with the same readers, tiles and matching rule.
- The same 9 µm map clears the pixel-level bar (nAUC 0.84 against a pre-registered 0.771) and fails the letter bar. Pixel AUC and high-pass correlation alone would have overstated progress.
- The student result rules out one tempting next step. Distilling the 2.4 µm map into a 9 µm model on the same scroll did not make the letters more legible. Instead, the model drew letter-like texture on scrambled inputs too.

### (3) What it makes possible that was not possible before

- **A frozen, reusable letter-level protocol** for comparing any 9 µm map with the 2.4 µm map of the same text. It has:
  - blind readers who see one tile at a time;
  - a matching rule fixed in advance: the same letter, with its centre within 0.5 mm of the reference box centre;
  - phase-scrambled and depth-shuffled null tiles;
  - a bar set relative to a ceiling read of the 2.4 µm map, fixed before any 9 µm read;
  - a scorer with 102 self-tests.
- **Depth-order controls scored beside a model-free render:** forward order, a 21-layer shuffle and a 41-layer shuffle.
  - On this segment, the model-free render on its own already exceeds its null in the 48 µm band.
  - A 48 µm control result is therefore not a clean test of whether a model reads depth order.
- **A pre-registered negative result** for same-scroll 2.4 → 9 µm distillation, with a letter read that shows how it fails.

### (4) What evidence is provided

- Sections A to E below. Every number is in [`results/`](results/) (counts only) and is produced by the code in [`code/`](code/).
- The pre-registrations are in [`prereg/`](prereg/). [`prereg/HASHES.md`](prereg/HASHES.md) gives the sha256 of each original as it was frozen, before the numbers it governs existed.
  - Three files are byte-identical to the frozen originals, so anyone can check their hashes: `prereg-v2.json` (49633d68…), `prereg-v2-note-2.txt` (36710c09…) and `recipe-frozen.json` (72ad196a…).
  - The other pre-registration files are redacted copies, with letter identities and private details removed. Their hashes differ from the frozen ones, and HASHES.md lists both.
- The letter-read counts come from the score files that `score.py` and `note4-verdict.py` wrote at the time of each read.
- Limit: the freeze times come from my own git history. No third party timestamped them.

## Prior art, and what is new

- **TAUIL-Abd-Elilah, cross-scan-ink-transfer.** The public README (read 2026-10-06) reports a mean AUC over PHerc0841's three labelled segments:
  - 0.804 for ft6k;
  - 0.812 for ShribyrLabs' reader-ft-s42;
  - 0.734 for the released `ink_9um`.
- The README says the gain plateaus near 0.80-0.81, and that neither model makes PHerc0841's letters legible at 9 µm.
- **My pixel-level result replicates that.** ft6k scores 0.810 on 252405 alone. My AUCsup counts as negatives only pixels marked supervised, so the two numbers are close but not computed the same way. TAUIL reached the conclusion that the letters are not legible first.
- **What I add:**
  1. **How illegible, in letters.** A pre-registered blind letter read gives 3 of 8 letters, against 6 of 8 for the 2.4 µm map of the same text. Null tiles measure how often the readers see letters in plain texture.
  2. **Depth-order controls beside a model-free render.** They show where a control result can and cannot be trusted on this segment.
  3. **A pre-registered negative result for same-scroll distillation**, with a letter read that shows what goes wrong.
- I have not seen a letter-count comparison of a 9 µm map against the 2.4 µm map of the same text in the public reports I read (listed under Credit). If one exists, this work replicates it too.

## A. Existing 9 µm readers on held-out 252405

### Selection, on w00big only

- I scored the existing checkpoints on w00big, a 1.0 cm² window of w00, against the HF v2 labels. No 9 µm model had run on 252405 yet ([`results/selection-w00big.csv`](results/selection-w00big.csv)).
- The rule was fixed in advance: highest AUCsup among four pre-registered recipes, and within 0.01 AUCsup, the higher high-pass correlation at 192 µm.
- How the rule played out:
  - R3 led on AUCsup (0.8926).
  - R1 (0.8915) and R2 (0.8903) were within 0.01 of it.
  - The tie-break picked R1: hp r +0.343, against +0.308 for R3 and +0.309 for R2.
- **R1 is TAUIL's ft6k on its own.** It is ScrollPrize's released `ink_9um`, fine-tuned by TAUIL-Abd-Elilah on native 9 µm PHerc0139 against the 2.4 µm prediction.
  - It runs on the reversed depth order and averages five depth centres.
  - The frozen recipe is [`prereg/recipe-frozen.json`](prereg/recipe-frozen.json) (72ad196a…).
- ShribyrLabs' reader-ft-s42 scored higher on w00big (0.9037). It has no stated checkpoint licence, so it was a reference arm only and never entered a decision.

### What "held out" means here

- 252405 was held out from selection, and from the student's training (section C).
- It is not unseen by the community. TAUIL's README reports a mean over PHerc0841's three labelled segments, and those segments cover one patch of the scroll.

### Metrics

- **AUCsup:** AUC with negatives restricted to supervised pixels. It is computed over 4,468,325 pixels at 9.612 µm: 279,497 ink and 940,946 supervised non-ink.
- **nAUC** = (AUCsup − 0.5) / (C − 0.5). C = 0.8689 is the AUCsup of ScrollPrize's 2.4 µm map on the same pixels.
  - The bars were set on w00, where C is 0.954, and carried over to 252405 as nAUC: 0.771 to go on, and 0.661 as the floor of the ambiguous band.
- **hp r:** Pearson correlation of the high-passed map with the high-passed 2.4 µm map. This is ShribyrLabs' letter-scale score.
  - It is computed at σ 48 µm or 192 µm, with the map blurred by 2 px.
  - The null is 48 shifted copies of the key, 1 to 3 mm away.

### Results ([`results/day1-252405-scores.csv`](results/day1-252405-scores.csv))

| map on 252405 | AUCsup | nAUC | hp r 192 µm [null max] | hp r 48 µm [null max] |
|---|---|---|---|---|
| ScrollPrize 2.4 µm map (the key, C) | 0.8689 | 1.000 | — | — |
| TAUIL ft6k (frozen recipe) | 0.8102 | 0.8409 | +0.3499 [+0.0421] | +0.1280 [+0.0115] |
| released `ink_9um` seed 43, 75k steps | 0.7640 | 0.7155 | +0.2713 [+0.0357] | +0.0702 [+0.0115] |
| model-free render, layers 18-22 | 0.5004 | 0.0010 | +0.0299 [+0.0217] | +0.0145 [+0.0086] |

- **Two values of C.** The pre-registration text gives C = 0.8609, measured on 48 µm cells. The test recomputed C on the 9.612 µm pixels as 0.868948 and applied the bars with that. With 0.8609, ft6k's nAUC would be 0.860. Either way it clears 0.771.

**Depth-order controls** (ft6k, same recipe, same pixels):

| control | AUCsup | hp r 192 µm vs null max | hp r 48 µm vs null max |
|---|---|---|---|
| forward order | 0.5297 | +0.0182 vs +0.0392, inside | +0.0083 vs +0.0154, inside |
| 21-layer shuffle | 0.5308 | +0.0288 vs +0.0337, inside | +0.0193 vs +0.0108, **outside** |
| 41-layer shuffle | 0.5687 | +0.0268 vs +0.0338, inside | +0.0107 vs +0.0071, **outside** |

**Sanity gates.** Both failed before scoring.
- **Gate 1:** the model-free render must sit inside its own null. It sits inside on raw correlation, but outside on the high-passed bands at blur 0:
  - 48 µm: +0.0110 against a null max of +0.0060;
  - 192 µm: +0.0254 against +0.0177.
  - On w00big the same render was inside its null everywhere.
- **Gate 2:** the released checkpoint's alignment against the 2.4 µm map must peak within 1 px of zero shift.
  - It peaks at (+1, +3) px.
  - The surface is flat: r is 0.5401 at the peak and 0.5387 at (0, 0).

**Verdict: AMBIGUOUS.**
- At 192 µm, the band the pre-registration treats as decisive, every condition for going on is met: nAUC 0.8409, hp r far above its null, and all three controls inside their nulls.
- The day-1 plan also listed the 48 µm band, and I read it as binding where the newer pre-registration is silent. Under that conservative reading, both shuffles fail.
- The failed sanity gates also cap any GO at AMBIGUOUS.
- The pre-registered consequence of AMBIGUOUS: run the letter read anyway, and mark the result ambiguous.
- [read] At 48 µm the model-free render already exceeds its null (+0.0145, z 3.8). The 48 µm control failures may therefore come from render-level structure rather than from depth-order ink reading. I have not tested that.

## B. The letter read on 252405

### Protocol

- **The reference.** "Reader A", a fresh AI agent, marked letters on ScrollPrize's 2.4 µm map of 252405. It saw no 9 µm tile.
  - It counted 10 letters as clear or probable, and 8 of those have centres inside the 9 µm render crop.
  - Those 8 (4 clear, 4 probable) are the scorable reference, with ids L1-2 to L2-7.
  - The reference was locked by its sha256 (b12f3b24…) before the ceiling read.
  - It is not published, because it is a partial transcription (see Licences and data terms).
- **Tiles:**
  - 8 mm square (832 px at 9.612 µm);
  - 25 real tiles on a 5 × 5 grid, overlapping by at least 4.5 mm, so every letter is whole in at least one tile;
  - one fixed display transform for every tile, real or null.
- **Null tiles:**
  - 25 phase-scrambled tiles per map: the Fourier phase is randomised on a mirror-extended window, followed by 40 IAAFT rounds, so the brightness histogram matches exactly;
  - for 9 µm models, also 25 depth-shuffled tiles: the same model run on a depth-permuted stack.
- **Readers:** fresh Claude agents run by a workflow script, one tile per agent and three at a time.
  - Each reader sees only its own tile and zooms of it.
  - Readers are blind to whether a tile is real or null, and never see the reference or the key.
  - A reader names each letter it can, with a box and a confidence.
- **Match:** the same letter (or one of the reference's listed alternatives), with the box centre within 0.5 mm of the reference box centre, one-to-one. Same-letter calls from overlapping real tiles are merged first.
- **Code:** [`code/letter-read/score.py`](code/letter-read/score.py) scores the calls, and `note4-verdict.py` computes the verdict.

### How the bar was set, in order

1. **The pre-registration**, written before any 9 µm model ran on 252405, required three things:
   - at least 6 of 8 letters matched (70% of 8, rounded up);
   - precision of at least 0.8;
   - zero calls on null tiles.
2. **Note 2**, written before any 9 µm map existed, ordered a ceiling read of the 2.4 µm map first, through the same readers. Had the ceiling matched fewer than 6, the bar would have dropped to 70% of the ceiling.
3. **The ceiling read** matched 6 of 8, so the bar stayed at 6. But it also made 16 calls on 9 of the 25 null tiles, for a precision of 0.22.
   - So the 2.4 µm map itself fails "precision ≥ 0.8 and zero null calls".
   - These readers call letters in ink-like texture at some rate, whatever map they look at.
4. **Note 4** therefore replaced only the false-alarm conditions, setting them relative to the ceiling:
   - at most 24 calls on each 25-tile null type (the ceiling's 16 plus 50%);
   - precision of at least 0.11 (half the ceiling's).
   - The matched bar of 6 did not change.
- **Disclosure:** note 4 was written after the day-1 pixel scores and overview images of the 9 µm maps existed, but before any 9 µm tile pool or letter read.
- **The verdict does not depend on note 4.** Arm A fails the matched bar, which note 4 did not touch. It would also fail the original conditions: its precision is 0.20, and it drew 5 calls on null tiles.

### Results ([`results/letter-read-252405.csv`](results/letter-read-252405.csv))

| map | role | matched of 8 | precision | calls on phase nulls (tiles hit) | calls on depth nulls (tiles hit) | verdict |
|---|---|---|---|---|---|---|
| ScrollPrize 2.4 µm map | ceiling | 6 | 0.22 | 16 (9 of 25) | — | — |
| TAUIL ft6k | arm A | 3 | 0.20 | 4 (4) | 1 (1) | NO-GO on matched |
| released `ink_9um` s43-75k | baseline | 0 | 0.00 | 1 (1) | none built | comparison only |
| student v1 (section C) | report-only | 3 | 0.15 | 27 (17) | 9 (6) | NO-GO on matched and on nulls |

- **Arm A:** 3 of 8 letters at 9 µm, against 6 of 8 at 2.4 µm on the same text.
  - One of the three was named through the reference's listed alternative.
  - Its matches sit 0.09 to 0.14 mm from the reference centres.
- **Diagnostics,** which do not decide anything:
  - **Located in box** (a call of any letter inside a reference box): ceiling 8, arm A 7, baseline 2, student 6. Of the 7 boxes where ft6k has a call, 3 meet the matching rule, 1 more names the reference letter but 0.61 mm from the reference centre (outside the 0.5 mm rule), and 3 are misnamed.
  - **Above a confidence floor of 0.40**, fixed from the ceiling's null calls before any 9 µm read:

    | map | matched | null calls |
    |---|---|---|
    | ceiling | 6 | 0 |
    | arm A | 0 | 0 |
    | student | 1 | 3 |

  - **Median reader confidence**, calls on real tiles against calls on phase nulls, computed from the reader calls:

    | map | real tiles | phase nulls |
    |---|---|---|
    | ceiling | 0.40 | 0.21 |
    | arm A | 0.30 | 0.25 |
    | student | 0.30 | 0.30 |

  - **Stroke width:** the median autocorrelation half-width inside the 8 reference boxes ([`results/letter-halfwidth-252405.csv`](results/letter-halfwidth-252405.csv)):

    | map | median half-width |
    |---|---|
    | ceiling | 327 µm |
    | baseline | 351 µm |
    | arm A | 399 µm |

    Arm A misnamed three of the letters it located (L1-3, L2-3 and L2-5). Those three broadened the most against the ceiling, by +37% to +59%. The three it matched broadened by +9% to +31%. A fourth (L2-7), which it named with the reference letter but 0.61 mm from the reference centre, broadened by +18%. [read] Arm A's misses look like strokes broadened into blobs.
- Letter identities, tiles and label images are deliberately left out (see Licences and data terms).

## C. Same-scroll 2.4 → 9 µm student (v1): NO-GO

**Design**, pre-registered in [`prereg/prereg-student.json`](prereg/prereg-student.json) and its amendment 1:
- **Base:** TAUIL's ft6k. I fine-tuned it on native 9.366 µm renders of w00 and 552896, with ScrollPrize's 2.4 µm map of each as the soft target.
  - 552896 carries w00's text, so the two are two renderings of the same papyrus, not new text.
- **Loss:** L = BCE + 0.5 × (1 − ρ_hp), where ρ_hp is a high-pass Pearson term at σ 48 µm.
- **Training:** 12,000 steps at batch 8, with no early stopping. A scramble twin was trained identically against a teacher rolled by 4 mm.
- **Held-out test:** 252405 was excluded by geometry: no training window within 228 µm in 3D or 2,096 µm along the sheet.
  - Before the test phase, the only 252405 files used were its mesh geometry and the day-1 result files that set the bars.
  - Amendment 1 discloses one incidental exposure: a text search across one folder also ran over 252405's label files.
- **Same-scroll data:** villa #1739 allows training on 2 µm data of the same scroll when the model is applied at 9 µm. Every input the student sees is 9.366 µm. The 2.4 µm data enter only as the teacher map.

**Results** ([`results/student-v1-252405-scores.csv`](results/student-v1-252405-scores.csv), [`results/student-v1-gates.csv`](results/student-v1-gates.csv)):

| gate | result | numbers |
|---|---|---|
| G1: AUCsup gap closure toward the teacher | FAIL | Student 0.8252, base 0.8102, teacher 0.8689. Closure 0.2549 (bar 0.5). Bootstrap 2.5/50/97.5 percentiles −0.3569 / 0.2472 / 0.7218 (bar on the 2.5th: 0.25). Student − base 95% [−0.0118, +0.0451]. |
| G2: high-pass closure on the human-label key | UNINFORMATIVE | On this key the 2.4 µm teacher itself scores below the base: +0.1972 vs +0.2073. The student scores +0.1823. |
| G3: student hp r 192 µm above its null | PASS | +0.4543 vs a null max of +0.0284 |
| G4: depth-order controls | FAIL | Both depth shuffles are outside their nulls at 192 µm: +0.0438 vs +0.0296, and +0.0453 vs +0.0300. The forward control and the 41-layer shuffle are outside at 48 µm. |
| G5: scramble twin flat | PASS | twin nAUC 0.0072 |
| G6: blind letter read | NO-GO (report-only) | 3 of 8 (bar 6). 27 calls on the 25 phase nulls (cap 24) and 9 on the depth nulls. Precision 0.15. |

**The letter read (G6), in detail:**
- The 3 matches sit 0.31 to 0.46 mm from the reference centres, looser than arm A's 0.09 to 0.14 mm. Only one of them (L1-2) is among arm A's three.
- The 27 calls landed on 17 of the 25 phase-null tiles. This is the first map in this work to fail the null condition.
- Calls on real tiles and calls on null tiles carry the same median confidence, 0.30.
- [read] The student draws letter-like texture on scrambled inputs as well as on text, and readers cannot tell its null texture from its text. That fits G4: the student reads more from depth-destroyed stacks than its base does.

**The 48 µm high-pass gain is not independent evidence.** On the ScrollPrize key at 48 µm, the student beats its base by:
- 2.09× at blur 2 (+0.2672 vs +0.1280);
- 3.0× at blur 0 (+0.2510 vs +0.0827);
- 1.7× at blur 4 (+0.2740 vs +0.1631).

This band, against this key, is the student's own training loss, so the gain says little on its own.

## D. Student v2: a second pre-registered attempt (also NO-GO)

After v1 failed its depth controls, I pre-registered a second recipe before training it ([`prereg/prereg-student-v2.json`](prereg/prereg-student-v2.json); the frozen original's sha256 is f348e4b89b19af4a682c65dfb672661296f909183ead703237711e547faa33d7, and the published copy differs only by removed local folder prefixes and two wording changes). It is identical to v1 except for one term: every training window is also fed with its 17 depth layers randomly reordered, and the student is penalised (lambda 0.5) for predicting ink on that depth-destroyed copy. The clean data stream was checked to be byte-identical to v1's at every one of the 12,000 steps.

Because v2 was designed after I had seen v1's numbers on the held-out segment, its test is a **second look** at 252405, and the pre-registration made the bar stricter: the baseline is v1, the closure must reach 0.5, and its bootstrap 1.25th percentile (Bonferroni for two looks) must stay at or above 0.25.

| Map on 252405 | AUCsup | hp r 192 um vs the 2.4 um map | hp r 48 um |
|---|---|---|---|
| 2.4 um map (teacher) | 0.869 | — | — |
| student v2 | 0.832 | +0.437 | +0.238 |
| student v1 | 0.825 | +0.454 | +0.267 |
| TAUIL ft6k | 0.810 | +0.350 | +0.128 |
| v2 scramble twin | 0.389 | -0.005 | -0.012 |

- **G1 FAIL:** closure over v1 is 0.146 (bar 0.5); the 1.25th bootstrap percentile is -0.388 (bar 0.25). The v2 minus v1 difference has a 95% interval of [-0.004, +0.018], which cannot be told from zero.
- **G4 FAIL:** at 192 um both depth-shuffle controls still read structure above their nulls (shuffle-21 +0.046 against a bar of +0.024; shuffle-41 +0.035 against +0.026), and the forward-order control is outside at 48 um. The penalty lowered how much ink v2 predicts on shuffled input (the shuffle-21 map's spread fell from 8.8 to 3.4 grey levels) but not the structure it reads from it, a limit the pre-registration named before training: the penalty acts on magnitude, while the control is a scale-free correlation.
- **G3 and G5 pass;** G2 was uninformative, fixed before any v2 number.
- [read] The v2 scramble twin's map is nearly flat but anti-correlated with the labels (AUCsup 0.389). G5 only catches a twin that gains, so it passes; I did not measure the cause.
- The letter read for v2 was not run before submission. Under the pre-registration it could not have turned a failed numeric gate into a GO.

**What the two students show together:** same-scroll distillation moves pixel AUC up in small steps (0.810, 0.825, 0.832) and sharpens strokes, but neither recipe produced a map whose letters blind readers could name more often, and both learned some structure that does not depend on the depth order of the scan. [read] A 9 um student that reads unseen text legibly probably needs more than a loss term: more paired training text, or surfaces seated closer to the ink layer.

## E. Registration and render checks (PHerc0841)

**Registration of the 2.403 µm scan into the 9.366 µm frame** ([`code/fl-register-0841.py`](code/fl-register-0841.py); the affine is [`results/registration-pherc0841-affine.json`](results/registration-pherc0841-affine.json)):
- 30 blocks were matched by normalised cross-correlation.
- Median residual: 10.8 µm in-sample and 12.3 µm held out (max 37.2 µm).
- On the same 30 blocks, the catalogue transform sits a median 15.7 µm (max 40.2 µm) from where the cross-correlation placed them.
- On 50,000 control-mesh vertices, the two transforms differ by a median 6.0 µm (p95 8.6 µm, max 9.3 µm).
- All my 9 µm renders sample the 9.366 µm volume through this affine.
- [read] It agrees with the catalogue transform to about 6 µm, with a small block-level difference on one scroll. That is not a demonstrated advantage.

**Render check on the w00big frame** (about 1.0 cm²):
- ScrollPrize's own native 9.366 µm surface volume equals my render at r 0.985, after a 9 px origin shift.
- `ink_9um`, which never trained on PHerc0841, correlates with the 2.4 µm Hecate map at r +0.470 (null max +0.211). That is a correlation, not a reading of letters.
- A higher figure from the Hecate 9.6 µm model is left out, because that model may have trained on PHerc0841.

## F. PHerc1203 at 2.4 µm: a scout null, not First Letters evidence

- **Coverage:** about 12.7 cm² of surface was actually shown to readers, out of 24.8 cm² rendered. No letter was named.
- **Sensitivity is unmeasured.** The positive control was one PHerc0841 frame, probably on the 2.4 µm model's own training labels, so its "2 of 2" is not a sensitivity estimate.
- **Not admissible:** a 2.403 µm result does not count for First Letters (villa #1739).
- **Smaller than prior work:** kadenpool's earlier public null on PHerc1203 at 2.4 µm covers about 67 cm².
- This is not evidence that there is no ink.

## Pre-registration record

| when (2026, UTC−5) | what | sha256 |
|---|---|---|
| 10-05 14:03-14:16 | Amendments 1-5 to the day-1 plan, before any recipe ran on 252405 | `AMENDMENT-1.txt` 330c64be… |
| 10-05 14:24 | Day-1 pre-registration, first version | `day1-prereg-v1.json` 2d314b1e… |
| 10-05 15:30 | Reader A marks the reference on the 2.4 µm map, blind to every 9 µm map | not published |
| 10-05 15:54 | Pre-registration v2. It supersedes v1 where they differ, and was written before any 9 µm model ran on 252405. | `prereg-v2.json` 49633d68… |
| 10-05 16:22-16:48 | Notes 1-3: diagnostics, ceiling read first, mirror nulls, the reference | see HASHES.md |
| 10-05 17:02 | Recipe frozen on w00big | `recipe-frozen.json` 72ad196a… |
| 10-05 17:16 | Reference locked | b12f3b24… (not published) |
| 10-05 18:24 | Day-1 pixel-level result on 252405 | — |
| 10-05 19:12 | Ceiling read scored. Note 4 sets the false-alarm conditions. | `prereg-v2-note-4.txt` 81e5d63a… |
| 10-05 19:43 | 9 µm pool keys recorded, before any reader ran | see HASHES.md |
| 10-05 23:43 | 9 µm letter read: NO-GO | — |
| 10-06 01:2x | Student pre-registration frozen | `prereg-student.json` f3ebf0ab… |
| 10-06 02:36 | Student amendment 1, before any training step | 3b0f1bc4… |
| 10-06 05:2x | Student numeric test: NO-GO | — |
| 10-06 05:53 | Student pool key recorded, before any reader ran | see HASHES.md |
| 10-06 07:51 | Student letter read (report-only) | — |

**Two amendments that changed something, both made before the numbers they touch existed:**
- **v1 to v2.** v1 claimed it had been frozen before the 252405 render and the w00big selection scores existed. It had not been, and v2 records that correction. v2 also restated the AUC bars as nAUC, because C on 252405 is lower than on w00.
- **Student amendment 1.** The pre-registered check that my full w00 render reproduces the w00big render failed on a 5 px border, where the smaller crop had used one-sided normals. The interior matched exactly. The amendment moved the check to the interior before any training step. It also disclosed two things. The freeze had read the day-1 result files to set the bars. A text search across one folder had run over 252405's label files, and a directory listing had touched one label-grid header. No 252405 label, map or model output entered training.

## Limits

- **Scope:** one scroll, one test segment and 8 scorable letters.
- **The reference** is itself an AI reading of a 2.4 µm model map.
- **The readers are AI agents.**
  - Their precision is low even on the 2.4 µm ceiling (0.22), so the bars are relative to that ceiling, not absolute.
  - A re-run with fresh readers will not reproduce the calls bit for bit.
  - The calls are kept back because they name letters. Only the counts are public.
- **Failed gates:** day 1's sanity gates failed, and so did the 48 µm controls, possibly for a render-level reason that is untested.
- **The student's failure covers one recipe:** two renderings of the same text, and one loss.
- **Verification:** the freeze timestamps are in my private history, and most published pre-registration files are redacted copies.
- **No uptake yet:** no merged PRs, and no community use.
- **Formats:** the render step reads the public OME-Zarr volumes and tifxyz meshes directly. Everything after it exchanges NumPy `.npy` arrays on the render grid. There is no OME-Zarr or TIFF output yet.
- **The code was written for one machine.** It is a reproducible record, not a packaged tool.

## Using and reproducing this work

- [`REPRODUCE.md`](REPRODUCE.md) lists the data ids, the model links with their licences and hashes, the environment, and the commands in order.
- **To score your own 9 µm map with the letter-read protocol:**
  1. Render it on the same grid ([`code/day1-render-9um.py`](code/day1-render-9um.py)).
  2. Cut a pool with [`code/letter-read/make-pool.py`](code/letter-read/make-pool.py).
  3. Run the blind readers with [`code/day1-letter-read-workflow.js`](code/day1-letter-read-workflow.js).
  4. Score with [`code/letter-read/score.py`](code/letter-read/score.py) against a locked reference.
- The tests run on synthetic data: `python code/letter-read/test_letter_read.py`. One check reads the w00 render grid and needs the data.

## Credit and prior work

- **ScrollPrize / Vesuvius Challenge:** the scans, the ink-labels dataset, the 2.4 µm ink maps, `ink_9um`, the Hecate models, and the villa inference code (`koine_machines`), which my map runs call unchanged.
- **TAUIL-Abd-Elilah:** cross-scan-ink-transfer, the ft6k checkpoint, and the PHerc0841 result this work replicates.
- **ShribyrLabs:** vesuvius-reports, the reader-ft-s42 checkpoint, and the letter-scale high-pass score that my hp r follows.
- **Nieuwlaar** (ink9um-dense-native) and **domenicor046** (ink9um-dense): checkpoints used in selection only.
- **pmh47:** the maintainer ruling in villa issue #1739.
- **kadenpool:** the earlier public PHerc1203 null at 2.4 µm.

## Licences and data terms

- **Code:** MIT ([`LICENSE`](LICENSE)), for my own code only. Checkpoints are linked, never rehosted.
- **Checkpoints** (sources read 2026-10-06):
  - **TAUIL ft6k:** the repository is MIT. Its README says the training data are Vesuvius Challenge data under CC BY-NC 4.0, and the weights carry no licence line of their own.
  - **`ink_9um`:** MIT, per its Hugging Face metadata.
  - **ink9um-dense (domenicor046):** the metadata says MIT, but the card says the checkpoints remain subject to the terms of their sources. Selection only. The code and the pre-registrations label it "MIT" or "MIT per card". Those are my record's labels and they overstate the card; the card's own provenance sentence governs.
  - **ink9um-dense-native (Nieuwlaar):** MIT, per its metadata. Selection only.
  - **reader-ft-s42 (ShribyrLabs):** no stated licence for the checkpoint. Reference arm only.
  - **My student:** not released. It would inherit ft6k's terms and the data terms.
  - **Hecate (ScrollPrize's 2.4 µm and 9.6 µm ink models):** I ran them only to make comparison maps. Their outputs appear here only as reference numbers (section E and the selection table), never in a decision, and no Hecate weights or maps are rehosted.
- **Scan data:** CC BY-NC 4.0 (scrollprize.org/data). Cite as: Angelotti et al., *Vesuvius Challenge - CT Scans of Herculaneum Papyri*, Vesuvius Challenge, with the full author list as the data page's "How to Cite" section gives it.
- **The data terms bar publishing full or partial transcriptions, or ink-label images, without written consent.**
  - This report and its repository therefore give counts only.
  - They contain no letter identities, positions, tiles, renders, maps, label images, checkpoints or scan data.
  - Reference letters are named by id only (L1-2 to L2-7).

## AI assistance

I used Claude (Anthropic) as a coding and analysis assistant throughout this work.
- It wrote most of the code, drafted the pre-registration files, ran the experiments on my machine under my direction, and drafted this report.
- Where my plan was silent, it filled in details, and the pre-registration files mark those details "[machine]" or "machine_fill_in".
- A second Claude session acted as a reviewer and made some protocol rulings, such as the switch to mirror-symmetric phase nulls.
- The blind letter readers are separate, fresh Claude agent instances run by a workflow script. They are part of the instrument, and their limits are listed above.

[author: confirm this section's wording before release]
