# S14 Assignment — Session 14: Mixture-of-Experts

Captured 2026-09-29 from the Axiom assignment page
(`https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/assignments/cmumdrhxw1c0j09l7px6v6d8d`),
page title "Session 14 Assignment QnA". The lesson's §21 carries the same text.

## Brief (verbatim)

> Train a Linear model and convert that into an MoE! Your call on model size and data trained on,
> but must show they continue to train and reduce loss!

## Submission block

- **Status:** Available. **Due Sat, Oct 3, 2026, 7:00 AM** (IST). **1000 points.** Resubmission
  allowed.
- **One field: "GitHub README Link"** (1000 pts), with a caption field. The field's note says:
  **"The Repo MUST have training logs."**
- Checkbox: "I tested this link in an incognito window — it's publicly accessible (not private)."
- **Rubric tab:** a Criterion / Description / Max Points table with no rows, only "Total 1000".
  It adds nothing beyond the brief.

## Instructor's framing (from `resources/s14-transcript.md`)

The class video is auto-captioned, so these are close paraphrases with the timestamps to check
against.

- **[2:11:34] The assignment as stated in class:** train a linear model and then convert it into an
  MoE of your own choice. You decide the model size and the data. Show two things: **first, that it
  continues to train** after the conversion, and **second, that the loss actually drops**. The MoE
  will be the bigger model, so **decide the MoE first** and size the linear model from it, so that
  it fits "on your laptop, your Mac or the Colab of your own choice". He calls it "a very simple
  experiment": you learn the dynamics of the change, what to take care of, and which dimensions you
  need to be aware of.
- **[2:18:11] Q&A:** a student asked whether they could reuse "last time's 20 million parameters
  thing" (S13's ~20M-param model). Answer: **"you can work on that one or even a bigger one."**
  Whether Colab will do the job was left as a rhetorical question.
- **What "linear model" means.** Throughout the class he says "linear model" for the **dense**
  model, one with a single ordinary feed-forward block per layer ("we have to start with a linear
  model … 2048 as our hidden dimension", [1:44:29]). So the brief is **dense → MoE upcycling**
  (lesson §15), not a literal single-layer linear regression.
- **[1:44:29–1:50:40] The growth recipe he described**, the Lightning LM "V4" path, is the template:
  - Train the dense model first, for a share of the run, to get the embeddings right.
  - Then convert. Map the dense FFN into a **shared expert** (e.g. the first 1024 of 2048: crop the
    first half, take a random half, or half copied and half fresh).
  - Build the routed experts by one of three methods, all in §15:
    - **copy**: every expert is the whole FFN;
    - **partition**: each expert is a slice;
    - **drop**: each expert is a copy with half its neurons redrawn.
  - A student raised the dimension point, and he confirmed it: identical copies keep the full width
    and are **weighted-averaged**, while partitioned slices are narrower and together rebuild the
    width.
- **[1:53:47–1:55:56] The failure to avoid:** clones that are **identical** under **hard top-k**
  routing collapse. Dead experts went 27 → 168 of 460 in a few hundred steps. The fix is to redraw
  neurons (drop-upcycling) and to use **probabilistic top-k selection** early on, so every clone
  gets gradient before the router settles.
- **[1:35:02–1:38:08] Balancing:** he wants **auxiliary-loss-free balancing** (a per-expert bias
  used only for selection, γ = 0.001), **not an auxiliary loss**. He told how a student's
  reintroduced aux loss wasted about two days of Lightning LM compute. He also wants **no token
  dropping** ([2:05:17]), and the router kept in **fp32** ([2:06:23]).
