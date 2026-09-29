# S14 TODO — Mixture-of-Experts (dense → MoE)

## ▶ STATUS (2026-09-29): scaffolded, not yet started

**Due Sat 2026-10-03 07:00** (1000 pts, resubmission allowed, one GitHub README link field,
"The Repo MUST have training logs").

## ▶ SCAFFOLDING

- [x] **Session verified** (2026-09-29). The lesson heading reads "Session 14:
      Mixture-of-Experts", which matches this folder.
- [x] **Assignment captured** → `S14-assignment.md`: the brief verbatim, the Axiom block, and the
      instructor's framing with timestamps. The Rubric tab is an empty table ("Total 1000").
- [x] **Lesson captured** → `resources/s14-session.md`, all 21 sections. `get_page_text` cut
      the last ~400 characters, which were read via JS.
- [x] **Transcript captured** → `resources/s14-transcript.md`, 123KB, from YouTube auto-captions
      (there is no Google Doc this session). yt-dlp is bot-blocked on this box, so it was read
      from the "Show transcript" panel in the user's browser. The total char count matched the
      browser's exactly (122,598).
- [x] **Widget extraction judged unnecessary** (see CLAUDE.md Conventions).
- [x] **Working set written**: `CLAUDE.md`, `AGENTS.md`, this file.
- [x] **Branch cut**: `s14-moe`, from `s13-reversibility`.
- [x] Copied `tools/` (unchanged) and `requirements.txt` from S13. `.gitignore` is S13's plus
      `docs/`.
- [x] **Transcript summary** → `docs/s14-transcript-summary.md` (gitignored). It has the class
      arc with timestamps and links to every referenced paper and model, with arXiv IDs
      verified via the API.

## ▶ DECISIONS — open

- [ ] **D1. Compute lane.** Options: this 2-core CPU box (a few-M-param run, as in S11, slow but
      free); the EC2 T4 via `era-v5-gpu-run` (as in S13, ~$0.83/h, fp16 only); or Colab.
      Depends on D2.
- [ ] **D2. Base model and data.** Reuse S13's setup (TinyStories, BPE-8192, nanoGPT; the
      instructor allowed it), or something smaller. S13's shape was d=256, L=24, which is deep
      and suited reversibility. For MoE the FFN share matters more than depth, so a shallower,
      wider shape may show upcycling better at the same budget.
- [ ] **D3. MoE shape.** Expert count E, expert width, top-k, shared expert or not. §4's rule is
      k × width = dense FFN width, so active compute matches the dense model. Example: dense FFN
      4d = 1024 → 8 experts × 128 with k = 2 + 1 shared, or 16 × 64 with k = 4. Decide the MoE
      first, as the instructor said.
- [ ] **D4. Upcycling method.** Copy (function-preserving, but clones must be broken apart),
      partition (Lightning LM's shared + overlapping-random-half routed), or drop-upcycling
      (r = 0.5). Pick one as the main arm; running a second as a comparison is optional.
- [ ] **D5. Router.** Softmax (Qwen3, our reference) or sigmoid (DeepSeek-V3 and later).
      Aux-loss-free bias balancing, γ = 0.001, is in either way. Decide whether to use
      probabilistic top-k for the first N steps after conversion, which is the lesson's fix for
      clone collapse.
- [ ] **D6. Token budget and control.** Split dense vs MoE phase tokens, and include a
      continue-dense control for the same MoE-phase tokens (Sparse Upcycling's own baseline).

## ▶ BUILD

- [ ] MoE FFN module: fp32 router, top-k plus renormalize, per-expert bias used only for
      selection, dropless dispatch (loop over experts or grouped index), optional shared expert.
- [ ] `dense_to_moe(model, method=…)` conversion, plus the matching router init (small, or tiled
      with noise).
- [ ] **Correctness gates** (notebook asserts):
  - [ ] At conversion, the MoE output equals the dense output where the method preserves
        function; otherwise the loss bump is measured and reported.
  - [ ] Bias balancing moves load toward uniform on a synthetic skewed router (sign rule check).
  - [ ] Parameter counts: total vs active, computed and asserted against the formula.
- [ ] Per-step logging to a committed file: phase, step, tokens, train loss, val loss at eval
      points, lr, and per-layer MaxVio / dead-expert count / load histogram snapshot.

## ▶ RUNS

- [ ] R1. Dense phase to its token budget. Save a checkpoint.
- [ ] R2. Convert to an MoE, continue training, and show the loss keeps dropping.
- [ ] R3. Control: continue the dense model for the same tokens.
- [ ] (optional) R4. A second upcycling method, or hard vs probabilistic top-k, to show
      clone-family collapse.

## ▶ RESEARCH / CITATIONS

- [ ] Cite §15's papers in the README. Verify each ID and date with the arXiv API (one ID per
      request); on this box `arxiv-library`'s corpus is absent and its search 406s. Pull any
      specific claim from the PDF with `pdftotext`, not from memory. The verified list is in
      `docs/s14-transcript-summary.md`.

## ▶ WRITE-UP AND SHIP

- [ ] README via `README.tmpl.md` → `build_readme.py`. It needs:
  - [ ] the loss curve across the switch, marked;
  - [ ] MoE vs continue-dense;
  - [ ] expert-load evolution;
  - [ ] the dimension bookkeeping (total/active params);
  - [ ] what did and did not work.
- [ ] Commit the notebook, `results.json`, `logs/` (training logs are mandatory), and `assets/`.
- [ ] The user creates `github.com/vpw/era-v5-s14`. Subtree split, push, verify anonymously,
      and submit in Axiom before 2026-10-03 07:00.
