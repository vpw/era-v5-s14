# S14 TODO — Mixture-of-Experts (dense → MoE)

## ▶ STATUS (2026-09-29): run complete on the T4 (58.8 min, ≈$0.81, box stopped and verified); README built. Remaining: ship.

| phase B run (+50M tokens) | val start → end | vs control | tok/s | peak |
| --- | --- | ---: | ---: | ---: |
| MoE, drop-upcycled (r=0.5, 200 sampled steps) | 2.2278 → 1.4523 | +0.0078 | 45.5K | 11.43 GiB |
| dense continued (control) | 1.7119 → 1.4445 | — | 97.0K | 6.70 GiB |
| MoE, hard top-k from start | 2.2278 → 1.4437 | −0.0008 | 45.6K | 11.43 GiB |

Dense phase A ended at 1.7119. The conversion jump is +0.143 at r=0 and +0.516 at r=0.5. The
MoE recovers past the dense checkpoint after 9.8M tokens and its gap to the control narrows
monotonically, but it ties rather than wins. There are no dead experts. **Candidate
follow-up, not run:** r=0 or r=0.25, the obvious test of whether the re-draw handicap is what
costs the win.

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

## ▶ DECISIONS — settled 2026-09-29 (user picked the recommendation on D1, D3/D4, D6, R4; D2 and D5 as proposed)

- [x] **D1. Compute: EC2 T4** via `era-v5-gpu-run` (`SSH_KEY=~/.ssh/id_ed25519`), fp16 +
      GradScaler, router in fp32. Estimate is under 1h (≈$1). Correctness gates and smoke tests
      run on CPU here first.
- [x] **D2. Base: reuse S13's pipeline** (TinyStoriesV2-GPT4, BPE-8192, nanoGPT, `tools/`) with a
      **shallower, wider shape**: d=384, L=8, 6 heads, T=512, tied embeddings, 4× GELU MLP
      (hidden 1536), ≈17M params. S13's L=24 served reversibility; MoE benefits from FFN width.
- [x] **D3. MoE shape:** per layer, **1 shared expert (width 768) + 16 routed experts
      (width 192), top-4**. The active FFN width is 768 + 4·192 = 1536, equal to the dense
      model (the lesson's §4 rule), and total FFN width is 3840, ≈2.5× dense.
- [x] **D4. Conversion:**
      - The shared expert is the first half of the dense neurons.
      - Each routed expert is a random half-overlapping subset of the 1536 dense neurons, with
        **50% of its neurons redrawn** (drop-upcycling, r = 0.5).
      - The loss is expected to jump at the switch. Measure the jump and its recovery.
      - **Copy-upcycling is a correctness gate only.** Assert that the MoE output equals the
        dense output at conversion, which proves the dispatch/combine code.
- [x] **D5. Router:**
      - **Softmax** over all experts, then top-k, then renormalize (the Qwen3 reference model).
      - fp32, with a small init (0.1× scale, §7).
      - **Aux-loss-free bias balancing** (γ = 0.001, sign rule, bias used only for selection,
        load counted over the whole batch).
      - Dropless, with no auxiliary loss.
      - **Probabilistic top-k** (sampling k without replacement from the router probabilities)
        for the first ~200 MoE steps, then hard top-k.
- [x] **D6. Budget:** dense **50M** tokens, with a checkpoint at the end. From that checkpoint,
      run **MoE +50M** and a **continue-dense +50M control** on the same data order.
      - The schedule is **WSD**: warmup, then a constant LR through the switch, then each
        continuation decays over its last 20%. S13's cosine-to-10% would have starved the MoE
        of learning rate.
      - Needs 100M train tokens pre-tokenized (S13 had 50M).
- [x] **R4 added:** the same MoE run with **hard top-k from step 0 after conversion**, to show
      the §15 clone-family collapse (dead-expert counts) and that probabilistic top-k fixes it.
      About +25 min on the T4.

## ▶ BUILD

- [x] MoE FFN module: fp32 router, top-k plus renormalize, per-expert bias used only for
      selection, dropless dispatch (loop over experts or grouped index), optional shared expert.
- [x] `dense_to_moe(model, method=…)` conversion, plus the matching router init (small, or tiled
      with noise).
- [x] **Correctness gates** (notebook asserts):
  - [x] At conversion, the MoE output equals the dense output where the method preserves
        function; otherwise the loss bump is measured and reported.
  - [x] Bias balancing moves load toward uniform on a synthetic skewed router (sign rule check).
  - [x] Parameter counts: total vs active, computed and asserted against the formula.
- [x] Per-step logging to a committed file: phase, step, tokens, train loss, val loss at eval
      points, lr, and per-layer MaxVio / dead-expert count / load histogram snapshot.

## ▶ RUNS

- [x] R1. Dense phase to its token budget. Save a checkpoint.
- [x] R2. Convert to an MoE, continue training, and show the loss keeps dropping.
- [x] R3. Control: continue the dense model for the same tokens.
- [x] R4. MoE with hard top-k from the switch (no probabilistic window), to show the §15
      clone-family collapse.

## ▶ RESEARCH / CITATIONS

- [x] Cite §15's papers in the README. Verify each ID and date with the arXiv API (one ID per
      request); on this box `arxiv-library`'s corpus is absent and its search 406s. Pull any
      specific claim from the PDF with `pdftotext`, not from memory. The verified list is in
      `docs/s14-transcript-summary.md`.

## ▶ WRITE-UP AND SHIP

- [x] README via `README.tmpl.md` → `build_readme.py`. It needs:
  - [x] the loss curve across the switch, marked;
  - [x] MoE vs continue-dense;
  - [x] expert-load evolution;
  - [x] the dimension bookkeeping (total/active params);
  - [x] what did and did not work.
- [x] Commit the notebook, `results.json`, `logs/` (training logs are mandatory), and `assets/`.
- [x] The user created `github.com/vpw/era-v5-s14`. Subtree split `94e9e56` pushed and verified anonymously (2026-09-29). Submit in Axiom (user),
      and submit in Axiom before 2026-10-03 07:00.
