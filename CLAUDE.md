# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 14 (S14) assignment of the ERA V5 course (The School of AI). The session topic is
**Mixture-of-Experts**. The instructor called it the last content session before the capstone,
and the next class does group division. The lesson has 21 sections and uses one reference model
throughout: the **Qwen3-30B-A3B shape**. It has 48 layers, d=2048, and GQA with 32 query and 4 KV
heads of dim 128. Each layer holds 128 SwiGLU experts of width 768, with top-8 and no shared
expert. That gives 30.53B total and 3.35B active parameters.

The arc in brief:
- **§0–2:** the MoE idea (choose, balance, place), the reference model's parameter count, and
  terminology.
- **§3–4:** the MoE layer y = Σ_{i∈top-k} g_i·E_i(x), with SwiGLU experts. 8 × 768 = 6,144, so the
  active experts have the width of one dense FFN.
- **§5–6:** why MoE works (compute follows active, memory follows total) and the total/active
  ratio trend (3.6 → 34.5).
- **§7:** the router. Softmax vs sigmoid (vs √softplus), top-k then renormalize, routed scaling
  factor, fp32 router, small init.
- **§8:** fine-grained experts and shared experts.
- **§9:** what experts learn (token type and syntax, not subject).
- **§10–14:** balancing. Collapse and MaxVio, capacity and dropping (obsolete, now dropless),
  Switch aux loss and z-loss, **aux-loss-free bias balancing** (b_i ← b_i + γ·sign(mean −
  load_i), γ = 0.001), and balancing scope (whole batch beats micro-batch).
- **§15: growing an MoE**, the section the assignment is about. Upcycling a dense model by
  **copy** (sparse upcycling), **partition** (Qwen1.5-MoE), or **drop** (drop-upcycling, r = 0.5
  best), and growing the expert count. Lightning LM's path was 2B dense → 5B with 20 routed + 1
  shared expert by partition → 460 experts by 23× cloning with half the neurons redrawn, top-k
  2 → 12. Its failure: **"collapse along clone families"** under hard top-k (dead experts
  27 → 168/460). The fix: **probabilistic top-k selection** early on.
- **§16–19:** expert parallelism (dispatch/combine all-to-all, 45.1 GB per sequence), combining
  it with ZeRO-1, case studies, and the layout recipe.
- **§20:** V5 decisions. Bias balancing, dropless, EP inside a node, fp32 router.

**What the assignment asks for** (`S14-assignment.md`, due **Sat 2026-10-03 07:00**, 1000 pts):
train a **dense ("linear") model**, **convert it into an MoE**, and show that the MoE **continues
to train** and that its **loss keeps dropping**. Size and data are our choice. The repo **must
have training logs**. In class, "linear model" means the dense model, so this is §15 upcycling.
The instructor said to decide the MoE first and size the dense model from it, and that reusing
S13's ~20M model is fine ("or even a bigger one").

**How this differs from S13.** S13 compared arms of equal cost (baseline vs reversible) on
throughput and memory. **S14 is a continuity experiment.** The graded claim is a loss curve that
stays continuous across the dense→MoE switch and keeps falling. The dynamics around the switch are
the substance:
- The loss at the moment of conversion should match the dense model's, if the conversion
  preserves function.
- Expert load and dead-expert counts after the switch show whether the router collapses.
- A fair comparison is continuing the dense model for the same extra tokens. That is the
  sparse-upcycling paper's own baseline, and it is what makes "the MoE is worth it" a claim
  rather than an anecdote.

## Layout

- `S14-assignment.md`: the brief verbatim, the Axiom submission block, the empty rubric, and the
  instructor's framing from the class with timestamps.
- `resources/s14-session.md`: full lesson, all 21 sections. The page is ~51.6K chars.
  `get_page_text` cut the last ~400 (end of §20 and §21), which were read with `javascript_tool`.
  Tables are re-laid-out as markdown, KaTeX is written as plain math, and widgets are captions
  only. §1 contains a **stray authoring note left on the live page**, reproduced and marked.
- `resources/s14-transcript.md`: live-class transcript from **YouTube auto-captions**, because
  there is no Google Doc this time. Video `qcEbFT8g1mg` "ERA 5 Session 14 Studio", published
  2026-09-25, 2:18:56. It has 1,030 caption segments merged into 131 one-minute paragraphs,
  123KB, with no speaker labels and frequent mis-hearings ("Quen" = Qwen, "ciglo" = SwiGLU,
  "V4/B4" = Lightning LM V4). **How it was fetched:** this box's IP is bot-blocked by YouTube
  (yt-dlp and youtube-transcript-api both fail), and the signed timedtext URL returns empty
  without the player's PO token. So it was read from the watch page's own "Show transcript" panel
  in the user's browser (DOM `transcript-segment-view-model`) and brought over in three
  `get_page_text` passes of ~44K each. The assignment segment is at [2:11:34].
- `docs/s14-transcript-summary.md`: **not committed** (gitignored). A structured summary of the
  class with links to every paper, model and tool referred to. A personal study aid, per the
  standing convention.
- `tools/`: copied unchanged from S13 (`py2nb`, `run_nb`, `build_readme`, `dump_log`).
  `requirements.txt` is copied from S13.
- Not yet created: `notebook_src.py`, the `.ipynb`, `results.json`, `README.tmpl.md`,
  `README.md`, `logs/`, `assets/`.

## Conventions

- **Submission is a GitHub README link**, public, and the repo **must contain training logs**
  (the Axiom field says so explicitly). Commit `logs/nbexec.log` plus a plain per-step training
  log (step, phase dense/moe, loss, lr, tokens seen, per-layer expert-load stats) as a CSV or
  JSONL, so the curves can be checked without opening the notebook. Ship via subtree split to
  `github.com/vpw/era-v5-s14`. The user creates the repo first, because there is no `gh` or token
  on this box. The split, push and anonymous-check commands are in the
  `era-v5-toolchain-environment` memory.
- **Every number in the README comes from a cell that ran**, through `notebook_src.py → .ipynb →
  results.json → build_readme.py → README.md`. Use `tools/` unchanged.
- **Follow the lesson's own recipe unless there is a reason not to,** and say why when not:
  - aux-loss-free bias balancing, not an aux loss (the instructor was emphatic);
  - dropless routing;
  - an fp32 router;
  - load counted over the whole batch.
  Record per-expert load, MaxVio and dead-expert counts after conversion; they are the evidence
  that the MoE is really training rather than collapsing onto a few experts.
- **The conversion must be checked, not assumed.** Measure the loss at the step before and the
  step after conversion. If the construction is meant to preserve function (copy with renormalized
  weights, or partition that rebuilds the full FFN), assert that the outputs match at conversion.
  If it is not (drop-upcycling redraws half the neurons), report the size of the loss bump and how
  fast it recovers.
- **Widget-data extraction judged unnecessary.** The §15 widget visualizes clone-family collapse,
  which the prose states numerically (27 → 38 → 122 → 154 → 168 dead of 460). The deliverable's
  numbers come from our own runs, not from lesson widgets.
- **Compute: EC2 T4** (TODO D1, settled 2026-09-29). Design decisions D1–D6 are settled in TODO.md. S13's ~20M model ran on the EC2 T4 through
  `era-v5-gpu-run` (`SSH_KEY=~/.ssh/id_ed25519`). A few-million-parameter dense→MoE run is
  feasible on this 2-core CPU box, the way S11 did it, but slowly. The instructor pointed at
  laptop or Colab scale. The T4 has no bf16: use fp16 + GradScaler, and keep the router in fp32
  regardless.
- **Citations.** The README will cite the §15 papers:
  - Sparse Upcycling (arXiv 2212.05055);
  - Drop-Upcycling (arXiv 2502.19261);
  - Auxiliary-Loss-Free Load Balancing (arXiv 2408.15664);
  - probably Switch Transformer and DeepSeekMoE.
  **Verify every ID and date with the arXiv API** before writing it. Don't recall them. On this
  box the `arxiv-library` corpus and RAG layers do not exist and its search script 406s, so use
  `curl -A "Mozilla/5.0" 'https://export.arxiv.org/api/query?id_list=<id>'` (one ID per request;
  a long `id_list` gets rate-limited) and `pdftotext` for claims pulled from a PDF. The
  verification status of each reference is in `docs/s14-transcript-summary.md`. Lightning LM is
  The School of AI's own report; cite it as the lesson does.
- **Branch `s14-moe`**, cut from `s13-reversibility` 2026-09-29. Push to `origin` (SSH) when
  there is something to push.
- **Ties back:** S13's nanoGPT/TinyStories/BPE-8192 pipeline and `tools/` are the natural
  starting point for the dense phase, since the instructor explicitly allowed reusing the 20M
  model. S11's "tune both sides before accepting a comparison" applies to the continue-dense vs
  convert-to-MoE comparison.
