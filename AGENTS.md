# Task

This directory is part of the assignments for the ERA V5 course of The School of AI (TSAI).
Specifically this is for the fourteenth session (S14).

`S14-assignment.md` has the exercise in full, the Axiom submission block (due **Sat 2026-10-03
07:00**, 1000 pts, one GitHub-README link field whose note says "The Repo MUST have training
logs"), and the instructor's framing from class.

# Details

The session is **Mixture-of-Experts**. It covers:
- the MoE layer, where a router scores E experts, keeps the top-k, and sums their outputs weighted
  by the renormalized scores;
- router score functions;
- fine-grained and shared experts;
- load balancing: aux loss, and the now-standard auxiliary-loss-free per-expert bias;
- **growing an MoE from a dense model (upcycling, §15)**, which is what the assignment is about;
- expert parallelism.

The upcycling methods in §15:
- **copy**: every expert is the whole dense FFN;
- **partition**: each expert is a slice of the FFN;
- **drop**: each expert is a copy with a fraction r of its neurons redrawn, r = 0.5 best.

Identical clones under hard top-k collapse "along clone families". The fix is to redraw neurons
and use probabilistic top-k early on.

**What the assignment asks for:**

1. Train a dense model. The brief and the instructor call it a "linear model", meaning a
   standard transformer with one FFN per layer.
2. Convert it into an MoE. Size, data and method are our choice, and S13's ~20M model is
   explicitly allowed.
3. Show that the MoE continues to train and that its loss keeps dropping after the conversion.
4. Submit a GitHub README link. **The repo must contain the training logs.**

As a capable agent, plan to:
1. Read `resources/s14-session.md` §3, §7, §13 and §15, plus the transcript's segments at
   [1:35:02] (balancing) and [1:43:28]–[1:58:01] (growing), and the assignment at [2:11:34].
2. Settle the open decisions in `TODO.md`: compute lane, base model, expert count/width/top-k,
   upcycling method, router score function, token budget per phase.
3. Build the MoE FFN (router in fp32, dropless, aux-loss-free bias balancing, optional shared
   expert) and a `dense_to_moe()` conversion. Assert functional equivalence at conversion where
   the method preserves function.
4. Run dense → convert → continue as MoE, alongside a continue-dense control for the same extra
   tokens. Log per-step loss and per-expert load (MaxVio, dead experts) to a committed log file.
5. Build the README through the S13 `tools/` pipeline so every number comes from a cell that ran.
6. Cite papers only after verifying IDs and dates with the arXiv API (see CLAUDE.md, Conventions).
7. Ship via subtree split, verify it anonymously, and submit before the deadline.

`TODO.md` tracks progress on these steps.

## References
Refer CLAUDE.md if it exists
