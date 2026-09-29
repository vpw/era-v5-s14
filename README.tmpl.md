# Growing a dense GPT into a Mixture-of-Experts

ERA V5, Session 14. **Assignment:** train a dense ("linear") model, convert it into a
mixture-of-experts model, and show that the MoE keeps training and its loss keeps dropping.

Everything below was measured on one **{{env.gpu}}** (fp16 autocast, router in fp32) in a single
top-to-bottom run of [`S14.ipynb`](S14.ipynb). Every number in this README is filled in from
[`results.json`](results.json), which that run wrote. **Training logs**, one JSON line per step
and per evaluation for every run, are in [`logs/`](logs/). The full notebook output is in
[`logs/nbexec.log`](logs/nbexec.log).

## Result

**Yes, the converted model keeps training.**
- Right after conversion the MoE's validation loss is {{summary.moe.val_start:.4f}}. The
  conversion itself costs {{conversion.jump_r05:+.3f}}; see below for why.
- It is back below the dense checkpoint's {{conversion.dense:.4f}} within
  {{summary.moe.recovery_tokens_M:.1f}}M tokens.
- It ends at **{{summary.moe.val_end:.4f}}**: {{summary.moe.drop_in_phase_b:.3f}} below where it
  started, and {{summary.moe.vs_dense_ckpt:+.3f}} relative to the dense checkpoint.
- It has **{{summary.moe.final_val_dead}} dead experts** on the validation set.

**It does not beat simply training the dense model for the same 50M tokens.**
- The control ends at {{summary.dense_cont.val_end:.4f}}.
- The drop-upcycled MoE finishes {{summary.moe.vs_control:+.4f}} behind it. The hard-top-k
  variant finishes level, at {{summary.moe_hard.vs_control:+.4f}}.
- The MoE learned faster per token once it had recovered: the gap closed steadily, from
  {{conversion.jump_r05:+.3f}} at conversion to {{summary.moe.vs_control:+.4f}} at the end. But
  it started from a handicap that 50M tokens did not quite erase.
- On this implementation it also costs **{{runs.moe.tokens_per_s:,.0f}} vs {{runs.dense_cont.tokens_per_s:,.0f}} tokens/s**
  of wall-clock throughput.

![loss curves](assets/loss_curves.png)

| run (phase B, +{{data.phase_tokens:,}} tokens each) | val loss at start | val loss at end | vs dense checkpoint ({{conversion.dense:.4f}}) | vs control | tokens/s | peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **MoE, drop-upcycled** | {{summary.moe.val_start:.4f}} | **{{summary.moe.val_end:.4f}}** | {{summary.moe.vs_dense_ckpt:+.4f}} | **{{summary.moe.vs_control:+.4f}}** | {{runs.moe.tokens_per_s:,.0f}} | {{runs.moe.peak_alloc_gib:.2f}} GiB |
| dense, continued (control) | {{summary.dense_cont.val_start:.4f}} | {{summary.dense_cont.val_end:.4f}} | {{summary.dense_cont.vs_dense_ckpt:+.4f}} | — | {{runs.dense_cont.tokens_per_s:,.0f}} | {{runs.dense_cont.peak_alloc_gib:.2f}} GiB |
| MoE, hard top-k from start (ablation) | {{summary.moe_hard.val_start:.4f}} | {{summary.moe_hard.val_end:.4f}} | {{summary.moe_hard.vs_dense_ckpt:+.4f}} | {{summary.moe_hard.vs_control:+.4f}} | {{runs.moe_hard.tokens_per_s:,.0f}} | {{runs.moe_hard.peak_alloc_gib:.2f}} GiB |

Phase A, the dense model: validation loss {{runs.dense.val_start:.3f}} → **{{runs.dense.final_val_loss:.4f}}**
over {{data.phase_tokens:,}} tokens, at {{runs.dense.tokens_per_s:,.0f}} tokens/s.

## What was built

**Data.** TinyStories V2 (GPT-4 split), the 8,192-token BPE from Session 13, and 512-token
windows in one fixed shuffled order. Phase A reads the first {{data.phase_tokens:,}} tokens
of that order. Every phase-B run reads the *next* {{data.phase_tokens:,}}, so the MoE, the
control and the ablation see identical batches. Validation uses {{data.eval_tokens:,}}
held-out tokens.

**The dense model** ({{model.dense_params_M:.2f}}M parameters): a GPT with width {{model.d}},
{{model.L}} layers, {{model.heads}} heads and context {{model.T}}. It uses tied embeddings and a
GELU feed-forward block of hidden width {{model.ffn_hidden}}.

**The MoE layer** replaces each feed-forward block. It follows the lesson's layer (§3), router
(§7) and balancing (§13–14):

```
x ─┬─► shared expert (width {{model.shared_width}}) ─────────────────────────────────────┐
   │                                                                                      ├─► sum + b ─► y
   └─► router (fp32 softmax over {{model.E}}) ─► top-{{model.k}} of (probs + bias) ─► {{model.k}} × expert (width {{model.expert_width}}) × g_i ─┘
                         g_i = renormalised probs × {{model.k}} (routed scaling factor)
```

- **Width per token matches dense.** A token passes through {{model.shared_width}} + {{model.k}} × {{model.expert_width}} = {{model.ffn_hidden}}
  hidden units, the same as the dense block (lesson §4).
  - Active feed-forward parameters per layer: {{model.ffn_moe_active_per_layer:,}}. This is
    the dense {{model.ffn_dense_per_layer:,}} plus the router.
  - Stored feed-forward parameters per layer: {{model.ffn_moe_total_per_layer:,}}, which is
    **{{model.ffn_ratio:.2f}}×** the dense block.
  - Whole model: **{{model.moe_params_M:.2f}}M total, {{model.moe_active_M:.2f}}M active per token.**
- **Router:** float32 (§7), initialised at one tenth of the usual scale.
- **Balancing:** auxiliary-loss-free, with a per-expert bias used only to choose and updated
  after each step by γ·sign(mean load − load_i), γ = {{model.gamma}}, counted over the whole
  batch (§13–14). There is no auxiliary loss.
- **No token dropping** (§11).
- **Probabilistic top-k** (Gumbel sampling in proportion to probs + bias) for the first
  {{model.prob_steps}} steps after conversion, then hard top-k (§15).

**The conversion** (`dense_to_moe`). A neuron of the dense block (a row of W1, an entry of b1, a
column of W2) is a one-unit MLP, and the block is the sum of its neurons. So:
- The **shared expert** is the first {{model.shared_width}} neurons, unchanged.
- **Each routed expert** takes {{model.expert_width}} neurons sampled at random from the other
  {{model.shared_width}}, independently per expert. With {{model.k}} experts per token and the
  scaling factor {{model.k}}, each of those neurons is used once per token in expectation, so
  the routed sum starts as an unbiased estimate of the dense half it replaces. This is the
  partition step of the lesson's Lightning LM recipe.
- **Drop-upcycling** (Nakamura et al., [arXiv 2502.19261](https://arxiv.org/abs/2502.19261)):
  - In each routed expert, r = {{model.r_drop}} of its neurons are re-drawn.
  - Their W1 rows, b1 entries and W2 columns come from normal distributions with the mean and
    standard deviation of the original values at those neurons (the paper's §3.2 eq. 4, and
    its appendix C.6.1 for fine-grained experts).
  - This breaks the symmetry between experts that start from the same weights.

## Checked before anything trained

These are assertions in the notebook, and a failure stops the run:

| gate | what must hold | measured |
| --- | --- | --- |
| 1. copy upcycling | every expert a full copy of the dense block, top-2, weights summing to 1: output equals dense, whatever the router picks | max \|Δ\| = {{gates.copy_max_err:.1e}} (float64) |
| 2. shared + partition | shared half + the other half cut into disjoint experts, all selected, scale k: output equals dense | max \|Δ\| = {{gates.partition_max_err:.1e}} (float64) |
| 3. balancing | the sign rule on a router skewed ~20× toward one expert drives MaxVio down | {{gates.maxvio_start:.2f}} → {{gates.maxvio_end:.2f}} |
| 4. parameter count | active FFN = dense FFN + router exactly; total ≈ 2.5× | {{model.ffn_moe_active_per_layer:,}} = {{model.ffn_dense_per_layer:,}} + {{model.d}}×{{model.E}}; {{model.ffn_ratio:.2f}}× |

## The conversion, measured on the trained checkpoint

| version of the phase-A checkpoint | val loss | Δ vs dense |
| --- | ---: | ---: |
| dense | {{conversion.dense:.4f}} | — |
| copy upcycling (2 full copies, top-2): must match | {{conversion.copy:.4f}} | {{conversion.copy_delta:+.4f}} |
| shared + 16 sampled experts, r = 0 | {{conversion.sample_r0:.4f}} | {{conversion.jump_r0:+.4f}} |
| **shared + 16 sampled experts, r = 0.5 (phase B starts here)** | **{{conversion.sample_r05:.4f}}** | **{{conversion.jump_r05:+.4f}}** |

- **The copy check confirms the conversion code on the real weights.** Two full copies with
  weights summing to one give the dense loss to {{conversion.copy_delta:.1e}}, which is
  float16 rounding.
- **Routing alone costs {{conversion.jump_r0:+.3f}}.** With nothing re-drawn, each token sees
  the shared half plus 4 of the 16 random slices of the other half. That is an unbiased but
  noisy estimate of the dense block.
- **Re-drawing half of every routed expert raises the jump from {{conversion.jump_r0:+.3f}} to
  {{conversion.jump_r05:+.3f}}.** This is the price drop-upcycling
  pays on purpose: the re-drawn neurons are what make experts built from the same dense weights
  start out different. The shared expert is untouched, which is why the model is still far
  better than random (a fresh model starts at {{runs.dense.val_start:.2f}}).

## Expert load

![expert load](assets/expert_load.png)

![final load](assets/final_load.png)

- **Balancing works.** At the end of both MoE runs the load on the validation set is close to
  uniform in every layer. Mean MaxVio over layers is {{summary.moe.final_val_maxvio:.3f}}
  (drop-upcycled) and {{summary.moe_hard.final_val_maxvio:.3f}} (hard top-k), meaning the
  busiest expert carries about 5% more than its fair share.
- **No expert died.** No expert received zero validation tokens in either run, at any
  evaluation.
  - Per training step, the drop-upcycled run never had an expert with no token (maximum
    {{dead_per_step.moe_max}} of 128).
  - The hard-top-k run briefly had at most {{dead_per_step.moe_hard_max}} of 128 during its
    first few hundred steps, and none over its last 100.
- **The probabilistic window changes the picture, and not for the better.** While experts are
  *sampled* in proportion to the router's probabilities, load is spread almost evenly by
  construction, so MaxVio sits near zero (left panel). The balancing bias therefore has nothing
  to correct and stays near zero too. When hard top-k switches on at step {{model.prob_steps}},
  the router's actual preferences appear all at once. That is the largest imbalance of either
  run, and the bias then needs a few hundred steps to catch up. The run that used hard top-k
  from the first step had moderate imbalance early and balanced steadily.

## Findings

1. **The assignment's two requirements are met.**
   - The MoE **continues to train**: no divergence, no dead experts, loss falling at every
     evaluation after the first.
   - Its **loss drops**: {{summary.moe.val_start:.4f}} → {{summary.moe.val_end:.4f}}, ending
     {{summary.moe.vs_dense_ckpt:+.3f}} below the dense model it was grown from.

2. **Against a fair control, it ties rather than wins at this budget.** Validation loss at
   matched tokens after conversion:

   | tokens after conversion | MoE (drop-upcycled) | MoE (hard top-k) | dense continued |
   | ---: | ---: | ---: | ---: |
   | 0 | {{curves.moe.val_loss.0:.4f}} | {{curves.moe_hard.val_loss.0:.4f}} | {{curves.dense_cont.val_loss.0:.4f}} |
   | {{curves.moe.tokens.3:,}} | {{curves.moe.val_loss.3:.4f}} | {{curves.moe_hard.val_loss.3:.4f}} | {{curves.dense_cont.val_loss.3:.4f}} |
   | {{curves.moe.tokens.8:,}} | {{curves.moe.val_loss.8:.4f}} | {{curves.moe_hard.val_loss.8:.4f}} | {{curves.dense_cont.val_loss.8:.4f}} |
   | {{curves.moe.tokens.13:,}} | {{curves.moe.val_loss.13:.4f}} | {{curves.moe_hard.val_loss.13:.4f}} | {{curves.dense_cont.val_loss.13:.4f}} |
   | {{curves.moe.tokens.16:,}} | {{curves.moe.val_loss.16:.4f}} | {{curves.moe_hard.val_loss.16:.4f}} | {{curves.dense_cont.val_loss.16:.4f}} |
   | {{curves.moe.tokens.18:,}} | **{{curves.moe.val_loss.18:.4f}}** | **{{curves.moe_hard.val_loss.18:.4f}}** | **{{curves.dense_cont.val_loss.18:.4f}}** |

   - The MoE gains ground at every evaluation: it has 2.5× the feed-forward capacity at the same
     active size.
   - But the closing slowed during the final learning-rate decay. **This run does not show
     whether the MoE would pull ahead with more tokens**, and this README does not claim it.
   - The lesson's sparse-upcycling result (the MoE beats continuing the dense model within
     10–60% extra budget) is for full-copy upcycling at far larger scale. It is not reproduced
     here. The gap it would have to overcome is the conversion handicap in finding 3.

3. **Most of the handicap is the re-draw.** Routing alone costs {{conversion.jump_r0:+.3f}}; the
   drop-upcycling re-draw at r = {{model.r_drop}} brings it to {{conversion.jump_r05:+.3f}}. The
   paper chose r = 0.5 for long training runs, where expert diversity has time to pay off. At
   50M tokens on a 17M model, keeping more of the dense model's knowledge is plausibly worth
   more. The follow-up below tests exactly this with r = 0.

4. **Probabilistic top-k did not help, as predicted.** If anything it hurt, but the final difference is within about twice the seed noise (finding 6).
   - The prediction (§11 of the notebook, written before the run) was a *small* effect. The
     failure it guards against, where near-identical clones under hard top-k collapse, needs
     clones, and these experts are 16 different random draws with half their neurons re-drawn.
   - Measured: hard top-k from the start ended at {{summary.moe_hard.val_end:.4f}}, against
     {{summary.moe.val_end:.4f}} with the sampling window, with no collapse (at most {{dead_per_step.moe_hard_max}} empty expert of 128 on any
     step).
   - The cost of sampling showed up as the imbalance spike when it switched off (see
     *Expert load*). Sampling routes tokens to experts the hard router would not choose, so
     those experts spend 200 steps training on assignments they will not get afterwards.

5. **Sparse compute is not free wall-clock time here.**
   - Throughput: {{runs.moe.tokens_per_s:,.0f}} tokens/s for the MoE against
     {{runs.dense_cont.tokens_per_s:,.0f}} for dense, at the same active parameters
     ({{model.moe_active_M:.2f}}M vs {{model.dense_params_M:.2f}}M).
   - Peak memory: {{runs.moe.peak_alloc_gib:.2f}} vs {{runs.dense_cont.peak_alloc_gib:.2f}} GiB.
   - The expert computation is a Python loop over 16 experts with gather and scatter. Production
     MoE code uses grouped or block-sparse matrix multiplication, as MegaBlocks does, and the
     lesson's §16 is about exactly this. The "same compute" claim holds in FLOPs, not on this
     implementation's clock.

6. **Caveats.**
   - This is one seed. Session 13 measured seed-to-seed differences of 0.004–0.006 in final
     validation loss on a similar TinyStories setup, so the final differences between the three
     phase-B runs (all under 0.01) are **not resolved**. The earlier, larger gaps are.
   - Both phase-B arms restart with a fresh optimiser and a short re-warmup, which is why the
     control's loss also rises briefly after the switch. The comparison is between two runs
     that had the same restart.

## Follow-up: the same MoE with nothing re-drawn (r = 0)

[`S14_r0.ipynb`](S14_r0.ipynb) ({{r0.runtime_min:.0f}} min) tested finding 3 by changing only r,
from {{model.r_drop}} to 0. It runs the main notebook's cells verbatim and uses hard top-k, so
it pairs with the main run's hard-top-k arm.
- It retrained the dense phase with the same seed. That reproduced the main run's dense loss
  to {{r0.summary.dense_rerun_delta_vs_main:+.4f}}.
- It reran the control from that checkpoint. The control reproduced to
  {{r0.summary.control_rerun_delta_vs_main:+.4f}}, which measures run-to-run noise at a
  fixed seed.

The prediction, written in that notebook before the run: a smaller jump, a faster recovery,
and a lower final loss than r = 0.5. **All three held.**

| | r = 0 (follow-up) | r = 0.5, hard top-k (main) | dense continued (same checkpoint) |
| --- | ---: | ---: | ---: |
| conversion jump | {{r0.summary.conversion.jump_r0:+.4f}} | {{conversion.jump_r05:+.4f}} | — |
| tokens to get back below the dense checkpoint | {{r0.summary.moe_r0_recovery_tokens_M:.1f}}M | {{summary.moe_hard.recovery_tokens_M:.1f}}M | — |
| final val loss | **{{r0.summary.moe_r0_val_end:.4f}}** | {{summary.moe_hard.val_end:.4f}} | {{r0.summary.control_val_end:.4f}} |
| vs the control | **{{r0.summary.moe_r0_vs_control:+.4f}}** | {{summary.moe_hard.vs_control:+.4f}} | — |

![r=0 vs r=0.5](assets/r0_vs_r05.png)

- Without the re-draw handicap, the MoE **overtakes the dense control**. It was behind at 40M
  tokens after conversion ({{r0.summary.curve.moe_r0.16:.4f}} vs {{r0.summary.curve.control_rerun.16:.4f}})
  and ahead at 45M and 50M.
- The lead is small: {{r0.summary.moe_r0_vs_control:+.4f}}. That is well above the same-seed
  rerun noise measured here, but at the edge of the 0.004–0.006 seed-to-seed spread Session 13
  measured. So read it as "no longer behind, probably slightly ahead", not as a resolved win.
- At this scale and budget, keeping the dense model's knowledge (r = 0) beat drop-upcycling's
  diversity (r = 0.5). This is one setting and one seed, not a refutation of the paper, which
  targets much longer training.
- No dead experts: {{r0.summary.moe_r0_final_val_dead}} on the validation set, and at most
  {{r0.summary.moe_r0_dead_per_step_max}} on any step. Final mean MaxVio was
  {{r0.summary.moe_r0_final_val_maxvio:.3f}}.

## Cost

- Total notebook runtime: {{meta.total_runtime_min:.1f}} min, about **${{meta.total_cost_usd:.2f}}**
  (g4dn.2xlarge, ${{meta.hourly_usd}}/h on-demand, ap-south-1).
- Tokens/s is steady-state: every step after the first 20, evaluation excluded, with a GPU sync
  before each clock read.
- Peak memory is `torch.cuda.max_memory_allocated()` over each run.

## Reproduce

```bash
pip install -r requirements.txt
python tools/py2nb.py notebook_src.py S14.ipynb     # the notebook is generated from notebook_src.py
python tools/run_nb.py S14.ipynb                     # executes top to bottom; any failing gate stops it
python tools/build_readme.py                         # README.md from README.tmpl.md + results.json
S14_SMOKE=1 python tools/run_nb.py S14.ipynb         # a 2-minute CPU dry run of the whole pipeline
```

| file | what it is |
| --- | --- |
| `notebook_src.py` | source of truth for the notebook |
| `S14.ipynb` | the executed notebook, with outputs from the T4 run |
| `notebook_r0_src.py` → `S14_r0.ipynb` | the r = 0 follow-up (executed on the T4), results under `r0` in `results.json` |
| `results.json` | every measured number, written by the notebook's last cell |
| `logs/train_<run>.jsonl` | per-step training log: loss, LR, grad norm, and for MoE runs per-layer MaxVio and dead-expert count; plus per-evaluation validation loss and full per-layer expert load |
| `logs/progress.txt` | the evaluation lines as they were printed during the run |
| `logs/nbexec.log` | every cell's printed output |
| `assets/` | plots, and the tokenizer |

## References

Lesson: *ERA V5 Session 14, Mixture-of-Experts* (The School of AI), §3, §7, §11, §13–15.
arXiv IDs were checked against arxiv.org.

- Nakamura et al., *Drop-Upcycling: Training Sparse Mixture of Experts with Partial
  Re-initialization*, [arXiv 2502.19261](https://arxiv.org/abs/2502.19261) (2025).
- Komatsuzaki et al., *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*,
  [arXiv 2212.05055](https://arxiv.org/abs/2212.05055) (2022).
- Wang et al., *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*,
  [arXiv 2408.15664](https://arxiv.org/abs/2408.15664) (2024).
- Qiu et al., *Demons in the Detail: On Implementing Load Balancing Loss for Training
  Specialized Mixture-of-Expert Models*, [arXiv 2501.11873](https://arxiv.org/abs/2501.11873)
  (2025), on whole-batch balancing.
- Fedus et al., *Switch Transformers*, [arXiv 2101.03961](https://arxiv.org/abs/2101.03961)
  (2021), on the fp32 router and small router init.
- Dai et al., *DeepSeekMoE*, [arXiv 2401.06066](https://arxiv.org/abs/2401.06066) (2024), on
  fine-grained and shared experts.
- Gale et al., *MegaBlocks*, [arXiv 2211.15841](https://arxiv.org/abs/2211.15841) (2022), on
  dropless MoE.
- Eldan & Li, *TinyStories*, [arXiv 2305.07759](https://arxiv.org/abs/2305.07759) (2023).
