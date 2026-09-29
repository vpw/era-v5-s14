# Growing a dense GPT into a Mixture-of-Experts

ERA V5, Session 14. **Assignment:** train a dense ("linear") model, convert it into a
mixture-of-experts model, and show that the MoE keeps training and its loss keeps dropping.

Everything below was measured on one **{{env.gpu}}** (fp16 autocast, router in fp32) in a single
top-to-bottom run of [`S14.ipynb`](S14.ipynb). Every number in this README is filled in from
[`results.json`](results.json), which that run wrote. **Training logs**, one JSON line per step
and per evaluation for every run, are in [`logs/`](logs/). The full notebook output is in
[`logs/nbexec.log`](logs/nbexec.log).

## Result

<!--HEADLINE-->

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

<!--CONVERSION-->

## Expert load

![expert load](assets/expert_load.png)

![final load](assets/final_load.png)

<!--LOAD-->

## Findings

<!--FINDINGS-->

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
