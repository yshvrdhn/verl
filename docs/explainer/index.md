# verl × Ray: visual explainers

Last updated: 06/06/2026.

These are companion explainers to the {doc}`hybrid_flow` and {doc}`single_controller`
programming guides. Where those pages describe the design in prose, the pages below are
**self-contained, interactive HTML** that you can open directly in a browser — including a
live demo of how a single driver call is split across GPUs and gathered back.

The bundle is maintained at the repository root in `explainer_docs/` and vendored into the
docs site at build time, so the source of truth is a single, portable set of files.

## Start here

```{eval-rst}
.. raw:: html

   <ul>
     <li><a href="../_static/explainer/index.html" target="_blank"><b>Explainer landing page</b></a>
         — the full set, with one-line descriptions.</li>
     <li><a href="../_static/explainer/concept-explainer.html" target="_blank"><b>Concept explainer</b> (interactive)</a>
         — the mental model: why RL post-training is two tangled programs, and how one method
         call fans a batch out across every GPU and collects it back. Includes a live
         dispatch/collect demo.</li>
     <li><a href="../_static/explainer/tech-deep-dive.html" target="_blank"><b>Technical deep dive</b></a>
         — the call chain file by file: <code>@register</code> → <code>MAGIC_ATTR</code> →
         <code>_bind_worker_method</code> → dispatch/collect → placement groups → colocation →
         <code>fit()</code>, with real source snippets.</li>
     <li><a href="../_static/explainer/verl-ray-rl-training.md" target="_blank"><b>Comprehensive written breakdown</b> (Markdown)</a>
         — architecture in five nouns, the single-controller mechanism, resource pools and the
         hybrid engine, the roles, <code>DataProto</code>, the PPO/GRPO step traced through
         <code>fit()</code>, and a source map.</li>
     <li><a href="../_static/explainer/status.html" target="_blank">Documentation status</a>
         — what is covered and what is deliberately left for follow-up.</li>
   </ul>
```

## What they cover

The throughline across every page: **verl runs the RL control flow in one ordinary driver
process, while the heavy model computation runs across Ray actors — one per GPU — bridged by
the `verl.single_controller` dispatch layer.** That separation is the HybridFlow design, and
it is what lets verl:

- swap the computation backend (FSDP ⇄ Megatron ⇄ torchtitan) or change device placement
  **without touching the algorithm loop**;
- reserve GPUs with topology-aware **placement groups** (`STRICT_PACK`) so colocated training
  and inference share fast intra-node NCCL;
- multiplex several roles (actor, rollout, reference) onto **one Ray actor per GPU** via
  `create_colocated_worker_cls` (the hybrid engine); and
- overlap independent stages through **async `ObjectRef` futures**.

## Where the code lives

| Area | File |
| --- | --- |
| Driver entrypoint | `verl/trainer/main_ppo.py` |
| RL loop | `verl/trainer/ppo/ray_trainer.py` (`RayPPOTrainer`, `fit`, `init_workers`) |
| Register / dispatch | `verl/single_controller/base/decorator.py` |
| WorkerGroup binding | `verl/single_controller/base/worker_group.py` |
| Ray WorkerGroup / pools / placement / colocation | `verl/single_controller/ray/base.py` |
| Role workers | `verl/workers/engine_workers.py` |
| Batch container | `verl/protocol.py` |

```{note}
The interactive pages are static HTML with no build step or external dependencies. If you are
reading a locally built copy of the docs, the links above resolve to `_static/explainer/`. You
can also open the originals directly from `explainer_docs/` at the repository root.
```
