# verl × Ray — How a single-controller RL framework drives a GPU cluster

> A comprehensive breakdown of **verl** and how it uses **Ray** to run reinforcement
> learning (RLHF / PPO / GRPO) on large language models *effectively*.
>
> Companion HTML docs in this folder:
> - [`concept-explainer.html`](concept-explainer.html) — the mental model (start here)
> - [`tech-deep-dive.html`](tech-deep-dive.html) — the call chain, file by file
> - [`status.html`](status.html) — what was documented and where
>
> Sources are file paths + line numbers in this repository, plus verl's own design
> notes in [`docs/hybrid_flow.rst`](../docs/hybrid_flow.rst) and
> [`docs/single_controller.rst`](../docs/single_controller.rst).

---

## TL;DR

verl is an implementation of the **HybridFlow** paper. Its core idea: an RL training
loop (PPO/GRPO) is a **two-level dataflow** problem.

- **Control flow** — the algorithm: *generate → score → compute advantages → update*.
  This is small, sequential, and changes often between algorithms.
- **Computation flow** — the heavy neural-network work: rollout generation, forward
  passes, backprop, optimizer steps. This is multi-GPU, multi-process, and rarely changes.

Most frameworks fuse the two into one SPMD program (every GPU runs the same script).
verl **decouples** them: the control flow runs in **one ordinary Python process — the
single controller / driver** — while the computation flow runs across a fleet of **Ray
actors**, one per GPU. Ray is what makes that split practical, because Ray lets you
expose any Python class method as a remote RPC.

The result reads like a single-process script:

```python
for prompt in dataloader:
    output      = actor_rollout_ref_wg.generate_sequences(prompt)
    old_log_prob = actor_rollout_ref_wg.compute_log_prob(output)
    ref_log_prob = ref_policy_wg.compute_ref_log_prob(output)
    values       = critic_wg.compute_values(output)
    rewards      = reward_wg.compute_rm_score(output)
    advantages   = compute_advantages(values, rewards)   # runs on the driver
    actor_rollout_ref_wg.update_actor(output)
    critic_wg.update_critic(output)
```

…but every `*_wg.method(...)` call fans a sharded `DataProto` out to dozens of GPUs,
runs the work in parallel, and gathers the results back — all hidden behind a decorator.

---

## 1. Why Ray, and why a single controller

From [`docs/single_controller.rst`](../docs/single_controller.rst): the module was born
from a request to turn a *toy single-process RLHF script* into a *distributed system with
minimal changes, while keeping it debuggable*.

The usual approach — PyTorch DDP — wraps an `nn.Module` and launches N identical
processes under different ranks. That has two problems for RLHF:

1. **PPO is multiple DAGs**, not one. Actor, critic, reference, reward, and rollout are
   different models with different parallelism, scheduled at different points.
2. **You can't easily inspect intermediate tensors** when every rank is racing through
   the same script.

verl instead breaks the loop into well-defined stages (`generate_sequences`,
`compute_log_prob`, `compute_advantages`, …) coordinated from one place. Ray was chosen
as the backend specifically because **it can expose a Python class method as an RPC
endpoint**. The catch Ray *doesn't* solve on its own: "one method call = one RPC", whereas
training an LLM needs **one logical call to hit many processes at once**. verl's
`single_controller` module is the layer that hides that fan-out.

---

## 2. The architecture in five nouns

| Concept | Where | What it is |
|---|---|---|
| **Driver / single controller** | `verl/trainer/main_ppo.py` → `RayPPOTrainer.fit()` | One process. Runs the RL algorithm. Holds no model weights. |
| **Worker** | `verl/single_controller/base/worker.py`, `verl/workers/engine_workers.py` | A `@ray.remote` actor pinned to one GPU. Holds model shards, does the math. |
| **WorkerGroup** | `verl/single_controller/base/worker_group.py`, `.../ray/base.py:RayWorkerGroup` | A driver-side proxy for a *list* of workers. One call → N RPCs. |
| **ResourcePool** | `.../ray/base.py:RayResourcePool` | A set of GPUs (a Ray *placement group*) that a WorkerGroup lives on. |
| **DataProto** | `verl/protocol.py` | The batch container that travels between driver and workers, and that knows how to `chunk`/`concat` itself. |

```
                         ┌──────────────────────────────────────────┐
   one Python process →  │            RayPPOTrainer.fit()            │   ← control flow
                         │   generate → logprob → reward → adv → upd │
                         └───────┬───────────┬───────────┬──────────┘
              WorkerGroup proxies │           │           │
        (dispatch + collect)      ▼           ▼           ▼
                         ┌─────────────┐ ┌──────────┐ ┌──────────┐
                         │ ActorRollout│ │  Critic  │ │  Reward  │   ← computation flow
                         │   Ref WG    │ │    WG    │ │    WG    │     (Ray actors, 1/GPU)
                         └─────────────┘ └──────────┘ └──────────┘
                          GPU0 GPU1 …      GPU0 …        GPU0 …
                          └── placement group / ResourcePool ──┘
```

---

## 3. The single-controller mechanism (the clever part)

This is the machinery that makes `actor_rollout_ref_wg.generate_sequences(data)` — a
plain method call on the driver — actually run on every GPU and come back merged.

### 3.1 `@register` tags a method with a dispatch policy

A worker method is decorated to declare *how its input should be split across workers* and
*how outputs should be merged back*:

```python
# verl/workers/engine_workers.py
class ActorRolloutRefWorker(Worker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto): ...
```

`register` (`verl/single_controller/base/decorator.py`) doesn't change behavior at
definition time — it just attaches metadata under a magic attribute so the WorkerGroup can
find it later:

```python
# verl/single_controller/base/decorator.py
MAGIC_ATTR = "attrs_3141562937"

def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, ...):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(inner, MAGIC_ATTR, attrs)
        return inner
    return decorator
```

### 3.2 Dispatch modes — the split/collect strategies

`Dispatch` is a dynamic enum (`decorator.py:26`). The modes that matter:

- **`ONE_TO_ALL`** — broadcast the *same* args to every worker. Used for
  lifecycle/setup calls like `init_model`, `save_checkpoint`. `dispatch_one_to_all`
  literally replicates each arg `world_size` times.
- **`DP_COMPUTE_PROTO`** — the data-parallel workhorse. `dispatch_dp_compute_data_proto`
  calls `DataProto.chunk(world_size)` to split a big batch into N shards (with auto-padding),
  sends shard *i* to worker *i*, then `collect_dp_compute_data_proto` concatenates the N
  results back into one `DataProto`.
- **`ALL_TO_ALL`** — pass args through untouched (caller handles layout).
- **`make_nd_compute_dataproto_dispatch_fn(mesh_name=...)`** — the modern,
  device-mesh-aware variant used by the unified engine workers, so a call is dispatched
  along the correct axis of an N-D parallelism mesh (e.g. only to data-parallel rank
  leaders of the `"actor"` / `"ref"` / `"train"` mesh).

The registry binds each mode to its pair of functions
(`DISPATCH_MODE_FN_REGISTRY` in `decorator.py`):

```python
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL:       {"dispatch_fn": dispatch_one_to_all,            "collect_fn": collect_all_to_all},
    Dispatch.DP_COMPUTE_PROTO: {"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_dp_compute_data_proto},
    ...
}
```

### 3.3 Binding: turning metadata into a real WorkerGroup method

When a `RayWorkerGroup` is constructed it (1) spawns the Ray actors, then (2) walks every
method of the worker class and, for each one carrying `MAGIC_ATTR`, **generates a new
method on the WorkerGroup itself** that wires dispatch → execute → collect together
(`worker_group.py:_bind_worker_method`, `base/decorator.py:func_generator`):

```python
# conceptually, what the generated WorkerGroup method does:
def generated(self, *args, **kwargs):
    args, kwargs = dispatch_fn(self, *args, **kwargs)     # split batch into N shards
    refs = execute_fn(method_name, *args, **kwargs)        # N async Ray RPCs → ObjectRefs
    out  = collect_fn(self, ray.get(refs)) if blocking else refs  # gather + concat
    return out
```

`execute_fn` is selected by `execute_mode`: `execute_all` fires the method on **all**
workers (`ray.get` returns a list of futures), `execute_rank_zero` only on rank 0.

The payoff (from the design doc): the distributed call is **textually identical** to the
single-process one. `rollout.generate_sequences(batch)` works whether `rollout` is a local
object or a 64-GPU `RayWorkerGroup`.

---

## 4. How Ray resources are actually carved up

### 4.1 ResourcePool → Ray placement group

A `RayResourcePool` turns a request like "2 nodes × 8 GPUs" into Ray **placement groups**
with the **`STRICT_PACK`** strategy, so all the GPUs of a group land on the *same node*
(critical for fast intra-node NCCL weight transfer):

```python
# verl/single_controller/ray/base.py : RayResourcePool.get_placement_groups
bundle = {"CPU": self.max_colocate_count}
if self.use_gpu:
    bundle[device_name] = 1                       # one GPU per bundle
pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]
pgs = [placement_group(bundles=bundles, strategy=strategy, ...)  # strategy="STRICT_PACK"
       for idx, bundles in enumerate(pg_scheme)]
ray.get([pg.ready() for pg in pgs])               # block until the cluster reserves them
```

Each **bundle = one worker = one GPU**. Each Ray actor is then scheduled onto a specific
bundle via `PlacementGroupSchedulingStrategy(placement_group, placement_group_bundle_idx=…)`
(`ray/base.py:RayClassWithInitArgs.__call__`), pinning rank *i* to bundle *i*.

### 4.2 Colocation / the hybrid engine — many roles, one GPU set

Naively, actor + critic + reference + reward + rollout would each want their own GPUs. verl
**colocates** compatible roles onto the *same* placement group with
`create_colocated_worker_cls` (`ray/base.py:988`). It dynamically builds a `WorkerDict`
class that inherits from each role's worker and multiplexes their registered methods, so
**one Ray actor per GPU exposes the methods of several roles**:

```python
# verl/trainer/ppo/ray_trainer.py : init_workers()
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict   = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    spawn_wg  = wg_dict.spawn(prefix_set=class_dict.keys())   # one WG view per role
    all_wg.update(spawn_wg)
```

Why colocate:

- **Actor + Rollout** share GPUs so updated policy weights move to the inference engine
  over **NCCL / IPC** without leaving the device — no checkpoint round-trip per step.
- **Actor + Reference** colocation enables efficient LoRA PPO (the reference policy *is*
  the base model; only adapters differ).

This is the **hybrid engine**: training and inference live on the same GPUs and take turns.
The vLLM rollout engine runs in **sleep mode** during the training phase to free KV-cache
memory, then `wake_up()`s and receives fresh weights before the next generation phase
(`verl/workers/rollout/vllm_rollout/vllm_async_server.py:sleep/wake_up`,
`checkpoint_manager.update_weights(...)` / `sleep_replicas()` in `fit()`).

`max_colocate_count` controls how many WorkerGroups (processes) share a pool — FSDP uses a
small count (actor/critic/ref fused), Megatron can use more.

---

## 5. The roles and their workers

`Role` (an enum in `ray_trainer.py`) maps each logical part of PPO to a worker class via
`role_worker_mapping`:

| Role | Worker class | Responsibility |
|---|---|---|
| `ActorRollout` / `ActorRolloutRef` | `ActorRolloutRefWorker` (`engine_workers.py:434`) | policy weights + vLLM/SGLang rollout (+ optional reference) |
| `Critic` | `TrainingWorker` w/ `model_type="value_model"` (`engine_workers.py:76`) | value function |
| `RefPolicy` | `ActorRolloutRefWorker` in ref mode | frozen reference log-probs for KL |
| `RewardModel` | reward worker / `reward_manager` | scores responses (model- or rule-based) |

Representative registered methods on `ActorRolloutRefWorker`:

- `init_model` — `Dispatch.ONE_TO_ALL`
- `generate_sequences` — `DP_COMPUTE_PROTO` (data-parallel rollout)
- `compute_log_prob` / `compute_ref_log_prob` — mesh-aware dispatch on the `actor`/`ref` mesh
- `update_actor` — mesh-aware dispatch on the `train` mesh, `blocking=False`
- `save_checkpoint` — `ONE_TO_ALL`

A **single-controller, hybrid-engine** RewardManager can also combine **model-based** and
**rule-based** rewards (e.g. `gsm8k.py`, `math.py` verifiers in `verl/utils/reward_score/`).

---

## 6. DataProto — the thing that flows on the edges

`DataProto` (`verl/protocol.py`) is the batch container that crosses the driver↔worker
boundary. It carries:

- `batch` — a `TensorDict` of padded tensors (`input_ids`, `attention_mask`,
  `response_mask`, `old_log_probs`, `advantages`, …),
- `non_tensor_batch` — numpy/object columns (uids, multi-modal inputs, raw prompts),
- `meta_info` — scalars/flags (`global_token_num`, sampling config, timing).

Crucially it implements **`chunk(n)`** and **`concat([...])`** — exactly what
`DP_COMPUTE_PROTO`'s dispatch/collect functions use to shard a batch across DP ranks and
reassemble the answers. `DataProtoFuture` lets one stage's *unmaterialized* output feed the
next stage's dispatch without a driver round-trip — futures are only `ray.get`-ed when a
blocking collect needs them (`_materialize_futures`). Each `union()` in the loop merges a
new column (log-probs, values, rewards, advantages) into the running batch.

---

## 7. The PPO/GRPO step, traced through `fit()`

`RayPPOTrainer.fit()` (`verl/trainer/ppo/ray_trainer.py:1362`) is the whole RL algorithm in
one readable loop. Per training step:

1. **Sample prompts** from the dataloader; build `gen_batch`; `repeat()` it `rollout.n`
   times for group sampling (GRPO).
2. **Generate** (`gen`): `async_rollout_manager.generate_sequences(combined_gen_batch)` →
   responses come back; rollout replicas then `sleep` to release memory.
   *(REMAX additionally folds a greedy baseline rollout into the same request.)*
3. **Balance** valid tokens across DP ranks (`_balance_batch`) so no GPU is starved.
4. **Reward** (`reward`): `_compute_reward_colocate(batch)` →
   `reward_wg.compute_rm_score(...)` and/or rule-based verifiers; `extract_reward`.
5. **Old log-probs** (`old_log_prob`): `actor_rollout_wg.compute_log_prob(batch)` —
   the proximal anchor π_old (or, in *bypass mode*, reuse rollout log-probs directly).
6. **Reference log-probs** (if KL): `ref_policy_wg.compute_ref_log_prob(batch)`.
7. **Values** (if a critic): `critic_wg.compute_values(batch)`.
8. **Advantages** (`adv`): optional KL penalty, then `compute_advantage(...)` — GAE for PPO,
   group-normalized returns for GRPO. **This runs on the driver**, on already-gathered data.
9. **Update critic** (`update_critic`): `critic_wg.update_critic(batch)`.
10. **Update actor** (`update_actor`): `actor_rollout_wg.update_actor(batch)` (after any
    `critic_warmup`). New weights are pushed to the rollout engine via the hybrid-engine
    weight sync before the next step.
11. **Checkpoint / validate / log** on the configured cadence.

Steps 2 and 5–10 are each *one* WorkerGroup call on the driver that fans out to all GPUs.
Step 8 is plain Python on the controller — which is exactly why implementing a new
advantage estimator or algorithm variant needs *no* changes to the distributed plumbing.

---

## 8. Why this is an *effective* use of Ray

- **RPC-on-class-methods** is Ray's killer feature here; verl leans on it so the control
  flow can call `worker.method.remote()` instead of reinventing an RPC layer.
- **Placement groups + `STRICT_PACK`** give verl exact, topology-aware GPU placement, so
  colocated training/inference can use intra-node NCCL for weight transfer.
- **One actor multiplexing many roles** (`WorkerDict`) keeps GPU memory busy and avoids
  idle reserved hardware — the hybrid engine's whole point.
- **Async by default** (`execute_all_async` returns `ObjectRef`s; many methods are
  `blocking=False`) lets independent stages overlap and only synchronizes at collect time.
- **Decoupled control/computation** means you can swap the computation backend
  (FSDP ⇄ Megatron ⇄ torchtitan) or change placement *without touching the algorithm*, and
  vice-versa — the reuse property HybridFlow was designed for.
- **Debuggability**: because the loop is one process, you can set a breakpoint in `fit()`
  and inspect any `DataProto` between stages — the original design goal.

### Gotchas worth knowing

- The driver holds the whole batch between stages; verl recommends **not** scheduling
  `main_task` on the Ray head node (it's memory-hungry, the head is resource-poor).
- Every driver↔worker hop **serializes a `DataProto`** — the price of decoupling. Keeping
  outputs as `DataProtoFuture`s and using `blocking=False` mitigates round-trips.
- Colocation requires roles to fit on the **same** placement group; for *different*
  parallel sizes per role, give each role its own ResourcePool instead of
  `create_colocated_worker_cls`.

---

## Source map

| Area | File |
|---|---|
| Driver entrypoint | `verl/trainer/main_ppo.py` |
| RL loop | `verl/trainer/ppo/ray_trainer.py` (`RayPPOTrainer`, `fit`, `init_workers`) |
| Register / dispatch | `verl/single_controller/base/decorator.py` |
| WorkerGroup binding | `verl/single_controller/base/worker_group.py` |
| Ray WorkerGroup / pools / placement / colocation | `verl/single_controller/ray/base.py` |
| Worker base | `verl/single_controller/base/worker.py` |
| Role workers | `verl/workers/engine_workers.py` |
| Rollout (vLLM hybrid engine) | `verl/workers/rollout/vllm_rollout/` |
| Batch container | `verl/protocol.py` |
| Design notes | `docs/hybrid_flow.rst`, `docs/single_controller.rst` |
