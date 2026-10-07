# PROJECT-ROADMAP — ComfyUI-XLA on TPU v5e-8

The living plan. It gates scope creep and records intent. Re-read it at every
drift check. It is never deleted — the history is the ledger of what was planned,
what was built, and what drifted.

---

## 1. Intent (verbatim)

> "clone https://github.com/radna0/ComfyUI-XLA and patch it so it works with
> modern models and tpuv5e8"

Unpacked into three verifiable outcomes:

1. **It imports and launches.** Today it does not — it dies with
   `AttributeError: 'Namespace' object has no attribute 'xla_eager'` on *every*
   platform, before any TPU is touched.
2. **It reaches the v5e-8 chips.** Today it cannot — `import torch_xla` raises
   `OSError: Failed to get TPU metadata` on this host.
3. **It can load current-generation models.** Today it cannot — the fork's
   `comfy/ldm` has 11 entries against upstream's 58.

## 2. Current state

| | Status |
|---|---|
| Clone | `origin/radna0/ComfyUI-XLA` @ `ec8a77a`, upstream base merged 2024-12-15 |
| Import on CPU | **BROKEN** — F1 |
| Attach to v5e-8 | **BROKEN** — F2 |
| Model zoo | 11 / 58 vs upstream |
| Host stack | torch 2.8.0+cpu, torch_xla 2.8.0, Python 3.12.13, transformers 5.12.1 |
| Provenance | every claim in `docs/XLA_PORT_MANIFEST.md`, §3 |

Note: this repo's README still advertises `torch~=2.5.0 torch_xla[tpu]~=2.5.0`.
The host runs 2.8.0. The README is stale — see Phase 4.

## 3. Phases

### Phase 1 — Make it run on this host. **DONE (2026-10-07)**
Goal: import succeeds; the TPU attach env is packaged; nothing silently depends
on a v3 constant or on the overlay.
Succeeds when: F1–F5 applied, each with an executed probe that can fail.

### Phase 2 — Prove real TPU compute. **NOT STARTED — needs the runtime agent**
Goal: a synchronised tensor op on v5e, then one generated image.
Succeeds when: `xla.sync()` has run on a real op **and** an image came out of
ComfyUI. Attach is not inference; a device count is not a result.

### Phase 3 — Calibrate SPMD memory for v5e. **NOT STARTED — blocked on Phase 2**
Goal: a `--xla_spmd_mem_divisor` value that is measured, not inherited from v3.
Succeeds when: a real UNet loads on the 8-chip slice and the `SPMD memory:` log
line records the ratio that made it fit.

### Phase 4 — Reach modern models. **NOT STARTED — a rebase, deliberately out of Phase 1's scope**
Goal: re-apply the 7 hunks in `docs/XLA_PORT_MANIFEST.md` §2 onto current
upstream ComfyUI, so `wan`, `qwen_image`, `chroma`, `hunyuan_video`, `hidream`
and the rest become reachable on TPU.
Succeeds when: the rebase carries all 7 hunks (and only those — §2 lists the two
grep false positives to *not* re-add), and the §3 probes still pass on the new base.

### Phase 5 — Update the documentation lie. **NOT STARTED**
Goal: README requirements corrected to the measured stack (torch 2.8.0 /
torch_xla 2.8.0 / Python 3.12), the MDS failure mode documented, and
`xla_tpu_v5e8.sh` documented in-repo.

## 4. Scope-creep defences

**In scope now:** import correctness, TPU attach env, cache/memory/mesh
parameterisation, and the documentation needed to re-use them.

**Out of scope now, and why:**
- *Porting the 58 upstream model dirs.* That is Phase 4, a rebase. Doing it
  inside Phase 1 would put two causal deltas in one experiment.
- *Re-tuning the SPMD divisor to a guessed v5e value.* No measurement exists on
  this host yet; a guess is a claim without a proof. Phase 3 measures it.
- *Installing ComfyUI's missing deps* (`torchsde`, `spandrel`, `kornia`,
  `soundfile`). Real, but a Phase 2 prerequisite, not a Phase 1 defect.
- *Claiming TPU correctness from CPU probes.* Structurally forbidden by the
  custody contract; that evidence does not exist yet.

**To add a phase:** the operator asks, or a Phase-N success criterion is met and
the next criterion is unreachable without new scope.

## 5. Controlled evolution

- A new flag appears only when a *measured* constant had to change. The divisor
  and the mesh both earned a flag because they were hardcoded; nothing else does.
- Every flag keeps its historical default. A patch that improves one host must
  not silently move the floor under every other host.
- A Phase transition requires the previous phase's success criterion, executed —
  not inferred.

**"Note for later"** (inactive; reactivated only if the operator asks or the gap
widens): non-SPMD `--xla` as a genuine multi-chip mode (it replicates, it does not
shard); `tpu_info` chip accounting for non-SPMD; eager mode as a usable default
rather than a flag.

## 6. Drift checkpoints

Re-read this file at every one of these:
1. Before touching any file.
2. After every phase transition.
3. If a change cannot be traced to a line in §3 or §4.
4. Before any commit.

**The one question:** *am I making code correct, or am I realizing intent?*
Correctness is how F1–F5 got found. Intent is Phases 2–4 actually running.