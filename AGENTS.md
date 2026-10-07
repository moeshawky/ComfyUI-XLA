# AGENTS.md — ComfyUI-XLA (TPU/XLA backend for ComfyUI)

## Project purpose

**Intent:** Make `radna0/ComfyUI-XLA` run correctly on a Kaggle **TPU v5e-8** slice
(2x4) against current-generation model checkpoints, without losing the XLA backend
that upstream ComfyUI does not have.

**Project roadmap:** PROJECT-ROADMAP.md — the living plan that gates scope creep and guides controlled evolution.

## Read this before touching code

| File | What it is |
|---|---|
| `PROJECT-ROADMAP.md` | Intent, current state, phases, scope-creep defences. **Read at every drift check.** |
| `docs/XLA_PORT_MANIFEST.md` | The complete XLA surface (7 hunks, 3 real files), every defect found and its probe, and the rebase checklist. |

## The three facts that will waste your time if you do not know them

1. **`import torch_xla` requires `TPU_SKIP_MDS_QUERY=1`.** Without it, torch_xla
   HTTP-GETs the GCE metadata server for `instance/attributes/tpu-env` and dies
   with `OSError: Failed to get TPU metadata`. Use `./xla_tpu_v5e8.sh`.

2. **`TPU_ACCELERATOR_TYPE` must satisfy torch_xla's regex**
   `^v(\d)([A-Za-z]?){7}-(\d+)$`. `v5litepod-8` matches → `version() == 5` →
   one XLA device per chip (8 total) and the v5 `LIBTPU_INIT_ARGS`.

3. **`/tmp` is not a scratch disk here.** `/`, `/tmp` and `/kaggle/temp` are one
   Docker overlay with a scarce COW store, and `df` lies about it. `/dev/shm` is
   real RAM. Compile cache and `HF_HOME` belong on `/dev/shm`. The launcher sets
   both.

## Working rules for this repo

- **One causal delta per change.** A patch that fixes two things at once proves
  neither. Every change here is tied to a numbered defect in
  `docs/XLA_PORT_MANIFEST.md` §3.
- **Every new flag keeps its historical default.** These patches exist to make
  *this* host work; they must not move the floor under every other host.
- **Grep is not evidence.** `grep -i xla` matches `XLabs-AI` and `mistoline` in
  this tree. §2 of the manifest lists the two known false positives.
- **`num_available_devices() == 8` is not proof of inference.** It counts
  PCI-attached chips. Prove compute with a synchronised tensor op, then with a
  generated image. Attach ≠ inference.
- **Do not claim TPU behaviour from CPU probes.** Source changes and live-TPU
  correctness are separate roles (see the TPU Flash Custody Contract in the
  operator's AGENTS.md). Hand the runtime agent commands; do not assert results.

## Entry gate

1. Read `PROJECT-ROADMAP.md` §1 (intent) and §4 (scope-creep defences).
2. Confirm the work maps to a phase in §3.
3. If it does not, stop and say so — do not "while I'm here" it.