# Upstream PR status

Live tracking of the two upstream PRs this repo depends on. When
both merge, this repo's `docker/Dockerfile` can drop the patch-apply
and fork-clone steps and install vanilla upstream instead.

Last updated: 2026-05-05.

## Summary

| Repo | PR | State | Estimate |
|---|---|---|---|
| [`deepseek-ai/DeepEP`](https://github.com/deepseek-ai/DeepEP) | [#612](https://github.com/deepseek-ai/DeepEP/pull/612) | OPEN | Filed 2026-04-28, rebased + pinged 2026-05-05; waiting on maintainer review |
| [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | [#41183](https://github.com/vllm-project/vllm/pull/41183) | OPEN | Filed 2026-04-29, 16 commits through 2026-05-04; author is a vLLM team member, actively iterating |

## PR #612: DeepEP AWS EFA optimizations

- **URL**: https://github.com/deepseek-ai/DeepEP/pull/612
- **Head branch**: `dmvevents/DeepEP-1:aws-efa-auto-qp-cap`
  (3 commits on top of `deepseek-ai/DeepEP@main`)
- **Filed**: 2026-04-28
- **Last activity**: 2026-05-05 (rebase + courtesy ping)
- **State**: OPEN, 0 reviews, 1 comment (maintainer pending)

### Commits

| SHA | Message |
|---|---|
| `fe20874` | `aws-efa: cap auto-QP at 2 on EFA to avoid 128-slot GIN ring overflow` |
| `0b78333` | `aws-efa: add EFA fast path in get_rdma_gbs to fix SM auto-sizing` |
| `c84dcac` | `aws-efa: raise dispatch kScaleoutUpdateInterval from 3 to 16` |

### What happens when this merges

- Drop `DEEPEP_REPO_URL` override in `docker/Dockerfile` (already
  defaults to vanilla `deepseek-ai/DeepEP.git`).
- Drop the `git am /tmp/patches/0001..03-deepep.patch` step.
- Delete `patches/0001-*.patch`, `patches/0002-*.patch`,
  `patches/0003-*.patch`.
- Remove this section from `UPSTREAM-STATUS.md`.

## PR #41183: vLLM [WideEP] Integrate DeepEP v2

- **URL**: https://github.com/vllm-project/vllm/pull/41183
- **Title**: `[WideEP] Integrate DeepEP v2`
- **Author**: `tlrmchlsmth` (Tyler Michael Smith, vLLM team member)
- **Head branch**: `tlrmchlsmth/vllm:deepep-v2-integration`
  (16 commits on top of `vllm-project/vllm@main`)
- **Filed**: 2026-04-29
- **Last activity**: 2026-05-04 (`6d7a3fab`:
  "Pass NCCL device group to DeepEP v2 ElasticBuffer")
- **State**: OPEN, labels `needs-rebase`, `nvidia`
- **CI**: DCO, Meta, readthedocs, Mergify Summary all SUCCESS.
  Only pre-commit (mypy/format) nits and a rebase conflict pending.
- **Merge gate**: reviewer approvals + rebase onto current main + a
  clean pre-commit pass.

### Commits (tip to base)

Commit tip `6d7a3fab` (2026-05-04): "Pass NCCL device group to
DeepEP v2 ElasticBuffer" - the all2all managers were passing the
gloo `cpu_group` which works in production but hangs in the
multi-process test harness; PR switches to the NCCL `device_group`
from the EP group's CudaCommunicator. Critical on EFA because
ElasticBuffer needs an NCCL-capable process group to register
Gin transports.

Commit `a2a4b00f` (2026-04-30): "Add DeepEP v2 (ElasticBuffer)
all2all backend for MoE EP" - the main PR commit. Adds new
`DeepEPV2PrepareAndFinalize` class using `do_expand=True` /
`do_expand=False` modes, `DeepEPV2All2AllManager` with
`ElasticBuffer` handle caching, `has_deep_ep_v2()` version gate,
FP8 block-quantized dispatch, DBO micro-batching support, and
three new env vars (`VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE`,
`VLLM_DEEPEP_V2_PREFER_OVERLAP`,
`VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION`).

### Key PR body note

> I couldn't get this working on an 8xB200 system as DeepEP v2's
> ElasticBuffer unconditionally asserts NCCL GIN availability even
> for intra-node NVLink-only. This is a TODO.

**Implication for AWS EFA**: on p5 / p5en with aws-ofi-nccl +
NCCL_GIN_ENABLE=1 + NCCL_GIN_TYPE=2, the Gin plugin IS present so
that assertion succeeds. This repo targets exactly that regime,
and is strictly additive evidence that the PR's implementation is
correct on a Gin-capable fabric.

### What happens when this merges

- Change `VLLM_REPO_URL` in `docker/Dockerfile` back to
  `https://github.com/vllm-project/vllm.git`, or more simply
  replace the clone+editable-install with
  `pip install vllm>=<first release containing PR #41183>`.
- Drop the `VLLM_BRANCH` / `VLLM_SHA` ARGs.
- Delete `patches/vllm-pr41183-deepep-v2.diff`.
- Remove this section from `UPSTREAM-STATUS.md`.

## Known limit: `deep_gemm` for FP8 MoE

vLLM's DeepEP V2 FP8 MoE kernel path calls into the `deep_gemm` library
for per-expert FP8 GEMMs. There is no pre-built `deep_gemm` wheel on
PyPI as of 2026-05-05; the upstream `deepseek-ai/DeepGEMM` repo is
source-only (SM90 build required).

The Wave 7a `docker/Dockerfile` attempts, in order:
1. `pip install deep_gemm` (PyPI)
2. `pip install git+https://github.com/deepseek-ai/DeepGEMM.git`
3. Log a warning and continue if both fail.

When `deep_gemm` is absent, Qwen3-FP8 MoE inference falls back to
bfloat16 at runtime via:

```
VLLM_USE_DEEP_GEMM=0 \
  vllm serve Qwen/Qwen3-30B-A3B --dtype bfloat16 ...
```

This fallback is functional and matches the Wave 6 validated path
(Run #10: 24 tokens Qwen3-FP8 DP=16 EP=16 over 2-node EFA with
`VLLM_USE_DEEP_GEMM=0`). Upgrade path: when `deep_gemm` gets a PyPI
wheel, the Dockerfile will auto-pick it up and FP8 native kernels
become available with no user-side changes.

## Related issues and PRs (for reviewers unfamiliar with the context)

- **DeepEP V2 release**:
  [deepseek-ai/DeepEP#605](https://github.com/deepseek-ai/DeepEP/pull/605)
  (merged 2026-04-29 as `b306af0`)
- **aws-ofi-nccl GIN ring fix**:
  [aws/aws-ofi-nccl@6e504db](https://github.com/aws/aws-ofi-nccl/commit/6e504db)
  (merged 2026-04-24; superseded our closed PR
  [#1206](https://github.com/aws/aws-ofi-nccl/pull/1206))
- **vLLM DeepEP Blackwell issue**:
  [vllm-project/vllm#41687](https://github.com/vllm-project/vllm/issues/41687)
  (separate track from this EFA work)
- **Sibling training stack**:
  [`antonai-work/nemo-rl-deepep-v2-efa`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa)
  - uses the same DeepEP V2 substrate with Megatron + NeMo-RL on
  the training side. Shares patches `0001-0003`.
