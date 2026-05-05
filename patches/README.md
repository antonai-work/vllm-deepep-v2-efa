# Patches - standalone extracts of the two upstream PRs

Four files:
- Three DeepEP patches (`0001`-`0003`) as `git format-patch` output from
  our open upstream PR [#612](https://github.com/deepseek-ai/DeepEP/pull/612).
  Apply with `git am` inside a fresh `deepseek-ai/DeepEP` checkout.
- One vLLM diff (`vllm-pr41183-deepep-v2.diff`) captured from the open
  upstream PR [#41183](https://github.com/vllm-project/vllm/pull/41183).
  This is the raw `diff --git` output, not a `git format-patch`;
  apply with `git apply` inside a fresh `vllm-project/vllm` checkout.

## Upstream PRs

| Patch files | Upstream repo | PR | Target base |
|---|---|---|---|
| `0001`, `0002`, `0003` | [`deepseek-ai/DeepEP`](https://github.com/deepseek-ai/DeepEP) | [#612 "AWS EFA optimizations for V2"](https://github.com/deepseek-ai/DeepEP/pull/612) | `main@b306af0` |
| `vllm-pr41183-deepep-v2.diff` | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | [#41183 "[WideEP] Integrate DeepEP v2"](https://github.com/vllm-project/vllm/pull/41183) | current PR tip, pinned SHA `6d7a3fab` (2026-05-04) |

## Applying patches

The `docker/Dockerfile` at the repo root does all of this automatically;
these instructions are for reviewers who want to inspect manually.

### DeepEP (patches 0001-0003)

```bash
git clone https://github.com/deepseek-ai/DeepEP
cd DeepEP
git checkout b306af0  # main at time of PR rebase
git am /path/to/patches/0001-*.patch /path/to/patches/0002-*.patch /path/to/patches/0003-*.patch
```

### vLLM (vllm-pr41183-deepep-v2.diff)

Default path - clone the PR author's fork directly, which is what the
`docker/Dockerfile` in this repo does:

```bash
git clone https://github.com/tlrmchlsmth/vllm -b deepep-v2-integration
cd vllm
git checkout 6d7a3fab  # PR #41183 tip as of 2026-05-04
VLLM_USE_PRECOMPILED=1 pip install -e .
```

Offline / air-gapped path - apply the bundled `.diff` on top of vanilla
`vllm-project/vllm` at a compatible base:

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
# Any main SHA between 2026-04-29 and 2026-05-04 should apply cleanly.
git apply /path/to/patches/vllm-pr41183-deepep-v2.diff
VLLM_USE_PRECOMPILED=1 pip install -e .
```

The `.diff` path is useful when the fork is unreachable (network-restricted
build farm), when you want a review-grade record of exactly what's on top
of upstream, or when the PR rebases and you want to freeze a known-good
snapshot.

## Patch-by-patch summary

### 0001 - DeepEP: cap auto-QP at 2 on EFA

`deep_ep/buffers/elastic.py` - when EFA is detected, override the default
of 129 QPs down to 2. Prevents wasteful QP allocation (EFA has a tighter
budget than Mellanox). Previously also prevented a crash, but
aws-ofi-nccl commit `6e504db` (2026-04-24) fixed the upstream ring-overflow
issue. Now perf-only on the crash axis, still the right default for EFA
resource efficiency.

### 0002 - DeepEP: EFA fast path in `get_rdma_gbs()`

`deep_ep/utils/envs.py` - `check_fast_rdma_atomic_support()` calls
`ibstat`, which is not installed on EFA instances. Without this patch,
SM auto-sizing gets a garbage NIC-bandwidth estimate leading to ~2x
perf regression. This commit adds an `FI_PROVIDER=efa` check and returns
25 GB/s directly.

### 0003 - DeepEP: raise `kScaleoutUpdateInterval` 3->16

`deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` - tunes how often the
hybrid_dispatch kernel flushes scaleout state updates. 3 is aggressive
for high-bandwidth fabrics; 16 is more appropriate for EFA's CPU-proxy
Gin backend. Measured ~22% D+C latency improvement on 2-node p5.48xlarge.

### vllm-pr41183-deepep-v2.diff - vLLM: integrate DeepEP V2

Adds the `deepep_v2` all2all backend to vLLM. New surface area:

- CLI flag `--all2all-backend deepep_v2`
- New file `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py`
  with `DeepEPV2PrepareAndFinalize` class
- New class `DeepEPV2All2AllManager` in the all2all manager registry
- New probe `vllm.utils.import_utils.has_deep_ep_v2()` (version-gated on
  NCCL >= 2.30.4)
- New env vars `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE`,
  `VLLM_DEEPEP_V2_PREFER_OVERLAP`, `VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION`

The PR uses DeepEP V2's MoE-shape `ElasticBuffer(group=pg,
num_max_tokens_per_rank=..., hidden=..., num_topk=..., use_fp8_dispatch=...,
explicitly_destroy=True)` constructor, which is the same ctor shape
used by `tests/elastic/test_ep.py` in the DeepEP upstream tree.

The PR author notes in the PR body: "I couldn't get this working on an
8xB200 system as DeepEP v2's ElasticBuffer unconditionally asserts NCCL
GIN availability even for intra-node NVLink-only." On AWS EFA with the
Gin plugin loaded, that assertion succeeds, which is exactly the regime
this repo targets.

## Notes on provenance

- `0001-0003` are `git format-patch` output against
  `deepseek-ai/DeepEP@b306af0`. SHAs in the patch headers are from the
  PR #612 fork branch.
- `vllm-pr41183-deepep-v2.diff` is the raw diff captured from vLLM
  PR #41183 at commit `6d7a3fab`. The Dockerfile does not apply this
  diff at build time; it clones the fork directly. The diff is bundled
  here so that (a) reviewers can see the exact change set offline, and
  (b) air-gapped build farms can reproduce without network access to
  the fork.

## What happens when PRs merge

### DeepEP PR #612 merges

- Drop `DEEPEP_REPO_URL=...` override from `docker/Dockerfile` (it
  already defaults to vanilla `deepseek-ai/DeepEP.git`).
- Drop the `git am /tmp/patches/0001..03-deepep.patch` step.
- Delete `patches/0001-*.patch`, `patches/0002-*.patch`,
  `patches/0003-*.patch`.
- Update `docs/UPSTREAM-STATUS.md` row.

### vLLM PR #41183 merges

- Change `VLLM_REPO_URL` in `docker/Dockerfile` back to
  `https://github.com/vllm-project/vllm.git`, or replace the `git clone`
  step entirely with `pip install vllm>=<release tag that contains PR #41183>`.
- Delete `patches/vllm-pr41183-deepep-v2.diff`.
- Update `docs/UPSTREAM-STATUS.md` row.

When both merge, `docker/Dockerfile`'s third-party content is pure
vanilla with no patches.
