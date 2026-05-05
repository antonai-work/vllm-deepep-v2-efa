# Architecture

## The build chain

One multi-stage Dockerfile produces the final image. Each stage
layers a specific upstream dependency with any patches applied.

```
+-------------------------------------------------------+
| FROM nvidia/cuda:12.9.0-devel-ubuntu24.04             |   Public NVIDIA mirror on Docker Hub.
| Public CUDA 12.9 developer base                       |   Byte-identical to nvcr.io/nvidia/cuda.
+-------------------------------------------------------+
                          |
                          v
+-------------------------------------------------------+
| EFA stack                                             |
|  aws-efa-installer 1.48.0 (--build-ngc flag)          |
|   + rdma-core 61.0 libraries                          |
|   + libfabric 2.4.0amzn (EFA-patched)                 |
|   + aws-ofi-nccl 1.19.0 (installer-bundled NGC)       |
|  NCCL 2.30.4 via `pip install nvidia-nccl-cu13`       |
|  GDRCopy 2.5.1                                        |
|  NVSHMEM wheel (link-only; not used at runtime)       |
|  aws-ofi-nccl source-built at 6e504db (2026-04-24)    |
|   - upstream fix for GIN ring overflow                |
+-------------------------------------------------------+
                          |
                          v
+-------------------------------------------------------+
| DeepEP V2 (patched)                                   |
|  Clone: deepseek-ai/DeepEP@b306af0 (main post-#605)   |
|  Patches 0001, 0002, 0003 from PR #612 applied.       |
|  pip install -e . against NCCL 2.30.4 above.          |
+-------------------------------------------------------+
                          |
                          v
+-------------------------------------------------------+
| vLLM (PR #41183 via fork pin)                         |
|  Clone: tlrmchlsmth/vllm @ deepep-v2-integration       |
|         pinned to SHA 6d7a3fab (2026-05-04)            |
|  VLLM_USE_PRECOMPILED=1 pip install -e .              |
|  (precompiled kernels; no nvcc compile step)           |
|  DeepEP V2 _C.so rebuilt against vLLM's torch ABI.    |
+-------------------------------------------------------+
                          |
                          v
           Final image (~12 GB compressed)
```

## The data flow at runtime

```
Client HTTP POST /v1/chat/completions
      |
      v
vLLM OpenAI-compatible API server (leader pod, port 8000)
      |
      | routes prompt to engine core
      v
vLLM engine core (DP=16, EP=16 across 2 pods * 8 GPUs)
      |
      | MoE layer forward pass
      v
DeepEPV2PrepareAndFinalize (vLLM PR #41183)
      |
      | buffer.dispatch(x, topk_idx, topk_weights, num_experts=N, ...)
      v
DeepEP V2 ElasticBuffer (group=NCCL device PG, num_max_tokens_per_rank,
                         hidden, num_topk, use_fp8_dispatch,
                         allow_hybrid_mode, explicitly_destroy=True)
  (patched: num_allocated_qps=2 on EFA, EFA fast path in get_rdma_gbs,
   kScaleoutUpdateInterval=16)
      |
      | NCCL GIN collective API (ncclTeamTagRail etc.)
      v
NCCL 2.30.4 (pip cu13)
      |
      | net plugin dispatch
      v
aws-ofi-nccl git-6e504db
  (source-built; contains active_put_signal bitset fix)
      |
      | libfabric EFA provider
      v
AWS EFA hardware (rdmap*s0 NICs, SRD transport)
      |
      v
Cross-node GPU-to-GPU RDMA over EFA
```

## Key design decisions

### Why no V1 -> V2 shim?

An earlier approach used a Python-level V1-Buffer-to-V2-ElasticBuffer
compat shim that monkeypatched `deep_ep.Buffer` to delegate to
`ElasticBuffer`. vLLM PR #41183 adds a native
`DeepEPV2PrepareAndFinalize` class plus the `--all2all-backend deepep_v2`
CLI flag, which makes that shim unnecessary on the vLLM side. This
repo does NOT ship a shim - the Dockerfile explicitly asserts
`DEEP_EP_USE_V2_SHIM=0` at runtime, and preflight Check 6 verifies
no `/opt/api-shim` directory exists.

### Why `b306af0` for DeepEP?

That's the merge commit of PR #605 (DeepEP V2 public release,
2026-04-29). Any `main` commit at or after this point contains V2.
We pin to the explicit SHA so reviewers can see the delta as exactly
three commits on top (PR #612).

### Why pin `tlrmchlsmth/vllm@6d7a3fab` instead of vanilla vLLM?

PR #41183 is OPEN in `vllm-project/vllm`. Until it merges, the
only way to get `DeepEPV2PrepareAndFinalize` into the vLLM package
is to install from the PR author's fork branch. SHA `6d7a3fab` is
the PR tip as of 2026-05-04 and contains the "Pass NCCL device
group to DeepEP v2 ElasticBuffer" fix, which is critical on EFA
because ElasticBuffer needs an NCCL-capable process group (the
default gloo group returns from `get_group_from_group_name()` for
the EP group doesn't carry Gin transports).

When PR #41183 merges, the Dockerfile's `git clone` step collapses
to `pip install vllm>=<release tag>`.

### Why NCCL 2.30.4, not newer?

DeepEP V2's `csrc/kernels/backend/nccl.cu` uses the
`ncclTeamTagRail` API which requires >= 2.30.4. vLLM PR #41183
explicitly version-gates `has_deep_ep_v2()` on this same version.
The pip wheel `nvidia-nccl-cu13>=2.30.4` is the standard way to
get it. Newer NCCL works; older does not.

PyTorch pins `nvidia-nccl-cu12==2.27.5` in its dep metadata, which
would shadow the 2.30.4 wheel. The Dockerfile forces 2.30.4 back on
top via `--force-reinstall --no-deps`, then relinks `libnccl.so` to
the 2.30.4 copy.

### Why build aws-ofi-nccl from source AND install the NGC plugin?

Two plugins coexist in the final image:
- `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so` - installer's NGC
  plugin (aws-ofi-nccl 1.19.0, pre-built for NGC)
- `/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so` - source-built at
  `6e504db` (includes the GIN ring `active_put_signal` fix)

The K8s manifest's `NCCL_NET_PLUGIN` env points at the source build
by default. For DeepEP V2's high-concurrency Gin traffic, the
source build is the right choice because the GIN ring fix avoids
the 128-slot overflow. If you're running on a cluster with tested
NGC-only plugins, you can switch `NCCL_NET_PLUGIN` to
`/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so` - both are present.

### Why `--all2all-backend deepep_v2` not `deepep_high_throughput`?

`deepep_high_throughput` is the V1 DeepEP backend (the pre-existing
`Buffer` class). `deepep_v2` is the new backend added by PR #41183
that uses V2 `ElasticBuffer`. On EFA with NCCL Gin, V2 is strictly
better: lower latency, no NVSHMEM/IBGDA dependency, and
cudagraph-capturable in vLLM's decode path (`do_expand=False`).

### Why FSX for weights (optional)?

Qwen3-30B-A3B-FP8 is ~31 GB. Downloading from HuggingFace on every
pod restart is ~15 minutes. A PVC-backed HF cache shares the
weights across the cluster. Optional - you can use any storage
backend; the reference manifest ships with the mount commented out
so the default run works without PVC setup.

## Host requirements

- NVIDIA GPUs with CUDA 9.0+ compute capability (H100 or H200
  recommended; A100 should work but untested)
- EFA-enabled instance type (p4d, p5, p5en)
- Linux kernel >= 5.15 with EFA 2.x driver
- `nvidia-device-plugin` and EFA K8s device plugin on EKS

## What's NOT in this repo

- Pre-built Docker images (you build from source)
- Model weights (you stage Qwen3-30B-A3B-FP8 separately via
  HuggingFace Hub; see the k8s manifest for the optional FSX layout)
- Cluster provisioning (you bring your own EKS + p5.48xlarge
  nodegroup; reference manifests assume HyperPod EKS but work on
  vanilla EKS with EFA plugin)
- Performance tuning beyond the three DeepEP patches (no NCCL
  topology tuning, no NUMA-affinity scripts - those are
  workload-specific)

## Companion repo

For MoE *training* over EFA (Megatron + NeMo-RL), see the sibling
public repo:
[`antonai-work/nemo-rl-deepep-v2-efa`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa).
Both repos share the same DeepEP V2 substrate (same patches
`0001-0003`, same `aws-ofi-nccl@6e504db`, same NCCL 2.30.4 pin) and
only diverge at the consumer layer.
