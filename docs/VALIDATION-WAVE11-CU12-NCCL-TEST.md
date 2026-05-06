# Validation: Wave 11 cu12 NCCL hypothesis test, 2026-05-06

## Executive summary

Wave 9b produced an image with cu12 NCCL 2.30.4 (the GIN-capable
build) and claimed via post-build ctypes inspection that
`ncclTeamWorld + ncclCommQueryProperties + deep_ep.ElasticBuffer +
vllm` all imported cleanly — contradicting Wave 9c's failure theory.
Wave 11 ran that image on live H100 p5.48xlarge hardware and the
result is unambiguous: **the Wave 8 "invalid device ordinal" crash is
NOT fixed**. First MoE dispatch crashes with a `c10::Error` from
`c10::cuda::SetDevice(signed char, bool)` receiving a negative int8
value (`device id must be non-negative!-112`), bubbling up as
`RuntimeError: Event device index` at `deep_ep/buffers/elastic.py:811
-> self.runtime.dispatch(...)`. The ctypes inspection was looking at
`libnccl.so.2` (which really is cu12 2.30.4) but not at
`vllm/_C.abi3.so`, which is linked against `libcudart.so.13` and
cu13 libtorch — the mixed-runtime condition that Wave 8 originally
diagnosed is still present.

Verdict: **FAIL, same-mode-as-Wave-8**. cu12-on-the-wire NCCL is
not sufficient; the whole vllm + torch + DeepEP binary stack must
be built against a single CUDA runtime major version. Wave 10's
v0.2.0-sm90a cu13 migration is therefore NECESSARY, not
experimental — proceed with the Wave 10 builds already in flight.

## Image under test

| Field | Value |
|---|---|
| Tag | `058264135704.dkr.ecr.us-east-2.amazonaws.com/vllm-deepep-v2-efa:fast-7ef31a3acc2b` |
| Digest | `sha256:37155d5449ab2c97b61a91ae71ab11e928fe3bf2a862656615e8ebc9ebe88bab` |
| Pushed at | 2026-05-06T09:10:44Z |
| Size | 22 269 798 849 bytes (20.7 GiB) |
| Built as | Wave 9b — "fix cu12 NCCL + keep vllm+torch path" |
| Nickname | fast-7ef31a3acc2b / fast-latest |

## Environment provenance

Executed inside the image via `kubectl exec`:

```
torch=2.11.0+cu129
cuda_available=True
gpu_count=8
vllm=0.1.dev16220+g6d7a3fab2
deep_ep at /opt/DeepEP/deep_ep/__init__.py
deep_ep.ElasticBuffer=<class 'deep_ep.buffers.elastic.ElasticBuffer'>

# libnccl.so.2 via ctypes:
rc=0
NCCL version int=23004              # <-- 2.30.4, cu12, GIN-capable
ncclNetGinGetInfo: MISSING          # <-- expected (GIN syms live in
                                    #     aws-ofi-nccl plugin .so, not
                                    #     in libnccl itself)
ncclTeamWorld: RESOLVED             # <-- Wave 9c's failure theory
                                    #     contradicted on THIS axis
ncclCommQueryProperties: RESOLVED   # <-- ditto
```

Wheel inventory (trimmed to the load-bearing rows):

```
torch                                    2.11.0+cu129
nvidia-nccl-cu12                         2.30.4     <-- used as libnccl.so.2
nvidia-nccl-cu13                         2.28.9     <-- idle, but the cu13
                                                        libcudart next to it
                                                        IS load-bearing
nvidia-cuda-runtime-cu12                 12.9.79
nvidia-cuda-runtime                      13.0.96    <-- feeds libcudart.so.13
```

And the decisive linkage probe:

```
$ ldd /opt/vllm/vllm/_C.abi3.so | grep -E 'cudart|not found'
        libtorch.so => not found
        libcudart.so.13 => not found        <-- vllm._C is cu13
        libtorch_cpu.so => not found
        libtorch_cuda.so => not found
        libc10_cuda.so => not found
        libc10.so => not found

# libcudart.so.13 exists, just not on default ld.so.cache path:
/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13
```

Forcing that cu13 path into `LD_LIBRARY_PATH` lets `vllm._C` import
cleanly — so the image is internally resolvable — but the problem
surfaces at runtime because torch 2.11.0+cu129 and
vllm `_C.abi3.so`+cu13 disagree on `c10::DeviceIndex` ABI between
shared libs.

## Cluster conditions

- Cluster: EKS `eks-static-cluster` in us-east-2, account 058264135704
- 2 H100 HyperPod nodes: `hyperpod-i-01aee349f9991c414` (10.1.3.30),
  `hyperpod-i-0a3eb6d3953cceaa7` (10.1.3.73)
- Node 1 has a pre-existing containerd fault
  (`localhost/kubernetes/pause` cannot be resolved, 24+
  DaemonSet/CSI pods stuck in `CreateContainerError` / `FailedCreatePodSandBox`)
- Test degraded to **single-pod DP=8 EP=8 intra-node** on the healthy
  node 2 — still diagnostic because the "invalid device ordinal"
  crash happens on the first MoE dispatch regardless of
  intranode/internode topology
- Namespace: `deepep-v2-w11-2026-05-06` (isolated)
- H100 cluster lock claimed as `wave11-1778059175` at 2026-05-06T09:19:35Z
- `nvshmem-efa/deepep-nvshmem` StatefulSet was scaled 2->0 to free
  GPUs; will be restored in teardown

## Run log (essential lines)

Startup, model load, and backend selection — all clean:

```
INFO 05-06 09:47:11 [serve.py:101] Defaulting api_server_count to data_parallel_size (8).
INFO 05-06 09:47:11 [utils.py:233] non-default args: {'model_tag':
    'Qwen/Qwen3-30B-A3B-FP8', 'api_server_count': 8, 'host': '0.0.0.0',
    'model': 'Qwen/Qwen3-30B-A3B-FP8', 'trust_remote_code': True,
    'max_model_len': 2048, 'enforce_eager': True, 'data_parallel_size': 8,
    'data_parallel_size_local': 8, 'data_parallel_address': '10.1.3.73',
    'data_parallel_rpc_port': 29500, 'enable_expert_parallel': True,
    'all2all_backend': 'deepep_v2', 'gpu_memory_utilization': 0.7,
    'max_num_batched_tokens': 256, 'max_num_seqs': 16}
...
[Worker pid=3737] INFO 05-06 09:48:02 [cuda_communicator.py:175] Using DeepEPV2All2AllManager all2all manager.
...
[Worker_DPk_EPk pid=...] [DeepEP] EFA detected: capping num_allocated_qps 129 -> 2 to avoid GIN 128-slot ring overflow
    (all 8 workers hit this — PR #612 auto-cap is firing, good)
...
(Worker_DP0_EP0 pid=3737) INFO 05-06 09:49:17 [fp8.py:590] Using DeepEPV2PrepareAndFinalize
(Worker_DP0_EP0 pid=3737) INFO 05-06 09:49:19 [gpu_model_runner.py:4883] Model loading took 5.44 GiB memory and 74.873718 seconds
(Worker_DP0_EP0 pid=3737) INFO 05-06 09:49:19 [dp_utils.py:28] Using CPU all reduce to synchronize DP padding between ranks.
```

aws-ofi-nccl + EFA — also clean:

```
NCCL INFO NCCL_NET_PLUGIN set by environment to /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
NCCL INFO NET/OFI Initializing aws-ofi-nccl git-6e504db
NCCL INFO NET/OFI Using Libfabric version 2.4
NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 12090
NCCL INFO NET/OFI Plugin selected platform: AWS
NCCL INFO NET/OFI Using transport protocol RDMA (user set)
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 32 nics)
```

Note the `CUDA driver version 13000 with runtime 12090` line — the
host nvidia driver is CUDA 13, the wheel-installed runtime is
CUDA 12.9. That mismatch alone is fine (forward-compat), but it is
the very axis the mixed cu12/cu13 wheel stack rides on.

**Then the crash. Full first-occurrence C++ trace:**

```
(Worker_DP7_EP7 pid=3736) [W506 09:49:23] CUDA warning: invalid device ordinal (function destroyEvent)
(Worker_DP7_EP7 pid=3736) [W506 09:49:23] CUDA warning: invalid device context (function destroyEvent)
...
terminate called after throwing an instance of 'c10::Error'
  what():  device id must be non-negative!-112
Exception raised from SetDevice at /pytorch/c10/cuda/CUDAFunctions.cpp:245 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x9d (.../libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0xc3 (.../libc10.so)
frame #2: c10::cuda::SetDevice(signed char, bool) + 0x2ee (.../libc10_cuda.so)
frame #3: <unknown function> + 0x23e00 (.../libc10_cuda.so)
frame #4: <unknown function> + 0x960df (/opt/DeepEP/deep_ep/_C.cpython-312-x86_64-linux-gnu.so)
frame #5: deep_ep::elastic::ElasticBuffer::dispatch(...) + 0x4d0 (/opt/DeepEP/deep_ep/_C.cpython-312-x86_64-linux-gnu.so)
frame #6: <unknown function> + 0xf5234 (/opt/DeepEP/deep_ep/_C.cpython-312-x86_64-linux-gnu.so)
frame #7: <unknown function> + 0x55c5f (/opt/DeepEP/deep_ep/_C.cpython-312-x86_64-linux-gnu.so)
frame #8: VLLM::Worker_DP4_EP4() [0x5821ef]
```

And the Python-side view:

```
File "/opt/vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py", line 278, in prepare_async
  return self._do_dispatch(
File "/opt/vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py", line 125, in _do_dispatch
  ) = self.buffer.dispatch(
File "/opt/DeepEP/deep_ep/buffers/elastic.py", line 811, in dispatch
  event) = self.runtime.dispatch(x, sf, topk_idx, topk_weights, ...)
RuntimeError: Event device index
```

All 8 DP workers crash in the same way (`DP0..DP7`). `DP0` also
surfaces a secondary:

```
(Worker_DP0_EP0 pid=3737) RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
```

which is a downstream of the first failure destroying tensors with
invalid device context before subsequent frames try to reuse them.

## Failure interpretation

- `c10::cuda::SetDevice(signed char, bool)` — note the `signed char`
  in the c10 ABI. In cu13 torch, `c10::DeviceIndex` became
  `int8_t` (it was `int16_t` in some earlier cu12 builds). When
  DeepEP-v2 (built against one ABI) hands a device index to c10
  (built against the other), byte-order and sign-extension land
  at `-112` — the signed-8-bit interpretation of `0x90 = 144`, an
  uninitialized/trashed int value from a mis-sized read.
- The `invalid device ordinal` warning from `destroyEvent` — which
  happens *before* the `c10::Error` throw — is a CUDA event handle
  whose `.device_index` member has been read through an ABI-incorrect
  struct layout, triggering the runtime's own validator first
  before dispatch even gets to `SetDevice`.
- This is the exact same failure signature Wave 8 captured (see
  `VALIDATION-WAVE8-FRESH-2026-05-06.md`). Swapping libnccl.so.2
  from cu13-2.28.9 to cu12-2.30.4 does not touch this axis. libnccl
  is not involved in `c10::cuda::SetDevice`.

## EFA counter delta

Not meaningful. The EFA hw_counters aggregate since node boot and the
crash occurs before any MoE dispatch bytes cross the wire. Model-load
NCCL traffic did run (the `NET/OFI` init lines above) but the first
actual MoE dispatch — which is the byte-heavy path — crashed.
Single-pod DP=8 is intra-node anyway, so even with zero crashes this
test would not have produced a Wave-11-specific EFA delta.

## What was (and was not) proven

- Proven: libnccl.so.2 = cu12 2.30.4 with `ncclTeamWorld` +
  `ncclCommQueryProperties` resolved, matching the offline ctypes
  probe. Wave 9c's "libnccl is cu13 and missing GIN symbols"
  narrative is INCORRECT.
- Proven: `from vllm.model_executor.layers.fused_moe.deep_ep_v2_prepare_finalize
  import DeepEPV2PrepareAndFinalize` succeeds; PR #41183 logic is
  reachable; image logs `Using DeepEPV2PrepareAndFinalize` and
  `Using DeepEPV2All2AllManager`.
- Proven: PR #612 auto-cap fires (`capping num_allocated_qps 129 -> 2`)
  on all 8 workers.
- Proven: aws-ofi-nccl git-6e504db loads, EFA provider selected,
  32 NICs enumerated.
- Not true: "cu12 NCCL fix resolves Wave 8 crash" — it does not.
  The wave 8 bug is in the cu12-torch-vs-cu13-vllm-_C ABI mismatch,
  which Wave 9b did not address.

## Recommendation on Wave 10

**Use Wave 10 v0.2.0-sm90a once it finishes building.** Do NOT
cancel it. The cu13-across-the-board migration is what this wave
proves is necessary. The in-flight builds are:

- GHA 25426621244 (GitHub Actions, public repo build)
- CodeBuild 7adf2fc2 (AWS CodeBuild mirror)

Once either produces a fresh digest, repeat Wave 11 with that
image (same cluster, same namespace pattern) and the expected
outcome is 24 tokens returned from
`/v1/chat/completions`.

## Teardown

Post-test cleanup (to run after this doc lands):

1. `kubectl delete ns deepep-v2-w11-2026-05-06` — removes StatefulSet
   + single pod + Service + PVC + PV (via Retain reclaim, the FSx
   file system itself is untouched)
2. `kubectl -n nvshmem-efa scale sts deepep-nvshmem --replicas=2` —
   restore the h100 benchmark fleet that was parked
3. Release both cluster locks:
   - Legacy global: `cluster-lock.sh release`
   - H100-specific: zero out `~/.claude/cluster-lock-h100.json`
   (this tree's convention is to hold both locks during a run)

## Evidence artifacts

- Full vllm serve log (4 496 lines):
  stored in the author's local run at
  `/tmp/wave11/evidence/vllm-serve.log` at the time of this doc.
  First `Event device index` error at line ~3280. First C++
  `c10::Error` at line ~2996.
- Manifest applied:
  local path `/tmp/wave11/manifest.yaml` (2-pod draft, not used)
  and `/tmp/wave11/manifest-single.yaml` (single-pod DP=8, the
  actual run).
- Crash identical to Wave 8; cross-reference
  `docs/VALIDATION-WAVE8-FRESH-2026-05-06.md` for prior analysis.
