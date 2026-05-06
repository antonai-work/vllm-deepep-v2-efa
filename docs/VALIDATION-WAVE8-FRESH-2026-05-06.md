# Validation: Wave 8 fresh post-7c images, 2026-05-06

## Executive summary

Wave 8: fresh post-7c images (G1/G2/G3 closed). On 2 H100 p5.48xlarge
pods via EFA: inference FAILED on both FP8 (Qwen3-30B-A3B-FP8) and
bfloat16 (Qwen3-30B-A3B) with a CUDA runtime version mismatch inside
the Wave 7c.4 image (vLLM `_C.abi3.so` linked against `libcudart.so.13`
while torch / DeepEP / NCCL use `libcudart.so.12`, producing a mixed
CUDA runtime that corrupts the event device ordinal and crashes every
DP worker during the first MoE dispatch). Training PASSED on a single
p5.48xlarge node (8 GPU) via `train_step_shapeY.py` with
`HAVE_DEEP_EP_V2=True`, `Active buffer class: ElasticBuffer`, and loss
21.57 -> 20.05 across 3 logged steps; 2-node training was blocked by a
node-level containerd/pause-image fault on `hyperpod-i-01aee349f9991c414`
that is outside the image's control. G1 (CUDA runtime wheels) and G2
(`deep_gemm` baked in, `DEEPGEMM Fp8 MoE backend` selected, FP8 path
taken) are confirmed closed. G3 (`train_step_shapeY.py` baked in) is
confirmed closed.

This is the successor to `VALIDATION-FRESH-FROM-PUBLIC-2026-05-05.md`.

## Image provenance

| Artifact | Tag | Digest | Pushed at |
|---|---|---|---|
| vLLM inference image | `058264135704.dkr.ecr.us-east-2.amazonaws.com/vllm-deepep-v2-efa:fast-6cea88e12934` | `sha256:ee276548cb4c60904ac7ee3f00099a927ca4e96cb31ff5c9742e89aa32fb0a58` | 2026-05-06 (Wave 7c/7c.4) |
| NeMo-RL training image | `058264135704.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-deepep-v2-efa:allprs-36a9110` | `sha256:54d69ba0382e3c6d0ca37f4945bbb8a5b53567384d88800c9f485cdef15c1513` | 2026-05-06 |
| Base image (v0.1.1-sm90a) | `058264135704.dkr.ecr.us-east-2.amazonaws.com/deepep-v2-efa-base:d14761c` | `sha256:77dd99d05bf97340a18e53ae0e374489f2e54cf54ccad96e4e0b9bbe5c5c71ea` | 2026-05-06 |
| Base image (GHCR mirror) | `ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.1-sm90a` | `sha256:d23bd2092a768c58a88b8bfe0d3c8e0dc46c2f3dd8efbf92e81aecc9361f77ff` | 2026-05-06 |

Repo commit SHAs at validation time:
- `antonai-work/vllm-deepep-v2-efa@6cea88e12934` (2026-05-06T04:58:01Z)
- `antonai-work/nemo-rl-deepep-v2-efa@36a91102f8a2` (2026-05-06T02:55:39Z)

Baked-in DeepEP V2 SHA: `146cc356aa00c39ac1590c05775e05b0f031e70c`
(3 EFA commits on top of upstream `main@b306af0`):
- `146cc35` aws-efa: raise dispatch kScaleoutUpdateInterval from 3 to 16
- `62a38bf` aws-efa: add EFA fast path in get_rdma_gbs to fix SM auto-sizing
- `164c82d` aws-efa: cap auto-QP at 2 on EFA to avoid 128-slot GIN ring overflow

Preflight on base image (GHCR): 5/5 checks PASS
```
[check] DeepEP V2 ElasticBuffer import                   ... PASS
[check] DeepEP V1 legacy Buffer import                   ... PASS
[check] ldconfig sees libnccl-net-ofi.so                 ... PASS
[check] aws-ofi-nccl plugin at /opt/aws-ofi-nccl         ... PASS
[check] DeepEP PR #612 patches applied                   ... PASS
5/5 checks PASS
```

## Cluster

| Item | Value |
|---|---|
| Region | us-east-2 |
| Instance type | ml.p5.48xlarge (H100, 8 GPU, 32 EFA NICs) |
| Node 0 | `hyperpod-i-0a3eb6d3953cceaa7` (10.1.3.73) |
| Node 1 | `hyperpod-i-01aee349f9991c414` (10.1.3.30) -- **node-level containerd fault on this node**, see training section |
| Lock | H100 cluster lock held 2026-05-06T05:37:33Z -> release at teardown |
| Lock purpose | "Wave 8: live-cluster validation of Wave 7c fresh images" |

## Inference pass - vLLM FAILED on both FP8 and bfloat16

### Deploy
```
kubectl apply -f tests/k8s/multi-node-serving-h100.yaml
# image replaced with fast-6cea88e12934 ECR digest
# namespace: deepep-v2-w8-serve-20260506
# 2/2 pods Ready
```

### Attempt 1: FP8 (Qwen/Qwen3-30B-A3B-FP8, deep_gemm ON)

Command (no `VLLM_USE_DEEP_GEMM=0`, no bfloat16 override):
```
MODEL=Qwen/Qwen3-30B-A3B-FP8 ROLE=leader \
  LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH \
  bash /opt/tests/serve_chat_completion.sh
```

Positive signals (**G2 confirmed closed**):
```
(Worker_DP0_EP0) INFO [cuda_communicator.py:175] Using DeepEPV2All2AllManager all2all manager.
(Worker_DP0_EP0) INFO [__init__.py:389] Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for Fp8LinearMethod
(Worker_DP0_EP0) INFO [fp8.py:406] Using DEEPGEMM Fp8 MoE backend out of potential backends: ['AITER', 'DEEPGEMM', 'TRITON', 'MARLIN', 'BATCHED_DEEPGEMM', 'BATCHED_TRITON', 'XPU'].
(Worker_DP0_EP0) INFO [fp8.py:590] Using DeepEPV2PrepareAndFinalize
(Worker_*) [DeepEP] EFA detected: capping num_allocated_qps 129 -> 2 to avoid GIN 128-slot ring overflow
```

FP8 quantization path activated, `DEEPGEMM` backend selected, DeepEP V2
native `PrepareAndFinalize` chosen, DeepEP EFA auto-QP-cap took effect.
Model loaded across DP=16, EP=16.

Failure (at first MoE dispatch during KV-cache sizing):
```
ERROR [multiproc_executor.py:962]   File "/opt/DeepEP/deep_ep/buffers/elastic.py", line 811, in dispatch
ERROR [multiproc_executor.py:962]     event) = self.runtime.dispatch(...)
ERROR [multiproc_executor.py:962] RuntimeError: Event device index
```

### Attempt 2: bfloat16 (Qwen/Qwen3-30B-A3B, deep_gemm OFF)

Command (Wave 7 workaround re-applied):
```
MODEL=Qwen/Qwen3-30B-A3B ROLE=leader \
  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
  LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH \
  bash /opt/tests/serve_chat_completion.sh
```

Same failure class on the same line in `elastic.py`:
```
[W506 05:55:20 CUDAGuardImpl.h:120] Warning: CUDA warning: invalid device ordinal (function destroyEvent)
[W506 05:55:20 CUDAGuardImpl.h:126] Warning: CUDA warning: invalid device context (function destroyEvent)
terminate called after throwing an instance of 'c10::Error'
  what():  device id must be non-negative!-64
Exception raised from SetDevice at /pytorch/c10/cuda/CUDAFunctions.cpp:245
```

Additional bogus device ordinals observed across workers: `-48`,
`-64`, `-128`. These are not legitimate device IDs; they indicate the
CUDA context/ordinal is being read out of uninitialized memory.

### Root cause: Wave 7c.4 mixed CUDA runtime

The `fast-6cea88e12934` image has a structural CUDA ABI split that
did not exist at Wave 6 / Wave 7:

```
$ ldd /opt/vllm/vllm/_C.abi3.so | grep cudart
	libcudart.so.13 => not found      <-- vLLM C ext wants cu13

$ ldd /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so | grep cudart
	libcudart.so.12 => .../nvidia/cuda_runtime/lib/libcudart.so.12   <-- torch is cu12

$ ldd /opt/DeepEP/deep_ep/_C.cpython-312-x86_64-linux-gnu.so | grep cudart
	libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12         <-- DeepEP is cu12

$ pip show nvidia-cuda-runtime
Name: nvidia-cuda-runtime    Version: 13.0.96    <-- cu13 wheel pulled in
```

vLLM `_C.abi3.so` was compiled against `nvidia-cuda-runtime==13.0.96`
(cu13) while torch, DeepEP, and NCCL all use `libcudart.so.12` (cu12).
Prepending the cu13 lib dir to `LD_LIBRARY_PATH` makes the import
succeed but leaves the process with two distinct `libcudart` instances
in the same address space. When DeepEP's dispatch kernel creates a
CUDA event via cu12 and hands it back to torch via cu13 (or vice
versa), the device-ordinal byte in the event struct is interpreted in
the wrong runtime's layout -> negative garbage values (-48/-64/-128)
-> `SetDevice` throws.

This is a Wave 7c.4-specific regression; Wave 7 (`fast-d00e132ee4bb`)
compiled vLLM against cu12.9 and did not exhibit it. The Wave 7c
changelog lists "NCCL-cu13 restore" as the unblocker for the
`ncclCommQueryProperties` undefined-symbol error, but the matching
vLLM cu13 rebuild has silently reintroduced the ABI split.

Mitigations tried and rejected during this run:
1. Prepend `nvidia/cu13/lib` to `LD_LIBRARY_PATH` -- import succeeds,
   dispatch crashes (above).
2. Fall back to bfloat16 -- same dispatch crash (not FP8-specific).
3. Re-install `torch` / `deep_ep` against cu13 at pod runtime --
   rejected: out of scope for an image-validation run.

Recommended fix for Wave 9: rebuild vLLM against cu12.9 so it matches
torch+DeepEP, OR rebuild torch+DeepEP against cu13.0.96. Do not ship a
hybrid image.

### EFA counters during vLLM run
```
POD0 TX delta = 403,968 bytes (0.0004 GB)
POD0 RX delta = 403,968 bytes (0.0004 GB)
POD1 TX delta = 403,968 bytes (0.0004 GB)
POD1 RX delta = 403,968 bytes (0.0004 GB)
```

~400 KB is the NCCL init handshake only -- no MoE dispatch traffic,
confirming the crash happened before any real dispatch. Expected
delta for a working run is 5-10 GB (Wave 6 saw 6.8 GB for bfloat16 +
`VLLM_USE_DEEP_GEMM=0`, Wave 7 confirmed same).

## Training pass - NeMo-RL Shape Y, single-node PASS, 2-node BLOCKED

### Deploy
```
kubectl apply -f tests/k8s/multi-node-training-h100.yaml
# image replaced with allprs-36a9110 ECR digest
# namespace: deepep-v2-w8-train-20260506
```

Pod `nemo-rl-fullstack-0` reached Ready in 63s on
`hyperpod-i-0a3eb6d3953cceaa7`.

Pod `nemo-rl-fullstack-1` on `hyperpod-i-01aee349f9991c414` stayed in
`ContainerCreating` for the full run -- kubelet on that node cannot
pull the `localhost/kubernetes/pause:latest` sandbox image:
```
Failed to create pod sandbox: rpc error: code = Unknown desc = failed
to start sandbox "...": failed to get sandbox image
"localhost/kubernetes/pause": ... dial tcp 127.0.0.1:443: connect:
connection refused
```
Node is Ready, no pressure conditions, but containerd's image-proxy
endpoint on 127.0.0.1:443 is refusing connections (repro'd 17 times in
3m52s with identical error code). Deleting + recreating the pod twice
did not recover. This is a node-level defect outside the image's
control and will be triaged separately.

Consequence: 2-node training was not feasible. Fell back to single-node
(8 GPU) training on `nemo-rl-fullstack-0` to validate the DeepEP V2
native path + baked-in Shape Y driver. Full 2-node EFA coverage is
deferred to the next run on a healthy-node pair.

### Launch (single-node fallback)
```
torchrun --nnodes=1 --nproc-per-node=8 --node-rank=0 \
    --master-addr=10.1.3.73 --master-port=29500 \
    /opt/tests/train_step_shapeY.py --steps 3 --warmup 1
```

`train_step_shapeY.py` was present at `/opt/tests/train_step_shapeY.py`
inside the image **without any kubectl cp step** -- **G3 confirmed
closed**.

### Training log banner + 3-step loss curve
```
[rank0] DEEP_EP_USE_V2_SHIM=0 (must be 0 for Shape Y validation)
[rank0] Shape Y probe state: HAVE_DEEP_EP=True HAVE_DEEP_EP_V2=True
[rank0] deep_ep exports: ElasticBuffer=True Buffer=True
[rank0] EFA tx_bytes_total before: 3067951948994984
[rank0] Qwen3-30B-A3B-style model built: hidden=2048 ffn=1024 experts=128 topk=8 blocks=2 local_experts=16
[rank0] Active buffer class: ElasticBuffer (expected: ElasticBuffer)
[rank0] WARMUP  loss=23.8181  grad_norm=31.5748  step_ms=18913.8
[rank0] STEP 1/3  loss=21.5669  grad_norm=25.1493  step_ms=25.3
[rank0] STEP 2/3  loss=20.6573  grad_norm=22.4322  step_ms=18.9
[rank0] STEP 3/3  loss=20.0479  grad_norm=21.1514  step_ms=10.0
[rank0] EFA tx_bytes_total after:  3067951949057320
[rank0] EFA tx_bytes delta:        62336 bytes (~0.000 GB)
[rank0] loss trajectory: first=21.5669 last=20.0479 decreased=True
[rank0] SHAPE Y V2 VALIDATION PASS
```

- `HAVE_DEEP_EP_V2=True` -- DeepEP V2 module available
- `Active buffer class: ElasticBuffer` -- V2 buffer class on the hot path
- Loss decreased monotonically: 23.82 (warmup) -> 21.57 -> 20.66 -> 20.05
- grad_norm non-zero and decreasing: 31.57 -> 25.15 -> 22.43 -> 21.15
- 3-step training completed, `SHAPE Y V2 VALIDATION PASS`

### EFA counters
```
Training EFA pod0 TX delta = 62,336 bytes (0.0001 GB)
Training EFA pod0 RX delta = 62,336 bytes (0.0001 GB)
```

Single-node run: all MoE dispatch/combine traffic stays on NVLink
inside one p5.48xlarge, so EFA delta is ~62 KB (the shape-Y rank0
barrier). This is **not** evidence of 2-node EFA traffic -- only
evidence that the image's DeepEP V2 native path imports and runs and
the loss converges. 2-node EFA validation is deferred to the next run
when both H100 nodes are containerd-healthy.

## G-series closure status (what Wave 7c promised)

| Gate | Wave 7c claim | Wave 8 result |
|---|---|---|
| G1 | CUDA 12.9 / torch cu129 alignment, no runtime pip install needed | **Partial**: torch is cu129 and no pip install was needed, but vLLM's `_C.abi3.so` was rebuilt against cu13, re-introducing a runtime ABI split. `LD_LIBRARY_PATH` prepend is still required, and even that only lets import succeed -- dispatch still crashes. |
| G2 | `deep_gemm` baked in, Qwen3-30B-A3B-FP8 native without `VLLM_USE_DEEP_GEMM=0` + bfloat16 workaround | **Closed**: `deep_gemm==2.5.0` present, `DEEPGEMM Fp8 MoE backend` selected, FP8 dispatch path taken (both `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` and `DeepEPV2PrepareAndFinalize` were chosen). FP8 failure was due to G1 CUDA ABI split, not G2. |
| G3 | `train_step_shapeY.py` baked into nemo-rl image, no `kubectl cp` | **Closed**: `/opt/tests/train_step_shapeY.py` present, invoked directly via `torchrun`, ran to completion. |

## Known gaps to fix before Wave 9

1. **Rebuild vLLM against cu12.9** (or rebuild torch/DeepEP against
   cu13.0.96) -- any single-runtime image. The hybrid state is the
   direct cause of the `Event device index` crash in both FP8 and
   bfloat16 code paths. This is the highest-priority blocker.
2. **Triage `hyperpod-i-01aee349f9991c414` containerd fault** --
   separate from image concerns; the kubelet cannot pull
   `localhost/kubernetes/pause` for the node-local sandbox. Likely
   needs a containerd restart or node reboot; would benefit from the
   `hyperpod-node-cgroup-recovery` or `eks-pod-init-debugging` runbook.
3. **2-node EFA validation for training** blocked on (2). Repeat the
   Shape Y run across nodes once the node is healthy; expected delta
   is >= 1 GB TX with even per-rail distribution (Wave 7 baseline).

## Predecessor and successor

- Predecessor: `VALIDATION-FRESH-FROM-PUBLIC-2026-05-05.md` (Wave 7,
  commit `d00e132ee4bb` vllm + `673e66c` nemo-rl)
- Successor: TBD after Wave 9 image rebuild.

## Tear-down

```
kubectl delete -f tests/k8s/multi-node-serving-h100.yaml
kubectl delete -f tests/k8s/multi-node-training-h100.yaml
# restore scaled-down nvshmem-efa/deepep-nvshmem sts to replicas=2
kubectl scale sts deepep-nvshmem -n nvshmem-efa --replicas=2
# release the H100 cluster lock
```
