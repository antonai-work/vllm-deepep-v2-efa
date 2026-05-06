# Wave 13 CU13 Cluster Validation — PASS

**Date**: 2026-05-06
**Cluster**: HyperPod EKS p5.48xlarge (H100), single-node fallback
**Image**: `058264135704.dkr.ecr.us-east-2.amazonaws.com/vllm-deepep-v2-efa@sha256:410157270f3fd86c8f5cac3cffe88d651ccc3b779f1c30c8d4092d8fa0e3d543`
**GHCR base**: `ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.1-sm90a` (cu13 native)
**Commit**: `f3731a6` on branch `chore/pins-env`
**Namespace**: `deepep-v2-w13-2026-05-06`
**Result**: **PASS — 24 tokens returned, no cu12/cu13 ABI crash**

## Verdict

vLLM served real chat completions on Qwen3-30B-A3B-FP8 via vLLM PR #41183 +
DeepEP V2 PR #612 on the cu13-native base image. The Wave 11
`invalid device ordinal` / `c10::cuda::SetDevice` assertion is gone. The
cu12-vs-cu13 ABI split that caused Wave 8 first-MoE-dispatch crashes is
resolved.

## Evidence

### Cu13 ABI chain confirmed end-to-end

```
$ python3 -c "import torch; print(torch.__version__)"
2.11.0+cu130

$ ldd /opt/DeepEP/deep_ep/_C*.so | grep -E "libcudart|libnccl|libnvshmem"
	libcudart.so.13 => /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13
	libnvshmem_host.so.3 => /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3
	libnccl.so.2 => /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2

$ python3 -c "import deep_ep; print(deep_ep.ElasticBuffer)"
<class 'deep_ep.buffers.elastic.ElasticBuffer'>

$ python3 -c "from vllm.utils.import_utils import has_deep_ep_v2; print(has_deep_ep_v2())"
True
```

Every layer (torch cu130 -> libcudart.so.13 -> libnvshmem_host.so.3 ->
libnccl.so.2 from `nvidia-nccl-cu13==2.30.4`) shares the cu13 ABI.

### In-image preflight

```
8/8 checks PASS
```
(CodeBuild log for build `1cd54f6f-ce3e-4ef6-aa98-3ffc7cffc7cc`).

### vLLM server activation (DeepEP V2 fast path)

```
INFO 05-06 12:15:17 [cuda_communicator.py:175] Using DeepEPV2All2AllManager all2all manager.
INFO 05-06 12:15:46 [fp8.py:590]                Using DeepEPV2PrepareAndFinalize
(Worker_DP0_EP0) [DeepEP] EFA detected: capping num_allocated_qps 129 -> 2
(Worker_DP1_EP1) [DeepEP] EFA detected: capping num_allocated_qps 129 -> 2
(Worker_DP2_EP2) [DeepEP] EFA detected: capping num_allocated_qps 129 -> 2
... [all 8 DP ranks]
INFO:     Application startup complete. (x8 ApiServers)
```

PR #41183 (`DeepEPV2All2AllManager`, `DeepEPV2PrepareAndFinalize`) is
active and all 8 DP workers see EFA and apply PR #612's auto-QP cap. No
shim, no `invalid device ordinal` crash, no `c10::DeviceIndex` assertion.

### Chat completion (real tokens, no bfloat16 fallback)

Request:
```
POST http://localhost:8000/v1/chat/completions
{"model":"Qwen/Qwen3-30B-A3B-FP8",
 "messages":[{"role":"user","content":"Say hello in exactly three different languages."}],
 "max_tokens":24}
```

Response:
```json
{"id":"chatcmpl-93ca9081c28e2c17",
 "model":"Qwen/Qwen3-30B-A3B-FP8",
 "choices":[{"message":{"role":"assistant",
              "content":"<think>\nOkay, the user wants me to say hello in exactly three different languages. Let me think about which languages to"},
             "finish_reason":"length"}],
 "usage":{"prompt_tokens":16,"total_tokens":40,"completion_tokens":24}}
```

Config at serve time: `dtype=torch.bfloat16, quantization=fp8` (native
Qwen3-FP8 with bfloat16 activations — the FP8 MoE path is active, no
`VLLM_USE_DEEP_GEMM=0` fallback).

### EFA counters (non-zero across 32 rails)

Snapshot delta after chat completion:
```
TOTAL TX DELTA: 62336 bytes across 32 rails
Per-rail range: 1272 - 2632 bytes
```

EFA counters incremented on every rail, confirming DeepEP V2 reached
the EFA fabric. Small absolute number is expected because on a single
8-GPU node with DP=8 intra-node, MoE token routing goes NVLink-preferred
and only synchronization/handshake traffic crosses EFA. A 2-pod DP=16
run would push GB-scale traffic.

## Deviation from brief

Target was 2-pod `deepep-v2-w13-2026-05-06` StatefulSet with DP=16 EP=16.
At deploy time the second node (`hyperpod-i-01aee349f9991c414`) was
stuck in a containerd fault unrelated to this image:

```
FailedCreatePodSandBox: failed to pull image "localhost/kubernetes/pause":
  failed to resolve reference "localhost/kubernetes/pause:latest":
  failed to do request: Head "https://localhost/v2/kubernetes/pause/manifests/latest":
  dial tcp 127.0.0.1:443: connect: connection refused
```

The other node (`hyperpod-i-0a3eb6d3953cceaa7`) pulled the image and
serviced the request fine. Per brief, fell back to single-pod DP=8
intra-node on the working node. The cu13 ABI validation is independent of
pod count — the crash site was first MoE dispatch which completed here
across 8 DeepEPV2 workers.

## The cu12-vs-cu13 saga — what was learned

| Wave | Outcome | Lesson |
|---|---|---|
| **Wave 8** | FAIL: `invalid device ordinal` at first MoE dispatch | Mixed cu12 base + cu13 realign torch caused ABI split |
| **Wave 9a/b/c** | cu12-unification exploration | Couldn't unify because vLLM precompiled kernels + torch 2.11 wheel are cu13-native |
| **Wave 11** | Cluster-test exposed torch/vllm._C cu13 wheel vs. cu12 base | The ABI story was: you can't force cu12 when your wheels are cu13 |
| **Wave 10-12** | cu13 completion: rebuild base on CUDA 13 | Every layer (base, torch, wheels) must share libcudart.so.13 |
| **Wave 13** | **VERIFIED** — `Using DeepEPV2PrepareAndFinalize`, 24 tokens, no crash | When base image matches wheel ABI, the stack runs clean |

The diagnostic insight: in a python-wheel-heavy stack (torch + vllm precompiled +
nvidia-nccl-cuX + nvidia-nvshmem-cuX + DeepEP built from source), you can't
choose your CUDA ABI independently of what your heaviest wheel ships. torch
cu130 + vLLM precompiled kernels require libcudart.so.13 at runtime; the only
sustainable story is to match the base image toolkit to the wheel ABI.

## Wave 13 Dockerfile change summary

Two commits on `chore/pins-env`:

1. `635df76`: bump `BASE_IMAGE_FAST=v0.2.1-sm90a`, NCCL pin to cu13, torch to
   cu130 in vanilla path, NVSHMEM to cu13, remove cu129 torch realign in release.
2. `f3731a6`: re-upgrade `nvidia-nvshmem-cu13>=3.6.0` after torch force-reinstall
   (torch 2.11 pins `nvidia-nvshmem-cu13==3.4.5` which is missing the
   `nvshmem_selected_device_transport, version NVSHMEM` symbol that the base's
   3.6.5 deep_ep `_C.so` was linked against).

Companion repo commit on `antonai-work/nemo-rl-deepep-v2-efa`:
`4ccc605` — same base image bump + NVSHMEM guard + bash-quoting fix on
`pins.env` (`NVIDIA_NCCL_PIN` unquoted `>=` caused `NAME=VALUE>=2.30.4`
to parse as assignment + redirect, silently truncating the version spec).

## Reproduction

```bash
# From the repo root on branch chore/pins-env @ commit f3731a6:
aws codebuild start-build --project-name vllm-deepep-v2-efa-build \
    --region us-east-2 --source-version chore/pins-env
# -> expect "8/8 checks PASS" in the build log.

# Then on a cluster with H100 p5.48xlarge node:
kubectl apply -f tests/k8s/multi-node-serving-h100.yaml
# (set image to the ECR digest above)

# Run chat completion:
POD=$(kubectl -n vllm-deepep-v2-efa get pod -o name | head -1)
kubectl -n vllm-deepep-v2-efa exec -it $POD -- \
  env ROLE=leader MASTER_IP=$(kubectl get pod ... -o jsonpath='{.status.podIP}') \
  bash /opt/tests/serve_chat_completion.sh &
# Wait for /health 200, then:
kubectl -n vllm-deepep-v2-efa exec $POD -- \
  curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"Qwen/Qwen3-30B-A3B-FP8","messages":[{"role":"user","content":"hi"}],"max_tokens":24}'
```

## Cross-reference

- Companion nemo-rl training validation:
  `antonai-work/nemo-rl-deepep-v2-efa/docs/VALIDATION-WAVE13-CU13-TRAINING-PASS.md`
- Wave 11 failure it supersedes:
  `antonai-work/vllm-deepep-v2-efa/docs/VALIDATION-WAVE8-FRESH-2026-05-06.md`
  (cu12 unification attempt that Wave 11 cluster-test invalidated)
