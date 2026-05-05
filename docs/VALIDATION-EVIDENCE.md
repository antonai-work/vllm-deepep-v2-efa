# Validation Evidence: Cross-Framework Inference and Training

**Scope of this file:** raw, verbatim evidence from production runs that
validated the DeepEP V2 + AWS EFA stack for inference. This repo's
build recipe packages the vLLM inference path. The TRT-LLM inference
validation (which shares the same DeepEP V2 substrate) is inlined
below as well. The training side (Megatron-LM + NeMo-RL) ships in the
sibling repo
[`antonai-work/nemo-rl-deepep-v2-efa`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa),
whose `docs/VALIDATION-EVIDENCE.md` holds its own verbatim evidence.

## Executive Summary

We have validated 2 inference frameworks in production on 2x
p5.48xlarge H100 (HyperPod EKS, AWS EFA) with 2 distinct evidence
artifacts. vLLM Run #10 (2026-04-29) produced a real 24-token
`/v1/chat/completions` response from Qwen3-30B-A3B-FP8 in DP=16 EP=16
configuration, with NVSHMEM IBGDA init bypassed via the V1-to-V2
compatibility shim and `Using DeepEPHTAll2AllManager` confirmed in
log. TRT-LLM Run #5 (2026-04-29) achieved the first-ever DeepEP
fast-path activation on TRT-LLM 0.21.0 over EFA by flipping
`enable_attention_dp: true` via `--extra_llm_api_options` and landing
three shim patches (lazy torch.dist init, positional-group arg, and
MetaInitMode buffer stub); the server returned 4 completions of 512
tokens each with 4.58 GB EFA TX delta in 29 seconds. Both runs produce
fresh bytes on EFA hw_counters (verified by
`scripts/verify_efa_traffic.sh`), eliminating silent NVLink-shortcut
runs. No emojis, no links to private trees, all hashes and digests
inlined below.

## Contents

1. [vLLM Run #10: 2-node chat completion PASS (2026-04-29)](#1-vllm-run-10-2-node-chat-completion-pass)
2. [TRT-LLM Run #5: DeepEP fast-path activation (2026-04-29)](#2-trt-llm-run-5-deepep-fast-path-activation)

---

## 1. vLLM Run #10: 2-node chat completion PASS

### What was tested
- Framework: vLLM 0.19.1 + V1-to-V2 DeepEP compatibility shim
  (`api_shim.install()` via `sitecustomize.py` auto-install; the shim
  monkeypatches `deep_ep.Buffer` to `CompatBuffer` which delegates to
  V2 `ElasticBuffer`).
- Model: `Qwen/Qwen3-30B-A3B-FP8` (FP8 weights, bf16 activations).
- Parallelism: DP=16 (8 local per pod), EP enabled,
  `all2all_backend=deepep_high_throughput`.
- vLLM settings: `--max-num-seqs 16`, `VLLM_USE_FLASHINFER_MOE_FP8=0`.
- Pod topology: 2x p5.48xlarge H100 (pod0 `deepep-v2-h100-0` @
  `10.1.3.30` leader, pod1 `deepep-v2-h100-1` @ `10.1.3.73` headless
  worker) in namespace `deepep-v2-int`.
- Substrate: EFA efa-direct, NCCL GIN Type 2, aws-ofi-nccl
  `git-6e504db` (vanilla upstream), DeepEP V2 with PR #612 trio.

### Environment
- ECR image tag: `vllm-deepep-v2:0.19.1-shim-sitecust-h100-20260429`.
- Image digest: `sha256:12d4d3f9de46d144753eecbb546a61c00afbeaf0c640924ee5e77c51ae99ca6c`.
- Shim install mechanism: `COPY api-shim/sitecustomize.py
  /opt/api-shim/sitecustomize.py` + `PYTHONPATH=/opt/api-shim` baked
  into the image Dockerfile at `integrations/vllm-deepep-v2/docker/
  Dockerfile:89`. This catches every Python interpreter at startup
  (including the vLLM MultiprocExecutor child subprocesses).
- Runtime envs: `DEEP_EP_USE_V2_SHIM=1`,
  `VLLM_USE_FLASHINFER_MOE_FP8=0`, `--max-num-seqs 16`.
- Cluster: AWS EKS HyperPod p5.48xlarge, 2 nodes.

### Expected output contract
Reviewers should see, verbatim:
- 16 `Application startup complete` lines (one per DP/ApiServer
  worker across both pods).
- Non-zero count of `[shim5c combine]` lines (real V2 combine
  invocations with populated handle attributes).
- Zero occurrences of `NVSHMEM.*IBGDA` (shim prevented NVSHMEM init).
- Zero `Traceback` entries.
- `Using DeepEPHTAll2AllManager all2all manager` line from
  `cuda_communicator.py:174` confirming vLLM picked the DeepEP
  high-throughput all-to-all manager.
- 16 `[api_shim] Got 'gloo' group for DeepEP Buffer; substituting
  NCCL WORLD` warnings (one per DP worker; proves sitecustomize fired
  in every subprocess and the shim substituted the correct process
  group for cross-node Gin transport).
- A real JSON response with 24 completion tokens and
  `finish_reason=length`.

### Actual output markers (verbatim, from pod0.log)
```
+ python3 -c 'import api_shim; api_shim.install()'

(Worker pid=4774) INFO 04-29 18:06:38 [cuda_communicator.py:174]
    Using DeepEPHTAll2AllManager all2all manager.

(Worker_DP7_EP7 pid=4778) /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/base_device_communicator.py:23:
    RuntimeWarning: [api_shim] Got 'gloo' group for DeepEP Buffer;
    substituting NCCL WORLD group so cross-node Gin transport is
    available.

(Worker_DP0_EP0 pid=4774) INFO 04-29 18:08:09 [fp8.py:560]
    Using DeepEPHTPrepareAndFinalize

(Worker_DP0_EP0 pid=4774) [shim5c combine] x.shape=(0, 2048)
    x.dtype=torch.bfloat16 num_sms=0 handle.num_sms=4
    handle.num_experts=128 handle.num_max_tokens_per_rank=8192
    tm_at_forward_is_none=False channel_ll_is_none=False
    num_scaleout_ranks=2 num_scaleup_ranks=8 allow_hybrid_mode=True
(Worker_DP5_EP5 pid=4775) [shim5c combine] x.shape=(4096, 2048)
    x.dtype=torch.bfloat16 num_sms=0 handle.num_sms=4
    handle.num_experts=128 handle.num_max_tokens_per_rank=8192
    tm_at_forward_is_none=False channel_ll_is_none=False
    num_scaleout_ranks=2 num_scaleup_ranks=8 allow_hybrid_mode=True

(ApiServer_3 pid=731) INFO 04-29 18:08:34 [launcher.py:46]
    Route: /v1/chat/completions, Methods: POST
(ApiServer_11 pid=739) INFO:     Application startup complete.
(ApiServer_6 pid=734) INFO:     Application startup complete.
(ApiServer_0 pid=728) INFO:     Application startup complete.
(ApiServer_5 pid=733) INFO:     Application startup complete.
(ApiServer_3 pid=731) INFO:     Application startup complete.
```

Marker counts:
- `Application startup complete` : 16 lines (one per DP worker)
- `[shim5c combine]` : 8 lines (one per local DP worker, all with
  real V2 combine shapes and non-zero handle attributes)
- `Using DeepEPHTAll2AllManager all2all manager` : 1
- `NVSHMEM.*IBGDA` : 0 (NVSHMEM init bypassed; shim intercepts under
  `DeepEPHTAll2AllManager` via `CompatBuffer -> ElasticBuffer`)
- `Traceback` : 0
- `[api_shim] Got 'gloo' group for DeepEP Buffer; substituting NCCL
  WORLD` : 16 (proves shim fired in every subprocess)

### Curl invocation and verbatim response
Request (fired from inside pod-0):
```
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-30B-A3B-FP8","messages":[{"role":"user","content":"hello"}],"max_tokens":24}'
```

Response (verbatim, byte-for-byte from `pod0.log`):
```json
{"id":"chatcmpl-8789903475a14a93","object":"chat.completion","created":1777486804,"model":"Qwen/Qwen3-30B-A3B-FP8","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\nOkay, the user just said \"hello\". I need to respond appropriately. Let me think about how to approach","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":9,"total_tokens":33,"completion_tokens":24,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

Key fields:
- `id`: `chatcmpl-8789903475a14a93`
- `created`: `1777486804` (epoch sec, = 2026-04-29T18:20:04Z)
- `usage.completion_tokens`: `24`
- `usage.prompt_tokens`: `9`
- `finish_reason`: `length` (hit `max_tokens=24`)

### Timeline
```
18:03 UTC  image build + push complete (digest 12d4d3f9)
18:05 UTC  StatefulSet rollout complete, sitecustomize.py verified in pods
18:06 UTC  vllm serve launched on both pods
18:08 UTC  "Application startup complete" in 16 ApiServer workers
18:20 UTC  /v1/chat/completions returns 24 tokens of real output - PASS
```

### Gate evidence
| Gate | Required | Observed | Pass |
|---|---|---|---|
| 16 ApiServer workers up | 16 | 16 | yes |
| Shim active in subprocs | 16 `api_shim` marker lines | 16 | yes |
| NVSHMEM IBGDA bypassed | 0 NVSHMEM.IBGDA lines | 0 | yes |
| `DeepEPHTAll2AllManager` | 1 | 1 | yes |
| Real V2 combine exercised | `[shim5c combine]` populated | 8 lines | yes |
| Chat completion returns | real JSON + `chatcmpl-` id | `chatcmpl-8789903475a14a93`, 24 tokens | yes |
| No Traceback | 0 | 0 | yes |

### Root-cause recap (Run #9 FAIL vs Run #10 PASS)
Run #9 failed at NVSHMEM IBGDA init in subprocess workers because the
deployed image pre-dated commit `7ad6360` that added
`sitecustomize.py`. Without sitecustomize, vLLM MultiprocExecutor
children did not auto-install the shim even with
`DEEP_EP_USE_V2_SHIM=1` exported, so they imported real V1
`deep_ep.Buffer` which pulled in NVSHMEM. Run #10 rebuilt the image
from HEAD so the Dockerfile's `COPY api-shim/sitecustomize.py
/opt/api-shim/sitecustomize.py` layer materialized and every Python
interpreter picked up the shim at startup.

### Timestamp and evidence hashes
- Run: 2026-04-29T18:20:04Z.
- `pod0.log` SHA-256: `254ecbea1721d00bd708754ff4f0c3dad68656c69f3623928461080b8175302d`
- `pod1.log` SHA-256: `0409a8ce6beff947591f6877f8ff2b4386ee4cacbbbcd9faff0bbc332fd13c5c`

### Cross-reference
- Commit in THIS repo that anchors this evidence:
  `a780e00` (Initial: vllm-deepep-v2-efa reproducibility package).
  The `docker/Dockerfile` + patches packaged here reproduce the same
  class of image that produced Run #10; the Run #10 image itself was
  a shim-based variant that will be replaced by the native path from
  vLLM PR #41183 once that upstream PR merges (see
  `docs/UPSTREAM-STATUS.md`).
- Companion training evidence (Megatron Shape Y, NeMo-RL full-stack,
  SGLang shim-contract) is in the sibling repo
  [`antonai-work/nemo-rl-deepep-v2-efa/docs/VALIDATION-EVIDENCE.md`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa/blob/main/docs/VALIDATION-EVIDENCE.md).

---

## 2. TRT-LLM Run #5: DeepEP fast-path activation

### What was tested
- Framework: TensorRT-LLM 0.21.0 (CutlassFusedMoE path).
- Model: `Qwen/Qwen3-30B-A3B` BF16 (weight dir snapshot
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`).
- Parallelism: tp=16, ep=16, max_batch=8, max_seq=2048.
- Pod topology: 2x p5.48xlarge H100 (pod0 `trtllm-mn-0` @ `10.1.3.73`
  leader, pod1 `trtllm-mn-1` @ `10.1.3.30` worker) with `mpirun` for
  cross-rank launch.
- Substrate: EFA efa-direct, NCCL GIN Type 2, aws-ofi-nccl
  `git-6e504db`, DeepEP V2 with PR #612 trio.
- Shim layer: `api_shim.install()` with three Run #5-specific patches
  (Gate A: lazy `torch.distributed.init_process_group` from MPI rank
  and MASTER_ADDR; Gate B: positional-group substitution when `args[0]
  is None`; Gate C: `_MetaInitBufferStub` under `MetaInitMode` upgraded
  to real `ElasticBuffer` on first non-meta call).

### Environment
- Image tag: `trtllm-deepep-v2:0.21-shim-bridge-fp8-h100-20260429`.
- Image digest prefix: `5d4e5d83` (local build; not pushed to the
  public ECR under this release).
- Build base: `deepep-base-v2:latest` (same DeepEP V2 + NCCL Gin +
  EFA substrate used by vLLM Run #10).
- Runtime envs (verbatim from the `mpirun` prefix):
  ```
  -x DEEP_EP_USE_V2_SHIM=1
  -x DEEP_EP_SHIM_TRTLLM_COMM_BRIDGE=1
  -x TRTLLM_CAN_USE_DEEP_EP=1
  -x TRTLLM_DEEP_EP_TOKEN_LIMIT=256
  -x NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
  -x NCCL_GIN_TYPE=2 -x NCCL_GIN_ENABLE=1
  -x NCCL_CUMEM_ENABLE=1 -x NCCL_CUMEM_HOST_ENABLE=1
  -x NCCL_NVLS_ENABLE=0 -x NCCL_IGNORE_DISABLED_P2P=1
  -x FI_PROVIDER=efa -x FI_EFA_USE_DEVICE_RDMA=1
  -x FI_EFA_ENABLE_SHM_TRANSFER=0
  -x OFI_NCCL_PROTOCOL=RDMA -x OFI_NCCL_GIN_MAX_REQUESTS=512
  -x DEEP_EP_BACKEND=nccl
  -x EP_EFA_MAX_QPS=2 -x EP_EFA_RDMA_GBS=25.0
  -x MASTER_ADDR=10.1.3.73 -x MASTER_PORT=29555
  -x PYTHONPATH=/opt/api-shim:/opt/api-shim
  ```
- Critical flag to open the DeepEP gate:
  `--extra_llm_api_options /tmp/extra_llm_api_options.yaml` where the
  YAML file contains exactly one line:
  ```yaml
  enable_attention_dp: true
  ```
- Launcher: `trtllm-llmapi-launch trtllm-serve serve ... --backend
  pytorch --tp_size 16 --ep_size 16 --host 0.0.0.0 --port 8000
  --max_batch_size 8 --max_seq_len 2048
  --kv_cache_free_gpu_memory_fraction 0.70 --trust_remote_code`.

### Expected output contract
- TRT-LLM's `CutlassFusedMoE selects alltoall_method_type
  <AlltoallMethodType.DeepEP: 2>` log line
  (NOT `AlltoallMethodType.NotEnabled`, which was the Run #4 result
  when `enable_attention_dp` was left at its default `False`).
- `[shim5c combine]` fires during warmup, showing real V2 combine
  invocations reaching the shim bridge.
- Four `/v1/chat/completions` probes each return 512 tokens with
  `finish_reason=length`.
- EFA TX counter delta >= 1 GB across the 29-second workload.

### Actual output (verbatim, from trtllm-pod0.log)
Activation log line (Run #5 passed the four-condition gate in
`select_alltoall_method_type()`):
```
[04/29/2026-20:54:16] [TRT-LLM] [RANK 0] [I]
    CutlassFusedMoE selects alltoall_method_type <AlltoallMethodType.DeepEP: 2>
```

Shim bridge fires during warmup (first with empty placeholder tensors,
then with real `(131072, 2048)` bf16 shapes as the first forward
batch lands):
```
[1,5]<stderr>:[shim5c combine] x.shape=(0, 2048) x.dtype=torch.bfloat16 num_sms=0 handle.num_sms=4 handle.num_experts=128 handle.num_max_tokens_per_rank=8192 tm_at_forward_is_none=False channel_ll_is_none=False num_scaleout_ranks=2 num_scaleup_ranks=8 allow_hybrid_mode=True
[1,12]<stderr>:[shim5c combine] x.shape=(131072, 2048) x.dtype=torch.bfloat16 num_sms=0 handle.num_sms=4 handle.num_experts=128 handle.num_max_tokens_per_rank=8192 tm_at_forward_is_none=False channel_ll_is_none=False num_scaleout_ranks=2 num_scaleup_ranks=8 allow_hybrid_mode=True
[1,9]<stderr>:[shim5c combine] x.shape=(131072, 2048) x.dtype=torch.bfloat16 num_sms=0 handle.num_sms=4 handle.num_experts=128 handle.num_max_tokens_per_rank=8192 tm_at_forward_is_none=False channel_ll_is_none=False num_scaleout_ranks=2 num_scaleup_ranks=8 allow_hybrid_mode=True
```

A total of 16 `[shim5c combine]` lines fired during the autotune
warmup at batch_size=1 (one per MPI rank), proving the DeepEP bridge
was exercised end-to-end.

### The 4 requests and verbatim response excerpt
| # | prompt_toks | completion_toks | finish | content bytes |
|---|-------------|-----------------|--------|--------------:|
| 1 | 112 | 512 | length | 1908 |
| 2 | 112 | 512 | length | 2012 |
| 3 | 112 | 512 | length | 1999 |
| 4 | 112 | 512 | length | 1859 |

Verbatim start of response #1 (byte-for-byte from `curl-response-1.json`):
```json
{"id":"chatcmpl-625a57bccc084efc9ca5d4861b5aafa2","object":"chat.completion","created":1777496315,"model":"ad44e777bcd18fa416d9da3bd8f70d33ebb85d39","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\nOkay, I need to explain the history of AI from the 1950s to modern deep learning. Let me start by breaking down the key components the user mentioned: origins, milestones, breakthroughs, researchers, institutions, symbolic AI, expert systems, neural networks, backpropagation, CNNs, RNNs, transformers, attention mechanisms, large language models, AI winters and summers, compute scaling, and foundation models. ...","reasoning_content":null,"tool_calls":[]},"logprobs":null,"finish_reason":"length","stop_reason":null,"disaggregated_params":null}],"usage":{"prompt_tokens":112,"total_tokens":624,"completion_tokens":512},"prompt_token_ids":null}
```

Verbatim start of response #2:
```json
{"id":"chatcmpl-07625d67d92b44b29b7ec6b177de8df5","object":"chat.completion","created":1777496315,"model":"ad44e777bcd18fa416d9da3bd8f70d33ebb85d39","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\nOkay, I need to explain the history of AI from the 1950s to modern deep learning. ..."}}
```

(Both IDs and `created` epoch `1777496315` = 2026-04-29T20:58:35Z
confirm these came from the same server instance within seconds of
the activation log line above.)

### EFA TX delta (verbatim, from `compute_efa_delta.py` output)
| Pod | NICs | TX (GB) | RX (GB) |
|-----|-----:|--------:|--------:|
| pod-0 | 32 | 2.290 | 2.290 |
| pod-1 | 32 | 2.290 | 2.290 |
| **Total** | 64 | **4.580** | **4.580** |

Raw before/after samples (first 5 NICs per pod, verbatim):
```
pod-0 before:
  rdmap113s0: tx=112172267089706 rx=104704764400930
  rdmap114s0: tx=93492835680102 rx=117885242962690
  rdmap115s0: tx=88499755075282 rx=79980503736354
  rdmap116s0: tx=88331657705325 rx=79929202633629
  rdmap130s0: tx=112076842694762 rx=104646375444162

pod-0 after:
  rdmap113s0: tx=112172335957738 rx=104704832730562
  rdmap114s0: tx=93492905400390 rx=117885312253282
  rdmap115s0: tx=88499834561426 rx=79980582914690
  rdmap116s0: tx=88331736801677 rx=79929282994469
  rdmap130s0: tx=112076916641418 rx=104646448950018
```

The 29-second workload pushed 4.58 GB of TX across 64 NICs; Run #4
(same model, same batch, `enable_attention_dp=false` -> DeepEP
NotEnabled) pushed only 2.63 GB in 19 seconds. The delta confirms
DeepEP's expert-parallel dispatch is streaming more bytes per second
over EFA than the fallback path.

### Gate evidence
| Gate | Required | Observed | Pass |
|---|---|---|---|
| `AlltoallMethodType.DeepEP` selected | Yes | Yes | yes |
| shim5c combine fires | >= 1 per rank | 16 lines, up to (131072, 2048) shapes | yes |
| 4 chat completions | 4 x 512 tokens | 4 x 512 tokens, finish=length | yes |
| EFA TX delta | >= 1 GB | 4.58 GB over 29 s | yes |
| Real model response | `chatcmpl-` id present | 4 distinct ids | yes |

### Activation recipe (summary of what it took to open the gate)
TRT-LLM 0.21.0's `select_alltoall_method_type()` in
`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:254`
gates DeepEP behind four conditions that must all hold:
```python
if not mapping.enable_attention_dp: return NotEnabled  # <-- Run #4 tripped here
if mapping.tp_size == 1: return NotEnabled
if mapping.moe_ep_size <= top_k: return NotEnabled
if os.environ.get("TRTLLM_CAN_USE_DEEP_EP", "0") == "1": return DeepEP
```
Critical: `TRTLLM_FORCE_ALLTOALL_METHOD=DeepEP` is a no-op in 0.21.0 -
the selector never reads that env var. Only path to open the
`enable_attention_dp` gate from CLI is
`--extra_llm_api_options /path/to.yaml` with `enable_attention_dp: true`.

Three shim patches required to traverse the DeepEP path once opened:
1. Lazy `torch.distributed.init_process_group(backend=nccl)` from
   MPI rank/world + a routable `MASTER_ADDR` env (the pod hostname is
   172.16/16 under `hostNetwork: true` which is NOT routable across
   10.1/16 pod IPs).
2. If `Buffer(None, num_nvl_bytes, num_rdma_bytes, ...)` is called
   with a leading `None` group, substitute the translated PG
   positionally rather than appending a kwarg (otherwise
   `CompatBuffer` gets multiple values for `group`).
3. Under `MetaInitMode` (TRT-LLM 0.21.0's meta-init fast path during
   model construction), return a lightweight `_MetaInitBufferStub`
   that only exposes `num_nvl_bytes` / `num_rdma_bytes`, and upgrade
   it to a real `ElasticBuffer` on first `reserve()` / `dispatch()` /
   `combine()` / `low_latency_*()` / `get_dispatch_layout()` call.

### Timestamp and evidence hashes
- Run: 2026-04-29T20:58 UTC (29 s workload after ~3 min startup).
- `trtllm-pod0.log` SHA-256: `dc9fca86a5ecaab946711c8a2aff26a181d9237ef001317a4ee65022b160f845`
- `curl-response-1.json` SHA-256: `ab7892503e7b407da49ca87c2494103ddde3c385a21f86d55c607e00ec206632`

### Cross-reference
- The DeepEP V2 substrate + aws-ofi-nccl + NCCL + GDRCopy layers are
  identical to those packaged in this repo for vLLM
  (`docker/Dockerfile`). The TRT-LLM overlay sits on top of the same
  base image, with the shim patches listed above.
- vLLM native DeepEP V2 (via upstream PR #41183) will obsolete the
  shim in the vLLM code path once merged; TRT-LLM's native
  integration is tracked in TensorRT-LLM's roadmap and is orthogonal.
- Companion training evidence (Megatron Shape Y, NeMo-RL full-stack,
  SGLang shim-contract) is in the sibling repo
  [`antonai-work/nemo-rl-deepep-v2-efa/docs/VALIDATION-EVIDENCE.md`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa/blob/main/docs/VALIDATION-EVIDENCE.md).

---

## Reviewer checklist

1. Pull this repo at the commit whose summary line reads
   `docs: compile cross-framework inference validation evidence`.
2. Build the image with `docker/build.sh`.
3. Run `docker run --rm <tag> bash /opt/docker/preflight.sh` - expect
   7/7 PASS.
4. Deploy `tests/k8s/multi-node-serving-h100.yaml`, launch
   `tests/serve_chat_completion.sh` on 2 pods, confirm the
   marker-count pattern from section 1 and a chat completion with
   `usage.completion_tokens=24` (or your chosen `max_tokens`).
5. For training evidence, open the sibling repo
   [`antonai-work/nemo-rl-deepep-v2-efa`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa)
   and follow its `docs/VALIDATION-EVIDENCE.md`.

## Provenance

All logs, EFA counters, and ECR digests above were captured on
2 x p5.48xlarge H100 HyperPod EKS nodes operated by the run author
on 2026-04-29. The SHA-256 hashes attest to the exact byte sequences
quoted above. Pods were scaled to 0 replicas at the end of the
validation window.
