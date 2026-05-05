# Validation reference

The expected-output contract for the serving driver at
`tests/serve_chat_completion.sh`, against the exact
`docker/Dockerfile` in this repo.

Measured on 2x p5.48xlarge H100 EFA, 2026-04-29. Every line below
is reproducible by a reviewer with the right hardware.

## Preflight (before cluster deploy)

```bash
docker run --rm <your-tag> bash /opt/docker/preflight.sh
```

Expected `stdout` (exact match in spirit; versions may vary by
day-0 build):

```
============================================================
vllm-deepep-v2-efa preflight  2026-05-05T...
host: <container hostname>
============================================================

Check 1/7: aws-ofi-nccl plugin on ld path
  [PASS] libnccl-net plugin on ld path
         NCCL will discover OFI plugin on AWS EFA

Check 2/7: runtime NCCL version >= 2.30.4
  [PASS] NCCL >= 2.30.4 at runtime
         DeepEP V2's ncclTeamTagRail symbol is available

Check 3/7: deep_ep.ElasticBuffer importable (V2 API surface)
  [PASS] deep_ep.ElasticBuffer importable
         V2 API surface present

Check 4/7: vllm.utils.import_utils.has_deep_ep_v2() = True (PR #41183)
  [PASS] has_deep_ep_v2() returns True
         vLLM PR #41183 probe is active

Check 5/7: DeepEPV2PrepareAndFinalize importable (PR #41183 class)
  [PASS] DeepEPV2PrepareAndFinalize importable
         PR #41183 code path wired into vLLM

Check 6/7: /opt/api-shim must not exist (no V1->V2 compat shim)
  [PASS] /opt/api-shim absent
         no shim installed

Check 7/7: DEEP_EP_USE_V2_SHIM env var must be 0
  [PASS] DEEP_EP_USE_V2_SHIM=0
         shim disabled by env

============================================================
Preflight result: 7 PASS, 0 FAIL
============================================================
```

Any FAIL here means the image was not built with all patches
applied or an ABI drift occurred. Re-run `docker/build.sh` from a
clean workspace.

## NCCL initialization log

When `vllm serve` starts, NCCL prints initialization lines with
`NCCL_DEBUG=INFO` (set in the reference k8s manifest). The key
lines to grep for:

```
NCCL INFO NET/OFI Initializing aws-ofi-nccl git-6e504db
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 32 nics)
NCCL INFO NET/OFI Using transport protocol RDMA
NCCL INFO Init COMPLETE
```

The `efa-direct` provider + `RDMA` protocol + 32 NICs confirms the
aws-ofi-nccl plugin is loaded and EFA is the active transport.
`git-6e504db` confirms the source-built plugin (with the
`active_put_signal` bitset fix) is in use, not the
installer-bundled NGC plugin.

Anti-signal to watch for: `NCCL INFO NET/Socket` - means NCCL
fell back to TCP because it couldn't find the OFI plugin. That's
usually an `LD_LIBRARY_PATH` issue. Check the image env with
`docker run --rm <image> bash -c 'echo $LD_LIBRARY_PATH'` - the
`/opt/aws-ofi-nccl/lib` prefix must appear first.

## DeepEP V2 activation log

vLLM's all2all manager selection prints a `Using ...` line early
in engine startup. For this stack the expected line is:

```
INFO [vllm] Using DeepEPV2All2AllManager all2all manager
```

Anti-signal: `Using DeepEPHTAll2AllManager` - means the CLI
argument `--all2all-backend deepep_high_throughput` was picked
instead of `deepep_v2`, or `has_deep_ep_v2()` returned False and
vLLM fell back to V1. Re-check that preflight Check 4 passes.

The `has_deep_ep_v2()` result also appears in the pod-0 log when
DeepEP is first imported:

```
has_deep_ep_v2 = True
```

## Engine startup messages

When both pods have finished booting DP workers, pod-0 log should
show 16 lines like:

```
INFO [engine] Application startup complete.
```

One per DP worker (8 local on leader + 8 on worker = 16). When the
leader has announced all 16, the `/health` endpoint returns 200.

## Chat completion response (end-to-end evidence)

With `max_tokens=24`, a simple `hello` prompt against the model
`Qwen/Qwen3-30B-A3B-FP8` returns a JSON response shaped like:

```json
{"id":"chatcmpl-XXXXXXXX",
 "object":"chat.completion",
 "created":1777486804,
 "model":"Qwen/Qwen3-30B-A3B-FP8",
 "choices":[{"index":0,
             "message":{"role":"assistant",
                        "content":"<think>\nOkay, the user just said \"hello\". I need to respond appropriately. Let me think about how to approach"},
             "finish_reason":"length"}],
 "usage":{"prompt_tokens":9,
          "total_tokens":33,
          "completion_tokens":24}}
```

Verbatim response captured during 2026-04-29 validation on 2x
p5.48xlarge H100 (DP=16 / 8 per pod, EP=16, bf16 activations,
`Qwen/Qwen3-30B-A3B-FP8`):

```json
{"id":"chatcmpl-8789903475a14a93",
 "object":"chat.completion",
 "created":1777486804,
 "model":"Qwen/Qwen3-30B-A3B-FP8",
 "choices":[{"index":0,
             "message":{"role":"assistant",
                        "content":"<think>\nOkay, the user just said \"hello\". I need to respond appropriately. Let me think about how to approach",
                        "refusal":null,"tool_calls":[]},
             "logprobs":null,
             "finish_reason":"length",
             "stop_reason":null}],
 "usage":{"prompt_tokens":9,
          "total_tokens":33,
          "completion_tokens":24},
 "service_tier":null,
 "system_fingerprint":null}
```

### What this proves

- **`finish_reason: "length"`**: the model generated the full 24
  tokens requested; it was not aborted by an internal error.
- **`completion_tokens: 24`**: the MoE forward path dispatched
  and combined tokens successfully across the DP/EP group for 24
  decode steps, each of which routes through DeepEP V2.
- **Non-empty `content` string**: the generation produced
  coherent text ("Okay, the user just said ..."), not random
  tokens or repeated garbage - the expert-weighted combine path
  is numerically correct.
- **`model: Qwen/Qwen3-30B-A3B-FP8`**: the full MoE model
  (128 experts, top-8 routing, FP8 expert weights + bf16
  activations) loaded and served without falling back to a
  non-MoE model.

### Variance expected

- Generated text will differ run-to-run. Only shape is stable.
- `completion_tokens` can be < 24 if the model hits a stop token
  before `max_tokens`. Set a large enough `max_tokens` to force
  `finish_reason: "length"` if you want deterministic length.

## EFA counter deltas

`tests/verify_efa_traffic.sh` snapshots
`/sys/class/infiniband/*/ports/1/hw_counters/tx_bytes` +
`rx_bytes` across all EFA NICs, before and after a serving run,
and computes the delta.

Expected after 10-20 chat completions on the Qwen3 MoE model:

```
Pod 0 tx_bytes delta across 16 NICs: >= 1 GB
Pod 1 tx_bytes delta across 16 NICs: >= 1 GB
PASS: EFA traffic verified real.
```

>= 1 GB per pod confirms:
- MoE dispatch/combine traffic actually went over EFA (not NVLink
  shortcut, not TCP socket fallback)
- Cross-node communication was balanced (both pods saw similar
  TX volume; no skewed expert placement)

### Why only 1 GB and not more?

10-20 completions at 24 tokens each is a tiny workload. Realistic
serving deployments would see hundreds of GB per hour. 1 GB is our
evidence floor - proof that EFA is active, not a performance
target.

For real-world tuning, benchmark with
`docs/DEEPEP-BENCHMARKS.md`'s D+C microbench (H100: ~930 us p50,
H200: ~740 us p50) rather than chat-completion throughput.

## Failure modes and what they mean

| Symptom | Likely cause |
|---|---|
| `NCCL INFO NET/Socket` instead of `NET/OFI` | `LD_LIBRARY_PATH` missing `/opt/aws-ofi-nccl/lib`. The image's `ENV LD_LIBRARY_PATH=...` should prepend it - verify with `docker run --rm <image> bash -c 'echo $LD_LIBRARY_PATH'`. |
| Server hangs at first engine init with `num_allocated_qps = 129` | DeepEP patch 0001 not applied; V2 trying to allocate more QPs than EFA provides. Check with `docker run --rm <image> grep 'num_allocated_qps' /opt/DeepEP/deep_ep/buffers/elastic.py`. |
| `has_deep_ep_v2() = False` in preflight | Either vLLM PR #41183 not applied (fork SHA drift), NCCL < 2.30.4 at runtime (Check 2 should catch this), or `deep_ep.ElasticBuffer` not importable (Check 3 should catch this). |
| `Using DeepEPHTAll2AllManager` in log (V1 path) | Either `--all2all-backend deepep_v2` flag missing from serve command, or vLLM fell back because `has_deep_ep_v2()` returned False. Inspect the vllm-serve command line and run the preflight. |
| `/v1/chat/completions` hangs with no response | Server still in JIT warmup (FlashInfer CUTLASS kernels compile). Tail the leader log for `engine loop is running` messages. If never appears, tail for traceback. |
| Chat completion returns `internal server error` | First check server stdout for `NCCL` or `deep_ep` traceback. A common failure mode is `use_fp8_dispatch=True` with non-FP8 weights - the reference manifest and serving script set `bfloat16` dtype + `use_fp8_dispatch=False`. |
| EFA TX delta ~= 0 | Either pods colocated (single-node run, no cross-node traffic) or `FI_PROVIDER != efa`. Check K8s manifest env vars: `kubectl -n vllm-deepep-v2-efa exec <pod> -- env | grep -E 'FI_|NCCL_GIN|DEEP_EP'`. |
| Step-1 latency > 60s then stable | Expected; first-time kernel JIT compile. Step 2+ should be fast. Enable a PVC-backed JIT cache to avoid re-compiling on every pod restart. |
