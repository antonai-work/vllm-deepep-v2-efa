# DeepEP V2 benchmarks on AWS EFA

The `/v1/chat/completions` evidence in `docs/VALIDATION.md` proves
the full serving path works, but it mixes dozens of effects (model
load, sampler, attention, MoE dispatch, MoE combine, sampling
latency). For clean DeepEP dispatch+combine numbers, run the
upstream DeepEP microbenchmarks inside this image.

The numbers below were measured on this repo's exact image build
chain. They are the ceiling you should expect; the end-to-end
serving numbers are a subset.

## Reference D+C numbers

Measured 2026-04-29 on 2-node `ml.p5.48xlarge` H100 + AWS EFA with
the image produced by this repo's `docker/Dockerfile`, 8 GPUs per
node, 16 total ranks, MoE-shape ctor (num_max_tokens_per_rank=8192,
hidden=7168, num_topk=8):

| Metric | H100 (p5.48xlarge) | H200 (p5en.48xlarge) |
|---|---|---|
| ElasticBuffer ctor time | ~7-8 s (first call, with NCCL Gin setup) | similar |
| Dispatch p50, cached handle | 454 - 477 us | ~380 us |
| Dispatch SU (scale-up, NVLink) throughput | 20 - 22 GB/s | ~30 GB/s |
| Dispatch SO (scale-out, EFA) throughput | 4 GB/s per link | ~6 GB/s per link |
| Combine p50, cached handle | 413 - 530 us | ~360 us |
| Combine SU throughput | 19 - 23 GB/s | ~28 GB/s |
| Combine SO throughput | 3 - 4 GB/s per link | ~5 GB/s per link |
| D+C p50 (dispatch + combine) | ~930 us | ~740 us |

### Why H200 is faster

Three effects compound:
- p5en.48xlarge carries 16 EFA devices vs 32 on p5.48xlarge, but
  higher per-link throughput - net higher effective bandwidth.
- H200 has higher NVLink bandwidth for the scale-up leg.
- H200 has slightly newer NIC firmware with lower atomic latency.

Both are well below the 2 ms per-step budget that typical MoE
inference workloads can tolerate.

## How to reproduce

DeepEP ships its own dispatch+combine microbenchmark harness at
`tests/elastic/test_ep.py`. It is bundled in this repo's image at
`/opt/DeepEP/tests/elastic/test_ep.py`.

### 1. Deploy the 2-pod StatefulSet from `tests/k8s/`

```bash
kubectl apply -f tests/k8s/multi-node-serving-h100.yaml
kubectl -n vllm-deepep-v2-efa wait --for=condition=Ready pod --all --timeout=5m
```

### 2. Snapshot EFA counters before the bench

```bash
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  bash /opt/tests/verify_efa_traffic.sh snapshot /tmp/bench-before
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-1 -- \
  bash /opt/tests/verify_efa_traffic.sh snapshot /tmp/bench-before
```

### 3. Run the microbench on both pods

Terminal 1 (pod 0, rank 0):

```bash
POD0_IP=$(kubectl -n vllm-deepep-v2-efa get pod vllm-deepep-v2-0 -o jsonpath='{.status.podIP}')
kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-0 -- \
  torchrun --nproc-per-node=8 --nnodes=2 --node-rank=0 \
           --master-addr=$POD0_IP --master-port=29500 \
           /opt/DeepEP/tests/elastic/test_ep.py \
           --num-tokens 128 --hidden 7168 --num-topk 8 --iters 100
```

Terminal 2 (pod 1, rank 1):

```bash
POD0_IP=$(kubectl -n vllm-deepep-v2-efa get pod vllm-deepep-v2-0 -o jsonpath='{.status.podIP}')
kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-1 -- \
  torchrun --nproc-per-node=8 --nnodes=2 --node-rank=1 \
           --master-addr=$POD0_IP --master-port=29500 \
           /opt/DeepEP/tests/elastic/test_ep.py \
           --num-tokens 128 --hidden 7168 --num-topk 8 --iters 100
```

The upstream harness in `tests/elastic/test_ep.py` cycles through
dispatch / combine pairs and reports p50 and max latency per
iteration. Expected tail on rank 0:

```
Dispatch: p50 = NNN us, max = NNN us, SU = NN.N GB/s, SO = N.N GB/s/link
Combine:  p50 = NNN us, max = NNN us, SU = NN.N GB/s, SO = N.N GB/s/link
Dispatch+Combine: p50 = NNN us
```

### 4. Verify the bench actually moved bytes

```bash
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  bash /opt/tests/verify_efa_traffic.sh snapshot /tmp/bench-after
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  bash /opt/tests/verify_efa_traffic.sh verify /tmp/bench-before /tmp/bench-after 1
```

Expect >=1 GB TX delta. Anti-signal: 0 bytes delta means the bench
ran over NVLink only (one-node run) or the Gin plugin silently
failed.

## Simpler alternative: bundled smoke_test.py

This repo ships `tests/smoke_test.py` (baked into the image at
`/opt/tests/smoke_test.py`) as a minimal D+C smoke check. It is
simpler than `test_ep.py` but covers the same MoE-shape
ElasticBuffer ctor that vLLM PR #41183 uses at runtime.

```bash
# Pod 0
kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-0 -- \
  torchrun --nproc-per-node=8 --nnodes=2 --node-rank=0 \
           --master-addr=$POD0_IP --master-port=29500 \
           /opt/tests/smoke_test.py

# Pod 1
kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-1 -- \
  torchrun --nproc-per-node=8 --nnodes=2 --node-rank=1 \
           --master-addr=$POD0_IP --master-port=29500 \
           /opt/tests/smoke_test.py
```

Expected tail on rank 0:

```
[rank 0/16] deep_ep: 2.0.x
[rank 0/16] ElasticBuffer created: <ElasticBuffer ...>
[rank 0/16] D+C mean: ~930us (20 iters)
[rank 0/16] SMOKE PASS
```

## Tuning flags that actually matter

The image's ENV block sets these defaults. Override via K8s env
or docker-run `-e` to experiment.

### `EP_EFA_MAX_QPS` (default `2`)
Cap on auto-sized QP count per peer. EFA's shared GIN ring holds
128 requests; the default stock auto-allocation is 129. Raise at
your own risk; 2 has been validated stable across H100 + H200.

### `EP_EFA_RDMA_GBS` (default `25.0`)
Effective per-node scale-out bandwidth used by
`ElasticBuffer.get_theoretical_num_sms()`. Set to what your fabric
actually delivers, not line rate. p5.48xlarge and p5en.48xlarge
measure ~25 GB/s post-Gin proxy; the default is tuned to those.

### `OFI_NCCL_GIN_MAX_REQUESTS` (default `512`)
Gin plugin's internal request ring size. Default covers the normal
`num_max_tokens_per_rank=8192` + top-8 workload. For very large
batched workloads (num_max_tokens > 16384), raise to 1024.

### `NCCL_GIN_TYPE` (default `2`)
Gin collective implementation. `1` = NVSHMEM, `2` = aws-ofi-nccl
proxy. On EFA, always `2`. On InfiniBand you could try `1`.

### `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE` (default `1`)
Allow DeepEP to use NVLink intra-node + Gin inter-node. Set to `0`
to force all-Gin (useful for isolating scale-out bugs). On
production inference, keep at `1` - NVLink is ~10x faster than
Gin for the scale-up leg.

### `VLLM_DEEPEP_V2_PREFER_OVERLAP` (default `0`)
Trade off SMs for compute/communication overlap. Default `0` uses
more SMs for dispatch/combine at the cost of GEMM throughput.
Experiment per workload.

### `VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION` (default `0`)
Use bfloat16 (lower precision, smaller transfers) vs fp32 accumulate
in combine. Default `0` is fp32 accumulate. Set to `1` to trade
precision for throughput on the scale-out leg.

## Anti-signals

When the bench looks too fast:

- **D+C p50 < 50 us** on a supposed 2-pod run - pods are colocated
  (k8s scheduler packed them onto the same node, violating the
  podAntiAffinity rule). Check `kubectl describe pod vllm-deepep-v2-{0,1}`
  and confirm they are on different `nodeName` values.
- **EFA TX delta = 0** with a bench that "passed" - Gin plugin
  didn't load and DeepEP fell back to all-NVLink. Check pod 0 log
  for `NCCL INFO Gin plugin loaded` near the top; if absent,
  `NCCL_GIN_ENABLE=1` did not take effect.

When the bench looks too slow:

- **D+C p50 > 2000 us** - almost always patch 0001 not applied
  (QP budget explosion). Re-run preflight Check 3 and grep the
  binary.
- **High per-rail imbalance in `verify_efa_traffic.sh`** - skewed
  expert placement; this is a Qwen3 / topk routing characteristic,
  not a bug.

## Companion repo

The same DeepEP V2 substrate is used for MoE *training* in the
sibling repo
[`antonai-work/nemo-rl-deepep-v2-efa`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa).
That repo's
[`docs/VALIDATION.md`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa/blob/main/docs/VALIDATION.md)
contains training-path evidence (real loss decrease + EFA TX deltas
over Megatron/NeMo-RL). Both repos apply the same DeepEP patches
`0001-0003` to the same `b306af0` base. If you are tuning DeepEP
perf on EFA, both guides are complementary.
