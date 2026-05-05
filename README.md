# vllm-deepep-v2-efa

**Reproducible DeepEP V2 MoE inference with vLLM on AWS EFA, built from
vanilla upstream sources + one open vLLM PR + one open DeepEP PR.**

This repo packages a complete multi-stage Docker build chain and a
Kubernetes manifest that produces a working Qwen3-30B-A3B-FP8 MoE
serving stack on 2x p5.48xlarge (or p5en.48xlarge) H100/H200 nodes
over AWS EFA. The core path goes vanilla
`nvidia/cuda:12.9.0-devel-ubuntu24.04` -> EFA userspace -> NCCL 2.30.4
-> DeepEP V2 (patched for EFA) -> vLLM (with PR #41183 applied via
fork pin) and produces a real `/v1/chat/completions` response over
NCCL Gin + EFA, with no V1-to-V2 compat shim in the serving path.

Companion repo for training:
[`antonai-work/nemo-rl-deepep-v2-efa`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa)
uses the same DeepEP V2 + EFA substrate for Megatron/NeMo-RL training.

## What's inside

- **`patches/`** - four standalone `.patch` / `.diff` files: three
  DeepEP patches (`0001-0003`, from our open PR #612) and one vLLM
  patch (`vllm-pr41183-deepep-v2.diff`, a snapshot of the upstream
  vLLM PR #41183 for offline replay when the fork SHA is reachable).
  Each patch header cites its upstream PR.
- **`docker/Dockerfile`** - single multi-stage build from
  `nvidia/cuda:12.9.0-devel-ubuntu24.04` through EFA + aws-ofi-nccl
  + NCCL + GDRCopy + DeepEP V2 (patched) + vLLM (fork pin).
- **`docker/build.sh`** - one-command build script.
- **`docker/preflight.sh`** - seven in-container validation gates
  that prove the stack is assembled correctly before you spend
  cluster time running it.
- **`tests/smoke_test.py`** - a 2-pod `torchrun` driver that
  constructs `deep_ep.ElasticBuffer` with the same MoE-shape ctor
  vLLM PR #41183 uses, runs a 20-iter dispatch+combine round trip,
  and prints `D+C mean: NNN.Nus  SMOKE PASS`.
- **`tests/serve_chat_completion.sh`** - the end-to-end serving
  driver: boots `vllm serve Qwen/Qwen3-30B-A3B-FP8` with
  `--all2all-backend deepep_v2` across 2 pods, waits for `/health`,
  and hits `/v1/chat/completions` with a prompt, expecting a real
  JSON response.
- **`tests/k8s/multi-node-serving-h100.yaml`** - 2-pod StatefulSet
  for EKS + EFA-enabled HyperPod (or vanilla EKS with EFA plugin).
- **`tests/verify_efa_traffic.sh`** - EFA counter snapshot + delta
  check that proves MoE traffic went over EFA, not NVLink.
- **`docs/`** - deeper explanations of architecture, validation,
  upstream status, and DeepEP microbenchmarks. See
  [`docs/EFA-TRAFFIC-EVIDENCE.md`](docs/EFA-TRAFFIC-EVIDENCE.md)
  for the aggregated cross-node EFA hardware-counter proof across
  all frameworks.

## Upstream PRs

Two PRs, one per upstream repo:

| Upstream repo | PR | Status | Patches |
|---|---|---|---|
| [`deepseek-ai/DeepEP`](https://github.com/deepseek-ai/DeepEP) | [#612](https://github.com/deepseek-ai/DeepEP/pull/612) | OPEN, 3 commits | `patches/0001-0003` |
| [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | [#41183](https://github.com/vllm-project/vllm/pull/41183) | OPEN, 16 commits (active) | `patches/vllm-pr41183-deepep-v2.diff` |

See `docs/UPSTREAM-STATUS.md` for live tracking of merge state.
When both merge, this repo's build chain reduces to vanilla clones
with no patches.

## Quick start

### Prerequisites
- Linux host with Docker and NVIDIA Container Toolkit
- 2x p5.48xlarge (H100) or p5en.48xlarge (H200) on AWS with EFA
  enabled and the EKS device plugin installed
- EKS cluster with kubectl access
- A PVC for model weights (Qwen3-30B-A3B-FP8 is ~31 GB)
- An image registry you can push to (e.g. ECR, GHCR)

### Build

```bash
git clone https://github.com/antonai-work/vllm-deepep-v2-efa
cd vllm-deepep-v2-efa
docker/build.sh vllm-deepep-v2-efa:latest   # ~40 min cold build
```

The build script applies all patches before compiling, so you know
you're getting vanilla upstream + exactly the two PRs linked above.

### Preflight (in-container)

Before deploying, confirm the image assembled correctly:

```bash
docker run --rm vllm-deepep-v2-efa:latest bash /opt/docker/preflight.sh
```

Expected output: `7/7 checks PASS`.

### Deploy + run serving

```bash
# Push to your registry (placeholder, replace with your ECR/GHCR path)
docker tag vllm-deepep-v2-efa:latest <your-registry>/vllm-deepep-v2-efa:latest
docker push <your-registry>/vllm-deepep-v2-efa:latest

# Edit tests/k8s/multi-node-serving-h100.yaml: update image: line and
# claimName for the weights PVC, then:
kubectl apply -f tests/k8s/multi-node-serving-h100.yaml
kubectl -n vllm-deepep-v2-efa wait --for=condition=Ready pod --all --timeout=5m

# Two terminals, one per pod; launcher scripts are baked into the image.
# Terminal 1 (leader, pod 0):
POD0_IP=$(kubectl -n vllm-deepep-v2-efa get pod vllm-deepep-v2-0 -o jsonpath='{.status.podIP}')
kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-0 -- \
  env MASTER_IP=$POD0_IP ROLE=leader bash /opt/tests/serve_chat_completion.sh

# Terminal 2 (worker, pod 1):
kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-1 -- \
  env MASTER_IP=$POD0_IP ROLE=worker bash /opt/tests/serve_chat_completion.sh
```

After both pods print `Application startup complete` 16 times
(one per DP worker), issue the chat-completion probe:

```bash
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"Qwen/Qwen3-30B-A3B-FP8",
         "messages":[{"role":"user","content":"hello"}],
         "max_tokens":24}'
```

### Validate

Expected log lines in pod 0:

```
NCCL INFO NET/OFI Initializing aws-ofi-nccl git-6e504db
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct
NCCL INFO NET/OFI Using transport protocol RDMA
has_deep_ep_v2 = True
Using DeepEPV2All2AllManager all2all manager
Application startup complete.
```

Expected chat-completion response shape (exact tokens vary):

```json
{"id":"chatcmpl-XXXX","object":"chat.completion",
 "model":"Qwen/Qwen3-30B-A3B-FP8",
 "choices":[{"index":0,"message":{"role":"assistant","content":"<think>..."},
             "finish_reason":"length"}],
 "usage":{"prompt_tokens":9,"completion_tokens":24,"total_tokens":33}}
```

EFA traffic check:

```bash
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  bash /opt/tests/verify_efa_traffic.sh snapshot /tmp/before
# ... issue a few chat completions ...
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  bash /opt/tests/verify_efa_traffic.sh verify /tmp/before /tmp/after 1
```

Expected: `EFA TX delta >= 1 GB (PASS)`.

## Reference validation (2026-04-29)

Measured on 2x p5.48xlarge H100 EFA with the same substrate this
repo builds from source:

| Gate | Observed |
|---|---|
| `has_deep_ep_v2() = True` | PASS |
| `DeepEPV2All2AllManager` active (no V1 shim in serving path) | PASS |
| NCCL transport | `efa-direct` with 32 NICs |
| `/v1/chat/completions` response | 24 real completion tokens |
| MoE DP/EP configuration | DP=16 (8 local per pod), EP=16 |
| Model | `Qwen/Qwen3-30B-A3B-FP8` bf16 activations |
| DeepEP V2 D+C p50 (microbench, 2-node) | ~930 us (H100), ~740 us (H200) |

Full per-step log lines and the verbatim chat-completion JSON are
quoted in `docs/VALIDATION.md` so reviewers without AWS access can
verify the traceability.

## Benchmarking

If you want to isolate DeepEP D+C performance from end-to-end
serving, run the upstream DeepEP microbenchmarks inside the same
image. The full dispatch+combine harness
(`tests/elastic/test_ep.py`) gives you the ~930 us p50 baseline on
2-node p5.48xlarge H100 (~740 us on p5en.48xlarge H200). Both are
covered in `docs/DEEPEP-BENCHMARKS.md`, including 2-pod
`kubectl exec` invocations, the flags that actually matter
(`EP_EFA_MAX_QPS=2`, `EP_EFA_RDMA_GBS=25.0`,
`OFI_NCCL_GIN_MAX_REQUESTS=512`), and how to read the output.

## Why a separate public repo?

The two PRs in the upstream repos above are independent and can
merge on their own schedules. But composing them together on AWS
EFA requires both applied simultaneously to matched versions of
their dependencies (NCCL 2.30.4, aws-ofi-nccl at `6e504db`,
`torch>=2.10`). This repo is the single source of truth for
"what version of everything" and "how do I assemble it into a
working image."

When both PRs merge upstream, this repo's build chain reduces to:

```
pip install vllm>=<release>   # pulls merged PR #41183
git clone https://github.com/deepseek-ai/DeepEP -b <tag>
```

Until then, the patches in `patches/` let anyone reproduce the
validated stack today.

## Validation

Full cross-framework verbatim evidence (marker counts, curl
responses, activation log lines, EFA counter deltas, image digests,
SHA-256 hashes of raw log chunks) is inlined in
[`docs/VALIDATION-EVIDENCE.md`](docs/VALIDATION-EVIDENCE.md). This
covers vLLM Run #10 (real 24-token Qwen3-30B-A3B-FP8 chat completion
on DP=16 EP=16) and TRT-LLM Run #5 (DeepEP fast-path activation
closing the `enable_attention_dp` gate, 4x512 tokens, 4.58 GB EFA
TX).

Training-side evidence (Megatron-LM, NeMo-RL, SGLang) is in the
sibling repo
[`antonai-work/nemo-rl-deepep-v2-efa`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa)
under the same filename.

## License

Apache 2.0. Patches under `patches/` inherit the license of their
upstream repositories (Apache 2.0 for `deepseek-ai/DeepEP` and
`vllm-project/vllm`).

## Related repos and references

- Upstream DeepEP V2: https://github.com/deepseek-ai/DeepEP
- Upstream vLLM: https://github.com/vllm-project/vllm
- vLLM DeepEP V2 PR: https://github.com/vllm-project/vllm/pull/41183
- DeepEP V2 release PR: https://github.com/deepseek-ai/DeepEP/pull/605
- AWS EFA installer: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html
- aws-ofi-nccl: https://github.com/aws/aws-ofi-nccl
- aws-ofi-nccl GIN fix: https://github.com/aws/aws-ofi-nccl/commit/6e504db
- Companion training repo:
  https://github.com/antonai-work/nemo-rl-deepep-v2-efa
