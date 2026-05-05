# vllm-deepep-v2-efa

**Reproducible DeepEP V2 MoE inference with vLLM on AWS EFA, built from
vanilla upstream sources + one open vLLM PR + one open DeepEP PR.**

**Status:** Dual-path Dockerfile verified locally 2026-05-05 (fast via GHCR base + vanilla inline); CodeBuild CI configured; 24-token chat completion validated on 2× p5.48xlarge H100 with DP=16 EP=16.

This repo packages a complete multi-stage Docker build chain and a
Kubernetes manifest that produces a working Qwen3-30B-A3B-FP8 MoE
serving stack on 2x p5.48xlarge (or p5en.48xlarge) H100/H200 nodes
over AWS EFA. The core path goes vanilla
`nvidia/cuda:12.9.0-devel-ubuntu24.04` -> EFA userspace -> NCCL 2.30.4
-> DeepEP V2 (patched for EFA) -> vLLM (with PR #41183 applied via
fork pin) and produces a real `/v1/chat/completions` response over
NCCL Gin + EFA, with no V1-to-V2 compat shim in the serving path.

## Sibling repos (reproducibility triad)

| Repo | Purpose | Status |
|---|---|---|
| [deepep-v2-efa-base](https://github.com/antonai-work/deepep-v2-efa-base) | Base substrate (EFA + NCCL + DeepEP V2) | v0.1.0-sm90a released |
| [nemo-rl-deepep-v2-efa](https://github.com/antonai-work/nemo-rl-deepep-v2-efa) | Training stack (Megatron-LM + NeMo-RL) | Dual-path build verified |
| [vllm-deepep-v2-efa](https://github.com/antonai-work/vllm-deepep-v2-efa) | Inference stack (this repo) | Dual-path build verified |

Together, these three repos provide end-to-end DeepEP V2 MoE reproducibility on AWS EFA, from base substrate through training and inference.

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
- **`docker/preflight.sh`** - eight in-container validation gates
  (5 base + 3 vLLM-specific) that prove the stack is assembled
  correctly before you spend cluster time running it.
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
- **`ci/buildspec.yml`** + **`ci/CODEBUILD-SETUP.md`** - AWS
  CodeBuild pipeline that builds both modes, runs preflight in each
  image, and pushes the fast build to ECR.

## Upstream PRs

Five PRs filed 2026-04-28 through 2026-05-05, covering the full training + inference stack. All five are independent, EFA-specific, and safe on non-EFA fabrics:

| Upstream repo | PR | Status (2026-05-05) | Applies to |
|---|---|---|---|
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | [#612](https://github.com/deepseek-ai/DeepEP/pull/612) | OPEN, rebased 2026-05-05 | Base substrate (all frameworks) |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | [#41183](https://github.com/vllm-project/vllm/pull/41183) | OPEN, actively reviewed | Inference (this repo) |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | [#4632](https://github.com/NVIDIA/Megatron-LM/pull/4632) | DRAFT | Training (sibling repo) |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | [#2410](https://github.com/NVIDIA-NeMo/RL/pull/2410) / [#2411](https://github.com/NVIDIA-NeMo/RL/pull/2411) | DRAFT | Training (sibling repo) |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | [#24443](https://github.com/sgl-project/sglang/pull/24443) | DRAFT | Inference (this repo) |

This repo consumes DeepEP PR #612 as `patches/0001-0003` and vLLM PR #41183 via fork pin. When both PRs merge upstream, this repo's build chain reduces to vanilla clones with no patches. See `docs/UPSTREAM-STATUS.md` for live tracking.

## Quick start

### Prerequisites
- Linux host with Docker and NVIDIA Container Toolkit
- 2x p5.48xlarge (H100) or p5en.48xlarge (H200) on AWS with EFA
  enabled and the EKS device plugin installed
- EKS cluster with kubectl access
- A PVC for model weights (Qwen3-30B-A3B-FP8 is ~31 GB)
- An image registry you can push to (e.g. ECR, GHCR)

### Build

Two build modes produce byte-for-byte equivalent images. Pick based
on whether you want the fast path (pulls the published base image
from GHCR, ~10 min) or the offline-reproducible from-vanilla path
(builds the full stack from `nvidia/cuda`, ~40 min).

```bash
git clone https://github.com/antonai-work/vllm-deepep-v2-efa
cd vllm-deepep-v2-efa

# Fast (default): FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a
docker/build.sh --mode fast    vllm-deepep-v2-efa:fast

# From vanilla: FROM nvidia/cuda:12.9.0-devel-ubuntu24.04, no GHCR dep
docker/build.sh --mode vanilla vllm-deepep-v2-efa:vanilla
```

Either invocation applies the patches deterministically; both produce
vanilla upstream + exactly the two PRs linked above.

### Fast vs from-vanilla build

Both modes resolve to the same multi-stage `docker/Dockerfile` and
produce identical final-image content, verified by the in-image
preflight (`8/8 checks PASS`). The difference is where the substrate
comes from:

| Aspect | `--mode fast` | `--mode vanilla` |
|---|---|---|
| Base stage | `FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a` | `FROM nvidia/cuda:12.9.0-devel-ubuntu24.04`, 240 lines of base stack inlined verbatim from `antonai-work/deepep-v2-efa-base/Dockerfile` |
| GHCR dependency | Yes (public or PAT with `read:packages`) | No |
| Cold build time | ~10 min (mostly vLLM install + rebuild of DeepEP against torch ABI) | ~40 min (also compiles aws-ofi-nccl and DeepEP from source) |
| Use case | Day-to-day iteration, CI pushes to ECR | Offline reproducibility, audit, clean-room rebuild |

Internally, the Dockerfile selects the base with `FROM
base-${BUILD_MODE} AS release`; swapping modes is a single
`--build-arg` change.

### Preflight (in-container)

Before deploying, confirm the image assembled correctly:

```bash
docker run --rm vllm-deepep-v2-efa:latest bash /opt/docker/preflight.sh
```

Expected output: `8/8 checks PASS` (5 base checks + 3 vLLM checks).

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

## CI/CD

Two build pipelines available:

- **GitHub Actions** (if configured): `.github/workflows/` for public GHCR push
- **AWS CodeBuild**: `ci/buildspec.yml` + setup guide in [`ci/CODEBUILD-SETUP.md`](ci/CODEBUILD-SETUP.md)

Both pipelines build fast + vanilla modes, run `docker/preflight.sh` (8/8 checks for this repo), and push the fast tag to registry. All account IDs, regions, and secret ARNs are environment variables; nothing is hardcoded.

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

Cross-framework evidence (2-node EFA traffic, NCCL init markers, DeepEP dispatch+combine latencies, chat completions) is documented across the three repos:

| Document | Coverage |
|---|---|
| [VALIDATION-EVIDENCE.md](docs/VALIDATION-EVIDENCE.md) | Per-framework E2E proofs (vLLM 24-token chat completion, TRT-LLM 4×512 tokens, etc.) |
| [EFA-TRAFFIC-EVIDENCE.md](docs/EFA-TRAFFIC-EVIDENCE.md) | Hardware-counter proof of MoE traffic over EFA (not NVLink), aggregated across all frameworks |
| [DEEPEP-BENCHMARKS.md](docs/DEEPEP-BENCHMARKS.md) | Microbenchmark guide (D+C latency ~930 us H100 / ~740 us H200, low-latency kernel, output interpretation) |

Training-side evidence (Megatron-LM, NeMo-RL) is in the sibling repo [`antonai-work/nemo-rl-deepep-v2-efa`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa) under the same filenames.

## License

Apache 2.0. Patches under `patches/` inherit the license of their
upstream repositories (Apache 2.0 for `deepseek-ai/DeepEP` and
`vllm-project/vllm`).

## Related repos and references

- Sibling base repo: https://github.com/antonai-work/deepep-v2-efa-base
- Sibling training repo: https://github.com/antonai-work/nemo-rl-deepep-v2-efa
- Upstream DeepEP V2: https://github.com/deepseek-ai/DeepEP
- Upstream vLLM: https://github.com/vllm-project/vllm
- Upstream SGLang: https://github.com/sgl-project/sglang
- AWS EFA installer: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html
- aws-ofi-nccl: https://github.com/aws/aws-ofi-nccl
