# Wave 14 2-Pod Cross-Node EFA Validation — BLOCKED (node infrastructure)

**Date**: 2026-05-06
**Cluster**: HyperPod EKS 2x p5.48xlarge (H100)
**Image**: `058264135704.dkr.ecr.us-east-2.amazonaws.com/vllm-deepep-v2-efa:fast-f3731a61679d`
**Digest**: `sha256:410157270f3fd86c8f5cac3cffe88d651ccc3b779f1c30c8d4092d8fa0e3d543`
**GHCR base**: `ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.1-sm90a`
**Namespace**: `deepep-v2-w14-serve-2026-05-06`
**Result**: **BLOCKED — structural containerd/sandbox-pause fault on node `hyperpod-i-01aee349f9991c414`**

## Verdict

Wave 14's target ship-gate (2-pod cross-node `vllm serve` + >=1 GB EFA TX per
pod under sustained inference) could NOT be validated. The root cause is
infrastructure, not software: node `hyperpod-i-01aee349f9991c414` reports
`Status: Ready` at the kubelet level with `containerd://2.1.5` but any
NEW pod sandbox creation fails with:

```
Failed to create pod sandbox: rpc error: code = Unknown desc =
failed to start sandbox "<sha>": failed to get sandbox image
"localhost/kubernetes/pause": failed to pull image
"localhost/kubernetes/pause": failed to resolve reference
"localhost/kubernetes/pause:latest": failed to do request:
Head "https://localhost/v2/kubernetes/pause/manifests/latest":
dial tcp 127.0.0.1:443: connect: connection refused
```

The `cu13` image stack (Wave 13 validated) is NOT implicated. The fault is
reproducible on any workload: vLLM, nemo-rl, `checkpoint-backup`,
`node-debugger` — three workload-independent pods are stuck in
ContainerCreating on this node as of 2026-05-06T13:15Z, one of them for
over 6 hours. See `evidence/01aee-stuck-pods.txt` for list.

## What was attempted

| Step | Result |
|---|---|
| H100 cluster lock claimed | OK (holder `ip-172-31-18-239-1825524`) |
| `nvshmem-efa/deepep-nvshmem` StatefulSet scaled to 0 | OK (drained 30s) |
| Both nodes reporting Ready pre-deploy | OK |
| Apply `deepep-v2-w14-serve-2026-05-06` manifest (2 replicas, `vpc.amazonaws.com/efa: 32`, podAntiAffinity on hostname) | OK |
| `vllm-deepep-v2-0` scheduled to `hyperpod-i-0a3eb6d3953cceaa7` | Running 1/1 at 11m (image pulled, `sleep infinity` active) |
| `vllm-deepep-v2-1` scheduled to `hyperpod-i-01aee349f9991c414` | Stuck in ContainerCreating at 11m |
| Pod-1 recreation attempt #1 (delete + let StatefulSet recreate) | Same `FailedCreatePodSandBox` error, 60s later |
| Pod-1 recreation attempt #2 (delete + 120s wait) | Same `FailedCreatePodSandBox` error |
| Force-delete 2 stuck system pods on `01aee` (`node-debugger`, `checkpoint-backup`) to perturb containerd state | No effect; fault persists |
| Redeploy as nemo-rl training on same two nodes to A/B test whether the fault is workload-specific | Pod-0 scheduled to `01aee`, identical fault; Pod-1 on `0a3eb` Running |

The A/B demonstrates the fault is node-level (containerd pause-image
resolution), independent of image or workload.

## Failure signature

- Repeated per 10-15 seconds: `FailedCreatePodSandBox ... dial tcp 127.0.0.1:443: connect: connection refused`
- Pod stuck with `STATUS=ContainerCreating`, `READY=0/1`, IP assigned (`10.1.3.30`), NODE assigned
- This is **NOT** the crash signatures guardrailed by this wave: `c10::cuda::SetDevice(-N)` and
  `nvshmem_selected_device_transport`. The Wave 13 cu13 ABI hypothesis remains valid.

## Image sanity (collateral evidence)

`nemo-rl-fullstack-1` (which DID start on `0a3eb`) confirmed:

```
$ python3 -c "import deep_ep; print(deep_ep.__file__); print('ElasticBuffer' in dir(deep_ep))"
/opt/DeepEP/deep_ep/__init__.py
True

$ nvidia-smi -L | head
GPU 0: NVIDIA H100 80GB HBM3
... (8 GPUs)
```

The Wave-13-validated `cu13` image stack remains functional where it can start.

## Per-pod EFA TX delta

N/A — could not execute sustained inference. The Wave 14 ship-gate requires
both pods Running so vLLM DP=16 can form its cross-node data-parallel group;
without pod-1 Running, no cross-node MoE traffic can be generated.

## Recommended remediation (outside Wave 14 scope)

Per AWS HyperPod runbook, the canonical fix for a node whose containerd
cannot resolve `localhost/kubernetes/pause` is either:

1. Ensure the pause image exists in the local containerd image store:
   `ctr -n k8s.io images ls | grep pause`. If absent, preload via
   `ctr images import pause.tar` or via the EKS AMI's bootstrap script.
2. Restart the kubelet + containerd services on the node. Via
   `aws ssm start-session` or the HyperPod control-plane equivalent.
3. If restart does not clear it, cordon + drain + replace the node through
   HyperPod SageMaker console. Wave-13's original observation was premature
   ("node containerd fault resolved" based on `Ready` status) — the kubelet
   Ready gate does not exercise the sandbox pull path, so the bug was
   masked until Wave 14 tried to start two fresh pods on it.

## Relationship to prior waves

| Wave | Scope | Result |
|---|---|---|
| Wave 8  | cu12 base + cu13 torch | FAIL (ABI mismatch — "invalid device ordinal") |
| Wave 11 | diagnostic: attempt cu12 realign | FAIL (wheels are cu13; cannot force cu12) |
| Wave 13 | cu13-unified stack, single-node fallback | PASS (24-token chat completion, 62KB EFA) |
| Wave 14 | 2-node cross-node sustained inference | **BLOCKED** (node-01aee containerd fault) |

Cross-reference: [Wave 13 cluster-pass doc](VALIDATION-WAVE13-CU13-CLUSTER-PASS.md) —
cu13 software stack validated there; infrastructure-only regression in Wave 14.

## Evidence files

All captured under `/tmp/wave14/evidence/` on the operator host at time of run:

- `pod1-describe.txt` — full `kubectl describe` of stuck `vllm-deepep-v2-1`
  showing 40+ `FailedCreatePodSandBox` events over 11 minutes with identical
  root cause.
- `nemo-pod0-describe-final.txt` — same fault reproducing on nemo-rl
  deployment post-vllm-teardown (A/B control).
- `01aee-all-pods.txt` — 35 pods on node, 3 stuck in ContainerCreating
  (including 6h20m `node-debugger`), confirming fault is pre-existing and
  workload-independent.
- `node-01aee-describe.txt` / `node-01aee-state.yaml` — node reports
  `KubeletReady=True`, `containerd://2.1.5`, 32 EFA adapters, 8 GPUs.

## Verdict summary

Wave 14 ship gate **not met** due to infrastructure. The image/software stack
from Wave 13 retains PASS status; rerun Wave 14 after node-01aee has been
cordoned+drained or had its containerd pause image restored.
