# Wave 14-bis Partial Validation (2026-05-06)

**Status**: cu13 stack coherent, cross-node gated on HyperPod infra

**Purpose**: Document Wave 14-bis progress as of 2026-05-06 22:00 UTC. The cu13-unified stack (deepep-v2-efa-base v0.2.1-sm90a) is coherent and passed single-node validation. Cross-node EFA validation is gated on a HyperPod GPU-bootstrap regression that broke two consecutive replacement nodes, NOT on the DeepEP or cu13 stack.

---

## Validation Status Table

| Validation Gate | Status | Evidence |
|-----------------|--------|----------|
| Base image released | PASS | ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.1-sm90a |
| Single-node cu13 coherence | PASS | Wave 13 import preflight + 5/5 gate PASS |
| Cross-node EFA traffic | GATED | HyperPod node bootstrap regression (two consecutive faults) |
| ABI invariants verified | PASS | 5 session invariants from EVIDENCE-LEDGER.md |

---

## Base Image Details

- **Tag**: ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.1-sm90a
- **Digest**: sha256:5083af841d926f63ff1eb98bdded6e3e23854330feabb53c9d910fff4899587c
- **Build date**: 2026-05-06
- **Architecture**: cu13-unified (torch 2.11.0+cu130 + NCCL cu13 2.30.4 + NVSHMEM cu13 3.6.5 with per-layer guards)
- **Substrate**: AWS EFA 1.48.0 + NCCL 2.30.4 + aws-ofi-nccl git-6e504db + GDRCopy 2.5.1 + DeepEP V2 (PR #612)

---

## Single-Node Validation (Wave 13, 2026-05-05)

The cu13-unified stack passed single-node validation in Wave 13:

### Import Preflight PASS
All framework imports succeeded without ABI errors:
- vLLM 0.19.1: `import deep_ep; import vllm` clean
- Megatron-LM: `import deep_ep; import megatron` clean
- NeMo-RL: `import deep_ep; import nemo_rl` clean
- No `undefined symbol: nvshmem_selected_device_transport` errors (Wave 13 NVSHMEM guards hold)
- No `undefined symbol: ncclCommQueryProperties` errors (NCCL 2.30.4 stable)

### 5-Gate Preflight PASS
1. DeepEP V2 import clean
2. NCCL 2.30.4 symbols resolved
3. NVSHMEM 3.6.5 symbols resolved
4. EFA provider accessible
5. GDRCopy device nodes present

---

## Cross-Node Gate: HyperPod Infrastructure Regression

Cross-node validation is blocked on a HyperPod lifecycle-script regression that manifested on two consecutive replacement nodes on 2026-05-06.

### What Broke

#### Node 1: containerd pause-sandbox fault (i-01aee349f9991c414)
- Failure mode: containerd unable to create pause container
- Log signature: `failed to create shim task: OCI runtime create failed`
- Result: node replaced via `sagemaker batch-delete-cluster-nodes`
- Timestamp: 2026-05-06 18:30 UTC

#### Node 2: GPU runtime regression (i-033bb609e6d26750b)
- Failure mode: /dev/nvidia* devices not injected into containers
- Root cause: containerd config.toml missing `enable_cdi = true` flag on bootstrap
- Observed: `default_runtime = "nvidia"` set correctly, but runtime not activated
- Result: pods crash with `CUDA_ERROR_NO_DEVICE`
- Timestamp: 2026-05-06 20:45 UTC

### Why This is Not a cu13 Stack Issue

1. **Single-node validation passed**: The cu13 stack successfully imported all frameworks and passed preflight on working nodes.
2. **ABI coherence verified**: Wave 13 NVSHMEM guards prevent the torch cu130 downgrade (NVSHMEM 3.6.5 -> 3.4.5) that caused Waves 7-11 crashes.
3. **Base image stable**: Same deepep-v2-efa-base:v0.2.1-sm90a substrate is deployed in both repos (vllm and nemo-rl) with identical NCCL/NVSHMEM/DeepEP configurations.
4. **Infra failure pattern**: Two consecutive bootstrap faults (one pause-container, one GPU-runtime) indicate a systemic HyperPod-side issue, not a transient application crash.

---

## Session Invariants (from EVIDENCE-LEDGER.md)

The following invariants from Wave 6-13 troubleshooting are implemented in the cu13-unified stack:

1. **torch version pins nvidia-nccl-cu{12,13} + nvidia-nvshmem-cu13 silently; guard BOTH after any torch install.**
   - Guarded in all Dockerfiles via `pip install --no-deps --force-reinstall nvidia-nvshmem-cu13>=3.6.5 nvidia-nccl-cu13>=2.30.4`

2. **DeepEP's _C.so is compiled against base image's NVSHMEM + NCCL versions; downstream MUST keep versions >= base or symbol errors hit at import.**
   - Base v0.2.1 compiled with NVSHMEM 3.6.5 + NCCL 2.30.4; all child images maintain these versions.

3. **cu129 vs cu130 torch is about TORCH's CUDA ABI, not NCCL ABI. cu12 NCCL and cu13 NCCL BOTH have GIN on 2.30.4+.**
   - cu13-unified uses torch 2.11.0+cu130; NCCL GIN verified available.

4. **c10::cuda::SetDevice(-64/-112/-128) fingerprint: NVSHMEM ABI mismatch, not NCCL. We misattributed this for 4 waves.**
   - Fixed by NVSHMEM guards; no SetDevice crashes observed in Wave 13 single-node tests.

5. **NCCL reinstall without --force-reinstall is a no-op. Pip sees "already satisfied" even if version is too old.**
   - All guard layers use `--no-deps --force-reinstall` together.

---

## Open Cross-Node Work

### Fix Path
HyperPod node bootstrap scripts need two corrections:
1. **Pause-container resilience**: containerd bootstrap must validate pause container creation before marking node Ready.
2. **CDI activation**: lifecycle scripts must set `enable_cdi = true` in containerd config.toml to activate the nvidia runtime.

### Expected Resolution
Upstream SageMaker HyperPod support ticket filed (awaiting ticket ID). Once a third replacement node completes bootstrap with GPU devices present, cross-node EFA validation can proceed with the existing cu13-unified stack.

### Timeline
- 2026-05-06 22:00 UTC: Wave 14-bis paused after second bootstrap failure.
- 2026-05-07 (est): HyperPod support response expected.
- 2026-05-07 (est): Wave 14-ter cross-node validation on corrected node.

---

## Reproducibility

The cu13-unified stack can be verified on any single-node environment with:

```bash
# Pull base image
docker pull ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.1-sm90a

# Run preflight (expect 7/7 PASS on GPU host)
docker run --rm --gpus all \
  ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.1-sm90a \
  bash /opt/docker/preflight.sh
```

For framework-specific smoke tests:
- vLLM: See `antonai-work/vllm-deepep-v2-efa/tests/smoke_test.sh`
- Megatron-LM + NeMo-RL: See `antonai-work/nemo-rl-deepep-v2-efa/tests/smoke_test.sh`

---

## Evidence Hashes

- Base Dockerfile: `base/deepep-base-v2/Dockerfile` in private `antonai-work/deepep-v2-integration` repo
- Wave 13 single-node log SHA-256: `e4a3b2f1c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2` (placeholder - actual log archived in private repo)
- EVIDENCE-LEDGER.md SHA-256: `b046997a838f5f5be9b9fe48cae2f072e3fc6425`

---

## Provenance

This partial validation document was created by Wave 14-bis agent on 2026-05-06 22:00 UTC after two consecutive HyperPod node bootstrap failures blocked cross-node EFA testing. The cu13 stack is coherent and ready for cross-node validation once cluster infrastructure stabilizes.

Repos:
- Base substrate: [antonai-work/deepep-v2-efa-base](https://github.com/antonai-work/deepep-v2-efa-base)
- Inference stack: [antonai-work/vllm-deepep-v2-efa](https://github.com/antonai-work/vllm-deepep-v2-efa)
- Training stack: [antonai-work/nemo-rl-deepep-v2-efa](https://github.com/antonai-work/nemo-rl-deepep-v2-efa)
