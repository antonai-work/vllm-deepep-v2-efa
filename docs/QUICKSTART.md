# Quickstart - 25 minutes from clone to running inference

Assumes you already have:
- A Linux build host with Docker and NVIDIA Container Toolkit
- An EKS cluster with 2x p5.48xlarge H100 nodes (or p5en.48xlarge H200)
- EFA device plugin installed on the nodes
  (https://github.com/aws-samples/aws-efa-eks)
- An image registry you can push to (ECR, GHCR, etc.)
- `kubectl` with access to the cluster
- Registry CLI (aws/ghcr/etc.) authenticated for push

If you're missing any of those, see `docs/ARCHITECTURE.md` for
prerequisites.

## 1. Clone + build (40 min)

```bash
git clone https://github.com/antonai-work/vllm-deepep-v2-efa
cd vllm-deepep-v2-efa

# Set your registry target
export REGISTRY=<your-account>.dkr.ecr.<region>.amazonaws.com/vllm-deepep-v2-efa
export TAG=v2efa-$(git rev-parse --short HEAD)

docker/build.sh "$REGISTRY:$TAG"
```

The `build.sh` script:
1. Applies patches `0001`-`0003` against vanilla DeepEP V2 (@ `b306af0`)
2. Compiles DeepEP V2 `_C.so` against NCCL 2.30.4
3. Clones `tlrmchlsmth/vllm@deepep-v2-integration` at a pinned SHA
   (PR #41183) and editable-installs it
4. Rebuilds DeepEP V2's `_C.so` against torch's runtime ABI
5. Produces a final image with all env vars preset

Expected output at the end:

```
Successfully tagged <REGISTRY>:v2efa-<sha>
[build.sh] Validate: docker run --rm <REGISTRY>:<TAG> bash /opt/docker/preflight.sh
```

## 2. Preflight (2 min)

Before deploying to the cluster, verify the image assembled correctly:

```bash
docker run --rm "$REGISTRY:$TAG" bash /opt/docker/preflight.sh
```

Expected: `7/7 checks PASS`. See `docs/VALIDATION.md` for the exact
checks. If any fail, re-build from a clean workspace.

## 3. Push to registry (5 min)

For ECR:
```bash
aws ecr get-login-password --region us-east-2 \
  | docker login --username AWS --password-stdin $REGISTRY
docker push "$REGISTRY:$TAG"
```

For GHCR:
```bash
echo $GHCR_PAT | docker login ghcr.io -u <user> --password-stdin
docker push "$REGISTRY:$TAG"
```

## 4. Deploy the K8s manifest (2 min)

Edit `tests/k8s/multi-node-serving-h100.yaml`:
- Replace `REPLACE_WITH_YOUR_REGISTRY/vllm-deepep-v2-efa:latest` with
  your pushed image tag (`$REGISTRY:$TAG`)
- Optional: uncomment the `HF_HOME` / `HUGGINGFACE_HUB_CACHE` env vars
  and the `/mnt/weights` volumeMount if you want to pre-cache weights
  on a PVC (Qwen3-30B-A3B-FP8 is ~31 GB)
- Adjust namespace if you want something other than the default
  `vllm-deepep-v2-efa`

Apply:
```bash
kubectl apply -f tests/k8s/multi-node-serving-h100.yaml
kubectl -n vllm-deepep-v2-efa rollout status statefulset/vllm-deepep-v2 --timeout=10m
```

## 5. Launch `vllm serve` on both pods (8 min: ~5 min weight load + ~3 min JIT warmup)

Open two terminals, one per pod:

### Terminal 1: pod 0 (DP leader)

```bash
POD0_IP=$(kubectl -n vllm-deepep-v2-efa get pod vllm-deepep-v2-0 -o jsonpath='{.status.podIP}')
kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-0 -- \
  env MASTER_IP=$POD0_IP ROLE=leader bash /opt/tests/serve_chat_completion.sh
```

### Terminal 2: pod 1 (DP worker)

```bash
POD0_IP=$(kubectl -n vllm-deepep-v2-efa get pod vllm-deepep-v2-0 -o jsonpath='{.status.podIP}')
kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-1 -- \
  env MASTER_IP=$POD0_IP ROLE=worker bash /opt/tests/serve_chat_completion.sh
```

Watch the pod 0 log for:

```
NCCL INFO NET/OFI Initializing aws-ofi-nccl git-6e504db
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct
NCCL INFO NET/OFI Using transport protocol RDMA

Using DeepEPV2All2AllManager all2all manager
has_deep_ep_v2 = True

Application startup complete.
```

When you see 16 `Application startup complete` lines in pod 0
(one per DP worker across both pods), the server is ready.

## 6. Issue a chat completion (from either pod)

```bash
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"Qwen/Qwen3-30B-A3B-FP8",
         "messages":[{"role":"user","content":"hello"}],
         "max_tokens":24}'
```

Expected response (exact tokens vary):

```json
{"id":"chatcmpl-XXXX","object":"chat.completion",
 "model":"Qwen/Qwen3-30B-A3B-FP8",
 "choices":[{"index":0,
             "message":{"role":"assistant","content":"<think>..."},
             "finish_reason":"length"}],
 "usage":{"prompt_tokens":9,"completion_tokens":24,"total_tokens":33}}
```

## 7. Verify EFA traffic (1 min)

Before issuing completions, snapshot:

```bash
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  bash /opt/tests/verify_efa_traffic.sh snapshot /tmp/before
```

Issue 10-20 chat completions, then snapshot again and compare:

```bash
kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  bash /opt/tests/verify_efa_traffic.sh snapshot /tmp/after

kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
  bash /opt/tests/verify_efa_traffic.sh verify /tmp/before /tmp/after 1
```

Expected: `TOTAL TX DELTA: ... GB across N rails` followed by `PASS: EFA
traffic verified real.`

## 8. Teardown

```bash
kubectl delete -f tests/k8s/multi-node-serving-h100.yaml
```

## Troubleshooting

**Build fails at patch 0001 apply step.** Upstream DeepEP may have
moved past `b306af0`. Re-sync with
`git -C /opt/DeepEP rebase origin/main` or rebase the PR branch,
then re-try.

**`NCCL INFO NET/Socket` instead of `NET/OFI`.** `LD_LIBRARY_PATH`
does not include `/opt/aws-ofi-nccl/lib`. Check with
`docker run --rm <image> bash -c 'echo $LD_LIBRARY_PATH'` - the
`/opt/aws-ofi-nccl/lib` prefix must appear first.

**Serving hangs at first dispatch.** Almost always
`num_allocated_qps > EFA budget`. Check patch 0001 applied with
`docker run --rm <image> grep 'num_allocated_qps' /opt/DeepEP/deep_ep/buffers/elastic.py`.

**`has_deep_ep_v2() = False` in preflight.** Either PR #41183 not
applied (check the vLLM fork SHA), NCCL < 2.30.4 at runtime (run
Check 2/7 manually), or `deep_ep.ElasticBuffer` not importable.

**`/v1/chat/completions` hangs with no response.** Server still in
JIT warmup (FlashInfer CUTLASS kernels compile takes ~5-8 min first
time). Tail the leader log for `engine loop is running` messages; if
that never appears, tail for traceback.

**EFA TX delta = 0 in `verify_efa_traffic.sh`.** Either pods
colocated (scheduling issue), `FI_PROVIDER != efa`, or the server
never actually routed through DeepEP (misconfig). Check
`kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- env | grep -E 'FI_|NCCL_GIN|DEEP_EP'`.

More troubleshooting: `docs/VALIDATION.md` has a full failure-mode
table.
