#!/usr/bin/env python3
"""
Smoke test for vllm-deepep-v2-efa: imports deep_ep, constructs an
ElasticBuffer with the same MoE-shape ctor that vLLM PR #41183 uses,
runs one dispatch + combine round trip, and prints D+C latency.

Mirrors DeepEP's own tests/elastic/test_ep.py ctor signature plus the
`group=pg, num_max_tokens_per_rank=..., hidden=..., num_topk=...,
use_fp8_dispatch=..., allow_hybrid_mode=...` kwargs that
DeepEPV2All2AllManager in PR #41183 threads through.

Run inside the image as:
    torchrun --nproc-per-node=8 --nnodes=2 --node-rank=0 \
             --master-addr=<pod-0-ip> --master-port=29500 \
             /opt/tests/smoke_test.py

Expected tail on rank 0:
    deep_ep: 2.0.x
    ElasticBuffer created
    D+C mean: NNN.Nus (20 iters)
    SMOKE PASS

Tunables via env (all optional):
    SMOKE_NUM_MAX_TOKENS      default 8192
    SMOKE_HIDDEN              default 7168
    SMOKE_NUM_TOPK            default 8
    SMOKE_NUM_EXPERTS         default 256
    SMOKE_ITERS               default 20
    SMOKE_ALLOW_HYBRID_MODE   default 1 (set 0 to force all-EFA on multi-node)
"""
import os
import sys
import time
import torch
import torch.distributed as dist
import deep_ep


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def main() -> int:
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    num_max_tokens = env_int("SMOKE_NUM_MAX_TOKENS", 8192)
    hidden = env_int("SMOKE_HIDDEN", 7168)
    num_topk = env_int("SMOKE_NUM_TOPK", 8)
    num_experts = env_int("SMOKE_NUM_EXPERTS", 256)
    iters = env_int("SMOKE_ITERS", 20)
    allow_hybrid_mode = bool(env_int("SMOKE_ALLOW_HYBRID_MODE", 1))

    # Dispatch a small slice of num_max_tokens so we actually move bytes
    # but stay well within the buffer's per-rank capacity.
    num_tokens = min(128, num_max_tokens)

    if rank == 0:
        print(f"[rank 0/{world}] deep_ep: {getattr(deep_ep, '__version__', '?')}", flush=True)
        print(f"[rank 0/{world}] num_max_tokens_per_rank={num_max_tokens} "
              f"hidden={hidden} num_topk={num_topk} num_experts={num_experts} "
              f"allow_hybrid_mode={allow_hybrid_mode}", flush=True)

    # MoE-shape ElasticBuffer ctor (matches PR #41183 /
    # DeepEPV2All2AllManager.get_buffer).
    buf = deep_ep.ElasticBuffer(
        group=dist.group.WORLD,
        num_max_tokens_per_rank=num_max_tokens,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=False,
        allow_hybrid_mode=allow_hybrid_mode,
        explicitly_destroy=True,
    )
    if rank == 0:
        print(f"[rank 0/{world}] ElasticBuffer created: {buf}", flush=True)

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, num_topk), dtype=torch.int64, device="cuda"
    )
    topk_w = torch.rand(num_tokens, num_topk, dtype=torch.float, device="cuda")

    # Warmup: 1 dispatch + 1 combine.
    recv_x, _, _, handle, _ = buf.dispatch(
        x, topk_idx, topk_w, num_experts=num_experts
    )
    recv_tensor = recv_x if isinstance(recv_x, torch.Tensor) else recv_x[0]
    buf.combine(recv_tensor, handle)
    torch.cuda.synchronize()

    # Timed loop.
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        recv_x, _, _, handle, _ = buf.dispatch(
            x, topk_idx, topk_w, num_experts=num_experts
        )
        recv_tensor = recv_x if isinstance(recv_x, torch.Tensor) else recv_x[0]
        buf.combine(recv_tensor, handle)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()

    mean_us = (t1 - t0) / 1000.0 / iters
    if rank == 0:
        print(f"[rank 0/{world}] D+C mean: {mean_us:.1f}us ({iters} iters)", flush=True)
        print(f"[rank 0/{world}] SMOKE PASS", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
