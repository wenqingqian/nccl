"""
Unified mask all-reduce test entry point.

Run with:
    cd tests
    torchrun --nproc_per_node=8 test_mask.py correctness
    torchrun --nproc_per_node=8 test_mask.py perf
    torchrun --nproc_per_node=8 test_mask.py perf --sizes 1MB,2MB,4MB --sparsities 0.5
    torchrun --nproc_per_node=8 test_mask.py perf --distributions block,random

Subcommands:
    correctness  Pack-level correctness tests (1MB tensor, force Simple).
    perf         Performance sweep over size x sparsity x distribution x version.

Environment:
    NCCL_PROTO=Simple is set automatically so the mask logic in
    src/device/prims_simple.h is exercised.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from mask_utils import (
    BYTES_PER_PACK,
    FLOATS_PER_PACK,
    generate_mask,
    list_mask_versions,
    mask_all_reduce,
    timed_all_reduce,
    vanilla_all_reduce,
    free_memory,
)


def _init():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    os.environ.setdefault("NCCL_PROTO", "Simple")
    return rank, world_size


def _check(rank, tensor, expected, desc):
    ok = torch.allclose(tensor.cpu(), expected, rtol=1e-5, atol=1e-5)
    if rank == 0:
        print(f"[{desc}] {'PASS' if ok else 'FAIL'}")
    if not ok and rank == 0:
        bad = (~torch.isclose(tensor.cpu(), expected, rtol=1e-5, atol=1e-5)).nonzero(as_tuple=True)[0]
        print(f"  first bad: {bad[:5].tolist()}")
    return ok


# -----------------------------------------------------------------------------
# Step 1: Correctness
# -----------------------------------------------------------------------------
def run_correctness(args):
    rank, world_size = _init()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    all_sum = sum(range(1, world_size + 1))
    all_ok = True

    if rank == 0:
        print("\n========== Correctness Tests (force Simple protocol) ==========")
        print(f"fp32 pack size: {BYTES_PER_PACK} bytes = {FLOATS_PER_PACK} floats")

    size = 1024 * 1024  # 1MB fp32

    cases = [
        ("mask=0|1 halves", torch.cat([torch.zeros(size // 2), torch.ones(size // 2)])),
        ("mask=00001111", torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).repeat(size // 8)),
        ("mask=0000111100001111", torch.tensor([0.0]*4 + [1.0]*4).repeat(size // 8)),
        ("mask=0000", torch.zeros(size)),
        ("mask=1111", torch.ones(size)),
    ]

    for name, mask_cpu in cases:
        mask = mask_cpu.to(device, dtype=torch.float32)
        tensor = torch.ones(size, device=device, dtype=torch.float32) * (rank + 1)
        expected = torch.where(
            mask.cpu() == 1.0,
            torch.full((size,), float(all_sum), dtype=torch.float32),
            torch.full((size,), float(rank + 1), dtype=torch.float32),
        )
        mask_all_reduce(tensor, mask)
        torch.cuda.synchronize(device)
        all_ok &= _check(rank, tensor, expected, name)
        del tensor, mask, expected
        free_memory()

    # Vanilla no-mask sanity check using the original upstream NCCL library.
    tensor = torch.ones(size, device=device, dtype=torch.float32) * (rank + 1)
    expected = torch.full((size,), float(all_sum), dtype=torch.float32)
    vanilla_all_reduce(tensor)
    torch.cuda.synchronize(device)
    all_ok &= _check(rank, tensor, expected, "no mask (vanilla lib)")

    dist.destroy_process_group()
    if rank == 0:
        print("\n=== Correctness: ALL PASSED ===" if all_ok else "\n=== Correctness: FAILED ===")
    sys.exit(0 if all_ok else 1)


# -----------------------------------------------------------------------------
# Step 2: Performance (size x sparsity x distribution x version)
# -----------------------------------------------------------------------------
DEFAULT_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_SPARSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_DISTRIBUTIONS = ["block", "alternating", "mixed", "random"]


def _parse_size(s):
    """Parse a size string like '1MB', '2GB', or a raw integer."""
    s = s.strip().upper()
    if s.endswith("GB"):
        return int(float(s[:-2]) * 1024 * 1024 * 1024 / 4)
    elif s.endswith("MB"):
        return int(float(s[:-2]) * 1024 * 1024 / 4)
    elif s.endswith("KB"):
        return int(float(s[:-2]) * 1024 / 4)
    else:
        return int(s)


def _parse_size_list(s):
    return [_parse_size(x) for x in s.split(",")]


def _parse_sparsity_list(s):
    return [float(x) for x in s.split(",")]


def _parse_distribution_list(s):
    return [x.strip() for x in s.split(",")]


def _no_mask_baseline(size, device, warmup, iters):
    """Measure no-mask latency with the fixed upstream NCCL baseline."""
    tensor = torch.ones(size, device=device, dtype=torch.float32)
    t = timed_all_reduce(tensor, mask=None, version="nomask", warmup=warmup, iters=iters)
    bytes_size = size * 4
    bw = (bytes_size / 1e9) / (t / 1000)
    del tensor
    free_memory()
    return t, bw


def _no_mask_version(size, device, version, warmup, iters):
    """Measure no-mask latency through a mask library version (mask=None)."""
    tensor = torch.ones(size, device=device, dtype=torch.float32)
    t = timed_all_reduce(tensor, mask=None, version=version, warmup=warmup, iters=iters)
    bytes_size = size * 4
    bw = (bytes_size / 1e9) / (t / 1000)
    del tensor
    free_memory()
    return t, bw


def _verify_mask(rank, world_size, tensor, mask, version):
    all_sum = sum(range(1, world_size + 1))
    n = tensor.numel()
    mask_cpu = mask.cpu()
    # Expand mask to pack-level: if any element in a 4-float pack is 1, the whole pack is reduced
    padded = torch.nn.functional.pad(mask_cpu, (0, (-n) % FLOATS_PER_PACK))
    pack_mask = padded.view(-1, FLOATS_PER_PACK).any(dim=1).float()
    expanded = pack_mask.repeat_interleave(FLOATS_PER_PACK)[:n]
    expected = torch.where(
        expanded == 1.0,
        torch.full((n,), float(all_sum), dtype=torch.float32),
        torch.full((n,), float(rank + 1), dtype=torch.float32),
    )
    return torch.allclose(tensor.cpu(), expected, rtol=1e-5, atol=1e-5)


def run_perf(args):
    rank, world_size = _init()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    sizes = args.sizes
    sparsities = args.sparsities
    distributions = args.distributions
    versions = list_mask_versions()

    if not versions:
        if rank == 0:
            print("ERROR: no mask versions found in /workspace/nccl/libs/v*/")
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        print("\n========== Performance Sweep (size x sparsity x distribution x version) ==========")
        print(f"sizes: {[s * 4 // (1024*1024) for s in sizes]} MB")
        print(f"sparsities: {sparsities}")
        print(f"distributions: {distributions}")
        print(f"versions: {versions}")
        print(f"warmup={args.warmup}, iters={args.iters}")
        print("")
        print(f"{'size(MB)':>8}  {'sparsity':>8}  {'distribution':<12}  {'real_sparsity':>13}  {'version':>7}  "
              f"{'no_mask_baseline(ms)':>20}  {'no_mask(ms)':>11}  {'mask(ms)':>8}  {'speedup':>7}  {'vs_prev':>7}  "
              f"{'no_mask_vs_baseline':>20}  {'no_mask(GB/s)':>13}  {'mask(GB/s)':>10}  {'check':>5}")

    for size in sizes:
        bytes_size = size * 4
        mb = bytes_size / (1024 * 1024)

        # Fixed upstream NCCL no-mask baseline, measured once per size.
        t_baseline, bw_baseline = _no_mask_baseline(size, device, args.warmup, args.iters)
        # Per-version no-mask result (mask=None), lazily measured once per size.
        version_no_mask = {}

        for target_sparsity in sparsities:
            for pattern in distributions:
                mask, actual = generate_mask(size, target_sparsity, pattern=pattern, device=device)
                prev_t_mask = None
                prev_version = None

                for version in versions:
                    if version not in version_no_mask:
                        version_no_mask[version] = _no_mask_version(
                            size, device, version, args.warmup, args.iters
                        )
                    t_no_version, bw_no_version = version_no_mask[version]

                    tensor = torch.ones(size, device=device, dtype=torch.float32) * (rank + 1)
                    t_mask = timed_all_reduce(tensor, mask=mask, version=version, warmup=args.warmup, iters=args.iters)
                    bw_mask = (bytes_size / 1e9) / (t_mask / 1000)

                    # Verify only the active part (mask=1) equals the full all-reduce sum.
                    with torch.no_grad():
                        check_tensor = torch.ones(size, device=device, dtype=torch.float32) * (rank + 1)
                        mask_all_reduce(check_tensor, mask, version=version)
                        torch.cuda.synchronize(device)
                        ok = _verify_mask(rank, world_size, check_tensor, mask, version)

                    if rank == 0:
                        speedup = t_baseline / t_mask if t_mask > 0 else float('inf')
                        if prev_t_mask is not None and prev_t_mask > 0:
                            vs_prev = prev_t_mask / t_mask
                            vs_prev_str = f"{vs_prev:6.2f}x"
                        else:
                            vs_prev_str = "      -"
                        no_mask_vs_baseline = t_baseline / t_no_version if t_baseline > 0 else float('inf')
                        check_str = "OK" if ok else "FAIL"
                        print(f"{mb:8.0f}  {target_sparsity*100:7.0f}%  {pattern:12s}  {actual*100:12.1f}%  "
                              f"{version:7s}  {t_baseline:20.3f}  {t_no_version:11.3f}  {t_mask:8.3f}  {speedup:6.2f}x  {vs_prev_str}  "
                              f"{no_mask_vs_baseline:19.2f}x  {bw_baseline:13.2f}  {bw_mask:10.2f}  {check_str:>5s}")

                    prev_t_mask = t_mask
                    prev_version = version

                    del tensor, check_tensor
                    free_memory()

                del mask
                free_memory()

    dist.destroy_process_group()


def run_prof(args):
    """Profile mode: warmup runs untraced, only iters are captured by nsys."""
    rank, world_size = _init()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    sizes = args.sizes
    sparsities = args.sparsities
    distributions = args.distributions
    versions = list_mask_versions()

    if not versions:
        if rank == 0:
            print("ERROR: no mask versions found in /workspace/nccl/libs/v*/")
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        print("\n========== Profile Mode ==========")
        print(f"sizes: {[s * 4 // (1024*1024) for s in sizes]} MB")
        print(f"sparsities: {sparsities}")
        print(f"distributions: {distributions}")
        print(f"versions: {versions}")
        print(f"warmup={args.warmup} (untraced), iters={args.iters} (profiled)")
        print("")

    for size in sizes:
        for target_sparsity in sparsities:
            for pattern in distributions:
                mask, actual = generate_mask(size, target_sparsity, pattern=pattern, device=device)

                for version in versions:
                    # Warmup: not profiled
                    for _ in range(args.warmup):
                        t = torch.ones(size, device=device, dtype=torch.float32) * (rank + 1)
                        mask_all_reduce(t, mask, version=version)
                        del t
                    torch.cuda.synchronize(device)

                    if rank == 0:
                        mb = size * 4 / (1024 * 1024)
                        print(f"Profiling: size={mb:.0f}MB sparsity={target_sparsity} "
                              f"dist={pattern} ver={version}")

                    # Signal nsys to start capturing
                    torch.cuda.cudart().cudaProfilerStart()

                    for _ in range(args.iters):
                        t = torch.ones(size, device=device, dtype=torch.float32) * (rank + 1)
                        mask_all_reduce(t, mask, version=version)
                        del t
                    torch.cuda.synchronize(device)

                    # Signal nsys to stop capturing
                    torch.cuda.cudart().cudaProfilerStop()

                    free_memory()

                del mask
                free_memory()

    # Also profile no-mask baseline for comparison
    if rank == 0:
        print("\nProfiling: no-mask baseline")

    for size in sizes:
        tensor = torch.ones(size, device=device, dtype=torch.float32)
        for _ in range(args.warmup):
            vanilla_all_reduce(tensor.clone())
        torch.cuda.synchronize(device)

        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.iters):
            vanilla_all_reduce(tensor.clone())
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()

        del tensor
        free_memory()

    dist.destroy_process_group()
    if rank == 0:
        print("\nProfile complete. Check nsys output for results.")


def main():
    parser = argparse.ArgumentParser(description="Mask all-reduce correctness and performance tests")
    parser.add_argument("subcommand", choices=["correctness", "perf", "prof"])
    parser.add_argument(
        "--sizes",
        type=_parse_size_list,
        default=None,
        help="Comma-separated sizes, e.g. '1MB,2MB,4MB' or raw fp32 counts. Default: 1MB..4GB",
    )
    parser.add_argument(
        "--sparsities",
        type=_parse_sparsity_list,
        default=None,
        help="Comma-separated target sparsities, e.g. '0.1,0.5,0.9'. Default: 0.1..0.8",
    )
    parser.add_argument(
        "--distributions",
        type=_parse_distribution_list,
        default=None,
        help="Comma-separated distributions, e.g. 'block,alternating'. Default: all",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations")
    args = parser.parse_args()

    # prof requires explicit sizes/sparsities/distributions.
    if args.subcommand == "prof":
        missing = []
        if args.sizes is None:
            missing.append("--sizes")
        if args.sparsities is None:
            missing.append("--sparsities")
        if args.distributions is None:
            missing.append("--distributions")
        if missing:
            parser.error(f"prof subcommand requires: {', '.join(missing)}")

    # Apply defaults for perf (prof already validated above).
    if args.sizes is None:
        args.sizes = [mb * 1024 * 1024 // 4 for mb in DEFAULT_SIZES_MB]
    if args.sparsities is None:
        args.sparsities = DEFAULT_SPARSITIES
    if args.distributions is None:
        args.distributions = DEFAULT_DISTRIBUTIONS

    if args.subcommand == "correctness":
        run_correctness(args)
    elif args.subcommand == "perf":
        run_perf(args)
    elif args.subcommand == "prof":
        run_prof(args)


if __name__ == "__main__":
    main()
