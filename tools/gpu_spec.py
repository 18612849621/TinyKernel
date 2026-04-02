"""
tools/gpu_spec.py — print GPU hardware specs and theoretical compute/bandwidth.

Compute throughput is derived from SM count × ops-per-SM-per-clock × boost clock.
Per-SM ops/clock by dtype are architecture constants (Ada=sm89, Hopper=sm90, Ampere=sm80/86).
Memory bandwidth = 2 × mem_clock × bus_width / 8.
"""

import subprocess
import torch


# ── architecture tables ───────────────────────────────────────────────────────
# (major, minor) -> ops per SM per clock for each dtype (FMA counts as 2 ops)
# Sources: NVIDIA Ada/Hopper/Ampere whitepapers
_OPS_PER_SM = {
    # Ada Lovelace  sm89
    (8, 9): {"fp32": 128, "tf32": 512, "fp16": 256, "bf16": 256, "fp8": 512, "int8": 512, "fp64": None},
    # Ampere A100   sm80
    (8, 0): {"fp32": 64,  "tf32": 512, "fp16": 256, "bf16": 256, "fp8": None, "int8": 512, "fp64": 64},
    # Ampere GA10x  sm86
    (8, 6): {"fp32": 128, "tf32": 512, "fp16": 256, "bf16": 256, "fp8": None, "int8": 512, "fp64": None},
    # Hopper        sm90
    (9, 0): {"fp32": 128, "tf32": 512, "fp16": 256, "bf16": 256, "fp8": 1024, "int8": 1024, "fp64": 64},
    # Turing        sm75
    (7, 5): {"fp32": 64,  "tf32": None,"fp16": 128, "bf16": None,"fp8": None, "int8": 256,  "fp64": None},
}

# Tensor Core supported formats per architecture (for display)
_TC_FORMATS = {
    (8, 9): "fp16, bf16, tf32, fp8 (e4m3/e5m2), int8",
    (8, 0): "fp16, bf16, tf32, fp64, int8",
    (8, 6): "fp16, bf16, tf32, int8",
    (9, 0): "fp16, bf16, tf32, fp8 (e4m3/e5m2), fp64, int8",
    (7, 5): "fp16, int8, int4, int1",
    (7, 0): "fp16",
}

# bus width (bits) by GPU name substring
_BUS_WIDTH = {
    "4090": 384, "4080": 256, "4070 Ti": 192, "4070": 192, "4060": 128,
    "3090": 384, "3080": 320, "3070": 256, "3060": 192,
    "A100": 5120, "H100 SXM": 5120, "H100 PCIe": 4096,
    "A6000": 384, "A5000": 256, "A4000": 256,
}


def _bus_width(name: str) -> int | None:
    for k, v in _BUS_WIDTH.items():
        if k in name:
            return v
    return None


def _smi(fields: list[str]) -> list[str]:
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(", ")
    return out


def main():
    p = torch.cuda.get_device_properties(0)
    name = p.name
    sm_count = p.multi_processor_count
    cap = (p.major, p.minor)
    vram_gb = p.total_memory / 1024**3
    l2_mb = p.L2_cache_size / 1024**2
    smem_kb = p.shared_memory_per_multiprocessor / 1024

    # clocks from nvidia-smi
    try:
        sm_mhz, mem_mhz = _smi(["clocks.max.sm", "clocks.max.memory"])
        sm_mhz, mem_mhz = int(sm_mhz), int(mem_mhz)
    except Exception:
        sm_mhz = mem_mhz = None

    bus = _bus_width(name)

    print(f"{'─'*52}")
    print(f"  {name}")
    print(f"{'─'*52}")
    print(f"  CUDA compute cap   {cap[0]}.{cap[1]}")
    print(f"  SM count           {sm_count}")
    print(f"  VRAM               {vram_gb:.1f} GB")
    print(f"  L2 cache           {l2_mb:.0f} MB")
    print(f"  Shared mem / SM    {smem_kb:.0f} KB")
    if sm_mhz:
        print(f"  Boost clock        {sm_mhz} MHz")
    if mem_mhz and bus:
        bw = 2 * mem_mhz * 1e6 * bus / 8 / 1e9
        print(f"  Mem clock          {mem_mhz} MHz  ({bus}-bit bus)")
        print(f"  Mem bandwidth      {bw:.0f} GB/s  (theoretical)")

    ops = _OPS_PER_SM.get(cap)
    tc_formats = _TC_FORMATS.get(cap)
    if tc_formats:
        print(f"\n  Tensor Core formats:  {tc_formats}")
    if ops and sm_mhz:
        print(f"\n  Theoretical throughput (TFLOPS / TOPS):")
        for dtype, ops_per_sm in ops.items():
            if ops_per_sm is None:
                print(f"    {dtype:<6}  —")
            else:
                tflops = sm_count * ops_per_sm * sm_mhz * 1e6 / 1e12
                print(f"    {dtype:<6}  {tflops:.1f}")
    elif not ops:
        print(f"\n  (no ops table for sm{cap[0]}{cap[1]})")

    print(f"{'─'*52}")


if __name__ == "__main__":
    main()
