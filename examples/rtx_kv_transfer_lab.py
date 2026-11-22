#!/usr/bin/env python3
"""Synthetic raw-KV H2D transport A/B/A without model weights.

  python3 -B rtx_kv_transfer_lab.py --fixture tiny \
      --output-dir "$DIT_LAB_ROOT/outputs/transfer-r1" --device cuda:0

The output parent must already be writable and have at least 1 GiB free.
A new output directory and '<run>.runtime' sibling are reserved.
--fixture large uses a teaching shape. To repeat a saved explicit geometry,
use --fixture explicit-shape --shape B N H D. Every fixture requires CUDA.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time
import uuid


SHAPES = {"large": (1, 4096, 32, 128), "tiny": (1, 16, 2, 8)}
HISTORY_CHUNKS = 5
TENSORS_PER_EPOCH = 2 * HISTORY_CHUNKS
GIB = 2**30
MODES = {"pageable_blocking": (False, False), "pinned_blocking": (True, False),
         "pinned_nonblocking": (True, True), "pin_each_read": (False, False)}


def payload_spec(fixture, explicit_shape=None):
    if fixture == "explicit-shape":
        if (not isinstance(explicit_shape, (list, tuple)) or len(explicit_shape) != 4
                or any(type(value) is not int or value <= 0 for value in explicit_shape)):
            raise ValueError("explicit-shape requires four positive integers: B N H D")
        shape = tuple(explicit_shape)
    else:
        if explicit_shape is not None:
            raise ValueError("--shape is valid only with --fixture explicit-shape")
        shape = SHAPES[fixture]
    per_tensor = math.prod(shape) * 2
    return {"fixture": fixture, "shape_per_raw_k_or_v": list(shape), "dtype": "torch.bfloat16",
            "history_chunks": HISTORY_CHUNKS, "tensors_per_epoch": TENSORS_PER_EPOCH,
            "bytes_per_tensor": per_tensor, "h2d_payload_bytes_per_epoch": per_tensor * TENSORS_PER_EPOCH,
            "order": [f"chunk{i}.{kind}" for i in range(HISTORY_CHUNKS) for kind in ("K", "V")],
            "real_model_shape": False,
            "provenance": {"kind": "deterministic synthetic values; no model tensors or weights",
                           "raw_layout": "[batch, tokens, heads, head_dim], contiguous BF16",
                           "derivation": "bytes_per_epoch = B * N * H * D * 2 bytes * 10 tensors"}}


def summarize(values):
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("timings must be nonempty, finite and positive")
    ordered = sorted(values)
    def percentile(p):
        index = (len(ordered) - 1) * p
        lo = int(index)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (index - lo)
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "mean": mean, "median": statistics.median(values),
            "min": ordered[0], "max": ordered[-1], "p05": percentile(.05), "p95": percentile(.95),
            "sample_stddev": deviation, "cv": deviation / mean}


def validate_timing(record, warmup, samples):
    if samples < 30 or len(record["hot"]) != samples or len(record["warmup"]) != warmup:
        raise ValueError("all requested warmup and at least 30 hot observations required")
    if record["epochs_per_observation"] != 1 or record["copies_per_epoch"] != TENSORS_PER_EPOCH:
        raise ValueError("one complete ten-copy epoch per observation required")
    for phase in ("warmup", "hot"):
        for metric in ("wall_ms", "cuda_event_ms"):
            stats = summarize([row[metric] for row in record[phase]])
            if phase == "hot" and stats != record["summary"][metric]:
                raise ValueError("summary disagrees with raw observations")


def aba_statistics(arms):
    result = {}
    for metric in ("wall_ms", "cuda_event_ms"):
        a1, b, a2 = [[row[metric] for row in arm["timing"]["hot"]] for arm in arms]
        if len({len(a1), len(b), len(a2)}) != 1:
            raise ValueError("A/B/A sample counts differ")
        for series in (a1, b, a2):
            summarize(series)
        baseline = [(x + y) / 2 for x, y in zip(a1, a2)]
        result[metric] = {"mean_A_over_B": statistics.mean(baseline) / statistics.mean(b),
                          "A2_over_A1": statistics.mean(a2) / statistics.mean(a1),
                          "A2_drift_fraction": statistics.mean(a2) / statistics.mean(a1) - 1,
                          "paired_A_over_B": [x / y for x, y in zip(baseline, b)],
                          "formula": "mean((A1_i+A2_i)/2) / mean(B_i); sequential A/B/A, not independent runs"}
    return result


def validate_options(args):
    if not re.fullmatch(r"cuda:\d+", args.device):
        raise ValueError("explicit CUDA device required; no CPU benchmark fallback")
    if not (1 <= args.warmup <= 20 and 30 <= args.samples <= 120 and 1 <= args.cpu_threads <= 16):
        raise ValueError("require warmup1..20, samples30..120, cpu_threads1..16")
    payload_spec(args.fixture, args.shape)
    if args.pin_each_read and args.case != "all":
        raise ValueError("--pin-each-read adds the negative case only with --case all")
    if not args.output_dir.is_absolute() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", args.output_dir.name):
        raise ValueError("absolute output directory with a simple unique run ID required")


def preflight_storage(output):
    runtime = output.with_name(output.name + ".runtime")
    if any(p.exists() or p.is_symlink() for p in (output, runtime)):
        raise FileExistsError("output/runtime exists; choose a new run ID; refusing overwrite")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ValueError("output parent must be an existing writable directory")
    free = shutil.disk_usage(parent).free
    if free < 2**30:
        raise ValueError("retain at least 1 GiB free on the selected filesystem")
    return {"resolved_parent": str(parent), "free_bytes_before_run": free}


def prepare_runtime(path):
    path.mkdir(exist_ok=False)
    locations = {"XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch", "CUDA_CACHE_PATH": "cuda",
                 "TORCH_EXTENSIONS_DIR": "extensions", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                 "PYTORCH_KERNEL_CACHE_PATH": "kernels", "TRITON_CACHE_DIR": "triton",
                 "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "PYTHONPYCACHEPREFIX": "pycache"}
    for key, suffix in locations.items():
        destination = path / suffix
        destination.mkdir(exist_ok=True)
        os.environ[key] = str(destination)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    return {key: str(path / suffix) for key, suffix in locations.items()}


def host_memory_gate(payload_bytes, include_negative, proc=Path("/proc"), cgroup=Path("/sys/fs/cgroup")):
    info = dict(line.split(":", 1) for line in (proc / "meminfo").read_text().splitlines())
    host = int(info["MemAvailable"].split()[0]) * 1024
    candidates = {cgroup, cgroup / "memory"}
    for line in (proc / "self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            base = cgroup if not controllers else cgroup / "memory"
            current = base / relative.lstrip("/")
            if ".." in current.parts:
                continue
            while current != base and base in current.parents:
                candidates.add(current); current = current.parent
    constraints = []
    for base in sorted(candidates):
        names = (("memory.max", "memory.current", "memory.stat", "inactive_file"),
                 ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.stat", "total_inactive_file"))
        for limit_name, current_name, stat_name, inactive_key in names:
            if not (base / limit_name).exists():
                continue
            raw = (base / limit_name).read_text().strip()
            if raw == "max" or int(raw) >= 2**60:
                continue
            limit, current = int(raw), int((base / current_name).read_text())
            stats = dict(line.split() for line in (base / stat_name).read_text().splitlines())
            inactive = min(current, max(0, int(stats.get(inactive_key, stats.get("inactive_file", 0)))))
            available = max(0, limit - current + inactive)
            constraints.append({"path": str(base), "limit": limit, "current": current,
                                "reclaimable_inactive_file": inactive, "available": available})
    available = min([host] + [x["available"] for x in constraints])
    required = (4 if include_negative else 3) * payload_bytes + 2 * GIB
    if available < required:
        raise RuntimeError(f"insufficient effective host RAM: {available} < {required} bytes")
    return {"host_mem_available_bytes": host, "cgroups": constraints, "effective_available_bytes": available,
            "required_bytes": required, "budget": "3*payload +2GiB; 4*payload with pin-each-read; cgroup inactive_file credit bounded by current"}


def require_cuda(torch, label):
    device = torch.device(label)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA device and available GPU required; no CPU timings emitted")
    torch.cuda.set_device(device)
    return device


def gpu_gate(torch, device, payload_bytes, record_hardware_id=False, allow_non_rtx=False):
    properties = torch.cuda.get_device_properties(device)
    raw = getattr(properties, "uuid", None)
    method = "torch.cuda.get_device_properties.uuid"
    if raw is None:
        import ctypes
        class CUuuid(ctypes.Structure):
            _fields_ = [("value", ctypes.c_ubyte * 16)]
        driver, ordinal, value = ctypes.CDLL("libcuda.so.1"), ctypes.c_int(), CUuuid()
        get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
        if driver.cuDeviceGet(ctypes.byref(ordinal), device.index) or get_uuid(ctypes.byref(value), ordinal):
            raise RuntimeError("physical CUDA UUID unavailable")
        raw, method = bytes(value.value), "CUDA driver visible ordinal"
    identifier = "GPU-" + str(uuid.UUID(bytes=raw) if isinstance(raw, bytes) else uuid.UUID(str(raw).lower().removeprefix("gpu-")))
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10)
    rows = [[v.strip() for v in line.split(",")] for line in inventory.splitlines()]
    match = next(row for row in rows if row[1].lower() == identifier.lower())
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True, timeout=10)
    processes = [line.strip() for line in applications.splitlines() if line.split(",")[0].strip().lower() == identifier.lower()]
    foreign = [line for line in processes if int(line.split(",")[1]) != os.getpid()]
    free, total = torch.cuda.mem_get_info(device)
    if ("RTX" not in properties.name and not allow_non_rtx) or foreign or int(match[5]) > 1024 or int(match[6]) > 5:
        raise RuntimeError("selected GPU is not an idle RTX; require no other compute processes, "
                           "at most 1024 MiB used and at most 5% utilization")
    required = max(2 * GIB, 2 * payload_bytes + 512 * 2**20)
    if free < required:
        raise RuntimeError(f"insufficient selected GPU free memory: {free} < {required}")
    result = {"physical_index": int(match[0]), "name": properties.name,
            "driver_version": match[4], "logical_device": str(device),
            "compute_capability": [properties.major, properties.minor], "free_bytes_before_fixture": free,
            "total_bytes": total, "required_free_bytes": required, "used_MiB_at_gate": int(match[5]),
            "utilization_percent_at_gate": int(match[6]),
            "compute_processes_at_gate": [
                {"pid": int(row.split(",")[1].strip()),
                 "used_gpu_memory_MiB": row.split(",")[2].strip()}
                for row in processes]}
    if record_hardware_id:
        result.update(uuid=identifier, uuid_source=method, pci_bus_id=match[3],
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return result


def make_pageable_sources(torch, shape, seed=5040):
    if (not isinstance(shape, (list, tuple)) or len(shape) != 4
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise ValueError("source shape requires four positive integers: B N H D")
    shape = tuple(shape)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode(False), torch.no_grad():
        return [torch.randn(shape, dtype=torch.bfloat16, generator=generator) for _ in range(TENSORS_PER_EPOCH)]


def tensor_sha256(tensor):
    # numpy exposes the CPU uint8 storage without a huge Python list/copy.
    return hashlib.sha256(memoryview(tensor.detach().contiguous().view(-1).view(__import__("torch").uint8).numpy())).hexdigest()


def tensor_state(tensors):
    return [{"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device),
             "stride": list(t.stride()), "data_ptr": t.data_ptr(), "version": t._version} for t in tensors]


def validate_states(before, after, expected_version_delta=0):
    if len(before) != len(after):
        raise RuntimeError("tensor list changed")
    for left, right in zip(before, after):
        if any(left[key] != right[key] for key in left if key != "version") or right["version"] != left["version"] + expected_version_delta:
            raise RuntimeError("tensor address/shape/dtype/device or real version counter changed unexpectedly")


def compare_cpu(torch, reference, candidate):
    if reference.device.type != "cpu" or candidate.device.type != "cpu" or reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("same-shape/dtype CPU tensors required")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    bits = torch.equal(reference.contiguous().view(torch.uint8), candidate.contiguous().view(torch.uint8))
    result = {"finite": finite, "bitwise": bits, "max_abs": None, "relative_l2": None}
    if not finite:
        return result
    if bits:
        return {**result, "max_abs": 0.0, "relative_l2": 0.0}
    distance, denominator, maximum = 0.0, 0.0, 0.0
    for left, right in zip(reference.reshape(-1).split(2**20), candidate.reshape(-1).split(2**20)):
        left, right = left.double(), right.double()
        delta = right - left
        distance += float(delta.square().sum()); denominator += float(left.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {**result, "max_abs": maximum,
            "relative_l2": math.sqrt(distance / denominator) if denominator else (0.0 if distance == 0 else None)}


def copy_epoch(pageable, pinned, targets, mode):
    source = pinned if MODES[mode][0] else pageable
    for item, target in zip(source, targets):
        if mode == "pin_each_read":
            item = item.pin_memory()  # includes CPU allocation/copy; blocking H2D keeps lifetime safe.
        target.copy_(item, non_blocking=MODES[mode][1])


def measure(torch, device, fn, warmup, samples, deadline, arm):
    if torch.cuda.current_device() != device.index:
        raise RuntimeError("timed CUDA device must be current")
    setup_started = time.perf_counter()
    stream = torch.cuda.current_stream(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream); end.record(stream); torch.cuda.synchronize(device)
    timing = {"warmup": [], "hot": [], "epochs_per_observation": 1,
              "copies_per_epoch": TENSORS_PER_EPOCH, "units": "milliseconds per full ten-copy epoch",
              "wall_scope": "ten H2D copy_ submissions + event recording + final selected-device synchronization; pin_each_read includes pinning",
              "event_scope": "current selected-device stream span including host launch/pin gaps; NOT pure DMA or a compute overlap measure",
              "cuda_stream": stream.cuda_stream,
              "event_setup_and_prime_seconds_excluded": time.perf_counter() - setup_started,
              "summary": None}
    arm["timing"] = timing  # retain partial observations if a later timeout or audit fails.
    with torch.cuda.stream(stream):
        for index in range(warmup + samples):
            if time.perf_counter() > deadline:
                raise RuntimeError("110-second experiment budget exceeded; incomplete result rejected")
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            start.record(stream)
            fn()
            end.record(stream)
            torch.cuda.synchronize(device)
            phase = "warmup" if index < warmup else "hot"
            timing[phase].append({"observation": index, "wall_ms": (time.perf_counter() - before) * 1000,
                                  "cuda_event_ms": start.elapsed_time(end)})
    timing["summary"] = {metric: summarize([row[metric] for row in timing["hot"]])
                         for metric in ("wall_ms", "cuda_event_ms")}
    validate_timing(timing, warmup, samples)
    return timing


def audit_outputs(torch, pageable, targets):
    return [compare_cpu(torch, reference, target.detach().cpu()) for reference, target in zip(pageable, targets)]


def run_case(torch, device, pageable, pinned, targets, modes, args, deadline, payload_bytes, result):
    fixed = tensor_state(pageable + pinned)
    arms = result["arms"]
    for name, mode in zip(("A1", "B", "A2"), modes):
        arm = {"name": name, "mode": mode, "status": "running"}
        arms.append(arm)
        initial = tensor_state(targets)
        for tensor in targets:
            tensor.fill_(float("nan"))
        poisoned = all(bool(torch.isnan(t).all()) for t in targets)
        copy_epoch(pageable, pinned, targets, mode)
        torch.cuda.synchronize(device)
        overwrite = audit_outputs(torch, pageable, targets)
        validate_states(initial, tensor_state(targets), 2)
        hot_start = tensor_state(targets)
        timing = measure(torch, device, lambda: copy_epoch(pageable, pinned, targets, mode),
                         args.warmup, args.samples, deadline, arm)
        final = tensor_state(targets)
        validate_states(hot_start, final, args.warmup + args.samples)
        validate_states(fixed, tensor_state(pageable + pinned))
        after = audit_outputs(torch, pageable, targets)
        passed = poisoned and all(row["finite"] and row["bitwise"] for row in overwrite + after)
        if not passed:
            raise RuntimeError("full-byte transfer/NaN poison overwrite check failed")
        arm.update({"status": "complete", "source_pinned": MODES[mode][0] or mode == "pin_each_read",
                     "non_blocking": MODES[mode][1], "pin_cost_in_wall": mode == "pin_each_read", "timing": timing,
                     "effective_payload_GB_per_s": {key: payload_bytes / (timing["summary"][key]["mean"] * 1e6)
                                                     for key in ("wall_ms", "cuda_event_ms")},
                     "poison_all_elements_nan": poisoned, "poison_overwrite_comparisons": overwrite,
                     "post_hot_comparisons": after, "source_real_version_counters_unchanged": True,
                     "destination_addresses_stable": True, "destination_versions_before_observations": [x["version"] for x in hot_start],
                     "destination_versions_after_observations": [x["version"] for x in final],
                     "expected_destination_version_delta": args.warmup + args.samples})
    result.update({"status": "complete", "paired_timing": aba_statistics(arms),
                   "numerical_gate": True, "max_abs_across_arms": 0.0, "relative_l2_across_arms": 0.0,
                   "all_arms_bitwise_equal_via_shared_cpu_reference": True,
                   "overlap_assessed": False, "e2e_speedup_claimed": False})


def run_lab(torch, device, args, receipt):
    started = time.perf_counter(); deadline = started + 110
    spec = receipt["payload"]
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    setup = time.perf_counter()
    pageable = make_pageable_sources(torch, spec["shape_per_raw_k_or_v"], receipt["seed"])
    receipt["setup_seconds"] = {"pageable_allocate_and_fill": time.perf_counter() - setup}
    setup = time.perf_counter(); pinned = [t.pin_memory() for t in pageable]
    receipt["setup_seconds"]["persistent_pin_and_copy"] = time.perf_counter() - setup
    setup = time.perf_counter(); targets = [torch.empty_like(t, device=device) for t in pageable]
    torch.cuda.synchronize(device)
    receipt["setup_seconds"]["gpu_destination_allocate_and_sync"] = time.perf_counter() - setup
    if any(t.is_pinned() for t in pageable) or not all(t.is_pinned() for t in pinned):
        raise RuntimeError("pageable/pinned fixture storage mismatch")
    source_checks = [compare_cpu(torch, left, right) for left, right in zip(pageable, pinned)]
    if any(not (row["finite"] and row["bitwise"]) for row in source_checks):
        raise RuntimeError("pinning changed source contents")
    receipt["source_pageable_vs_pinned_comparisons"] = source_checks
    receipt["tensors"] = {"pageable_sources": tensor_state(pageable), "pinned_sources": tensor_state(pinned),
                          "shared_gpu_destinations": tensor_state(targets)}
    source_hashes = [tensor_sha256(t) for t in pageable]
    receipt["source_tensor_sha256"] = source_hashes
    receipt["cases"] = []
    cases = [("pinning", ("pageable_blocking", "pinned_blocking", "pageable_blocking")),
             ("submission", ("pinned_blocking", "pinned_nonblocking", "pinned_blocking"))]
    if args.pin_each_read:
        cases.append(("pin_each_read_negative", ("pinned_blocking", "pin_each_read", "pinned_blocking")))
    for name, modes in cases:
        if args.case not in ("all", name):
            continue
        result = {"name": name, "status": "running", "arms": []}
        receipt["cases"].append(result)
        run_case(torch, device, pageable, pinned, targets, modes, args, deadline,
                 spec["h2d_payload_bytes_per_epoch"], result)
    if source_hashes != [tensor_sha256(t) for t in pageable] or source_hashes != [tensor_sha256(t) for t in pinned]:
        raise RuntimeError("source contents changed across A/B/A")
    receipt["source_hashes_unchanged_after_all_arms"] = True
    peak = torch.cuda.max_memory_allocated(device)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt["memory_peak"] = {"cuda_allocated_bytes": peak, "cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                              "cuda_target_bytes": 2 * GIB, "cuda_target_met": peak < 2 * GIB,
                              "host_peak_rss_cumulative_bytes": rss if sys.platform == "darwin" else rss * 1024,
                              "host_rss_scope": "process cumulative high-water mark; includes setup/validation and pinned allocator caching"}
    if not receipt["cases"] or peak >= 2 * GIB:
        raise RuntimeError("no completed case or 2GiB GPU allocation budget exceeded")
    elapsed = time.perf_counter() - started
    receipt["experiment_seconds_including_setup_audits"] = elapsed
    if elapsed > 110:
        raise RuntimeError("110-second budget exceeded including setup and final CPU audits; complete samples retained, run rejected")
    receipt["technical_passed"] = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-hardware-id", action="store_true",
                        help="include GPU UUID, PCI bus ID and CUDA_VISIBLE_DEVICES in the local receipt")
    parser.add_argument("--fixture", choices=tuple(SHAPES) + ("explicit-shape",), default="large")
    parser.add_argument("--shape", nargs=4, type=int, metavar=("B", "N", "H", "D"),
                        help="four positive dimensions; requires --fixture explicit-shape")
    parser.add_argument("--case", choices=("all", "pinning", "submission"), default="all")
    parser.add_argument("--pin-each-read", action="store_true", help="add pinning-in-timed-read negative A/B/A")
    parser.add_argument("--allow-non-rtx", action="store_true", help="allow non-RTX GPUs, including data-center devices")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    receipt = None
    try:
        validate_options(args)
        storage = preflight_storage(args.output_dir)
        payload = payload_spec(args.fixture, args.shape)
        host = host_memory_gate(payload["h2d_payload_bytes_per_epoch"], args.pin_each_read)
        runtime = args.output_dir.with_name(args.output_dir.name + ".runtime")
        args.output_dir.mkdir(exist_ok=False)
        receipt = {"schema": "rtx-kv-transfer-lab.v1", "run_id": args.output_dir.name,
                   "checked_at": datetime.now(timezone.utc).isoformat(), "technical_passed": False,
                   "quality_assessed": False, "e2e_speedup_claimed": False, "compute_overlap_assessed": False,
                   "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "payload": payload, "seed": 5040, "storage": storage, "host_memory_gate": host,
                   "hardware_ids_recorded": args.record_hardware_id,
                   "warmup_per_arm": args.warmup, "samples_per_arm": args.samples,
                   "experiment_budget_seconds": 110,
                   "budget_scope": "fixture setup through all observations, full CPU audits, hashes and cleanup; cooperative checks between operations; import/preflight/receipt IO excluded",
                   "output_file": str(args.output_dir / "receipt.json"),
                   "scope": "synthetic raw-KV H2D only; same CPU values and preallocated GPU targets; no model weights/attention/denoise/PRoPE/VAE",
                   "hot_excludes": ["persistent pin/setup/allocation", "event initialization/priming", "NaN poison/overwrite audit", "CPU numerical/bitwise/hash audits",
                                    "real version/address checks", "receipt IO", "pre-observation synchronization"],
                   "limitations": ["No compute is issued in timed regions: nonblocking submission is NOT measured overlap.",
                                   "Pageable transfers may use driver staging; reported GB/s is payload/epoch time, not bus peak bandwidth.",
                                   "Sequential A/B/A samples are not independent machine/scene repeats.",
                                   "Sources are deterministic synthetic values at the selected dimensions; no saved model KV contents."]}
        receipt["runtime_paths"] = prepare_runtime(runtime)
        torch = importlib.import_module("torch")  # runtime/cache paths established before importing CUDA/PyTorch.
        device = require_cuda(torch, args.device)
        receipt["physical_gpu"] = gpu_gate(torch, device, payload["h2d_payload_bytes_per_epoch"],
                                           record_hardware_id=args.record_hardware_id,
                                           allow_non_rtx=getattr(args, "allow_non_rtx", False))
        receipt["software"] = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                               "python": sys.version, "cpu_threads": args.cpu_threads}
        with torch.inference_mode(False), torch.no_grad():
            lab_started = time.perf_counter()
            try:
                run_lab(torch, device, args, receipt)
            finally:
                receipt["experiment_seconds_including_setup_audits"] = time.perf_counter() - lab_started
            if receipt["experiment_seconds_including_setup_audits"] > 110:
                receipt["technical_passed"] = False
                raise RuntimeError("110-second experiment budget exceeded; all collected samples retained, run rejected")
    except Exception as exc:
        if receipt is None:
            print(json.dumps({"technical_passed": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
            return 2
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        for case in receipt.get("cases", []):
            if case["status"] == "running":
                case["status"] = "failed"
            for arm in case["arms"]:
                if arm["status"] == "running":
                    arm["status"] = "failed"
    output = args.output_dir / "receipt.json"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "technical_passed": receipt["technical_passed"],
                      "quality_assessed": False, "receipt_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
    return 0 if receipt["technical_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
