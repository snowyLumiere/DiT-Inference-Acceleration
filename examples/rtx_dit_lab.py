#!/usr/bin/env python3
"""Small PyTorch teaching experiments; NOT a complete DiT or quality benchmark.

Example (the writable output parent must already exist):
  python rtx_dit_lab.py --output "$DIT_LAB_ROOT/outputs/lab.json" --case all \
      --device cuda:0 --warmup 5 --samples 30
No downloads, torch.compile, trace files, directory creation or CPU GPU-fallback.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time
import uuid
import warnings

import torch
import torch.nn.functional as F


def summarize(values):
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("observations must be nonempty, finite and nonnegative")
    ordered = sorted(values)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": ordered[0], "max": ordered[-1], "p05": percentile(.05), "p95": percentile(.95),
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0}


def numerical(reference, candidate):
    reference, candidate = reference.detach().cpu().contiguous(), candidate.detach().cpu().contiguous()
    if reference.shape != candidate.shape:
        raise ValueError("paired output shapes differ")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    bits = reference.dtype == candidate.dtype and torch.equal(
        reference.view(torch.uint8), candidate.view(torch.uint8))
    if not finite:
        return {"finite": False, "bitwise": bool(bits), "max_abs": None, "relative_l2": None}
    left, right = reference.double(), candidate.double()
    difference = right - left
    denominator, distance = float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(difference))
    relative = distance / denominator if denominator else (0.0 if distance == 0 else None)
    return {"finite": True, "bitwise": bool(bits), "max_abs": float(difference.abs().max()),
            "relative_l2": relative, "relative_l2_defined": relative is not None}


def attention_acceptance(comparisons, *, flash_executed):
    """Accept only this seeded BF16 fixture; not a model or perceptual quality gate.

    Both limits are deliberately explicit: relative L2 <= 0.5% and maximum
    absolute error <= 0.005, against FP32 and the same-dtype MATH baseline.
    A missing FLASH backend is a skipped comparison, never a numerical pass.
    """
    limits = {"max_abs": 0.005, "relative_l2": 0.005}
    baseline = ("A1_vs_fp32_reference", "A2_vs_fp32_reference", "A2_vs_A1")
    required = baseline + (("B_vs_fp32_reference", "B_vs_A1") if flash_executed else ())
    checks = {}
    for name in required:
        comparison = comparisons.get(name, {})
        checks[name] = comparison.get("finite") is True and all(
            type(comparison.get(metric)) in (int, float)
            and math.isfinite(comparison[metric])
            and 0 <= comparison[metric] <= limit
            for metric, limit in limits.items())
    repeat_bitwise = comparisons.get("A2_vs_A1", {}).get("bitwise") is True
    baseline_passed = all(checks[name] for name in baseline) and repeat_bitwise
    if not baseline_passed or (flash_executed and not all(checks.values())):
        status = "failed"
    else:
        status = "complete" if flash_executed else "skipped"
    return {"status": status, "passed": status == "complete", "limits": limits,
            "checks": checks, "A2_vs_A1_bitwise": repeat_bitwise,
            "baseline_passed": baseline_passed, "flash_assessed": flash_executed,
            "scope": "Fixed seeded BF16 attention fixture only; no full-model or perceptual quality assessment."}


def descriptor(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
            "stride": list(tensor.stride()), "bytes": tensor.numel() * tensor.element_size()}


def tensor_sha256(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()


def _timing_record(warm, hot, *, inner, cuda):
    return {"warmup": warm, "hot": hot, "iterations_per_observation": inner,
            "units": "milliseconds per call, observation batch divided by iterations",
            "wall_scope": "host launch + event record + final device synchronization" if cuda else "CPU wall",
            "event_scope": "CUDA stream event span, including launch gaps; not pure kernel time" if cuda else None,
            "summary": {"wall_ms": summarize([v["wall_ms"] for v in hot]),
                        "cuda_event_ms": summarize([v["cuda_event_ms"] for v in hot]) if cuda else None}}


def validate_timing(record, *, samples, warmup, cuda):
    """Validate stored observations and summaries, independently of a speed claim."""
    if samples < 30 or len(record["hot"]) != samples or len(record["warmup"]) != warmup:
        raise ValueError("requires all requested warmup and at least 30 hot observations")
    if type(record["iterations_per_observation"]) is not int or record["iterations_per_observation"] < 1:
        raise ValueError("invalid iterations_per_observation")
    for rows in (record["warmup"], record["hot"]):
        summarize([v["wall_ms"] for v in rows])
        if cuda:
            summarize([v["cuda_event_ms"] for v in rows])
        elif any(v["cuda_event_ms"] is not None for v in rows):
            raise ValueError("CPU observations must not claim CUDA timings")
    for key in ("wall_ms", "cuda_event_ms"):
        if key == "cuda_event_ms" and not cuda:
            if record["summary"][key] is not None:
                raise ValueError("CPU event summary must be null")
            continue
        expected = summarize([v[key] for v in record["hot"]])
        if any(not math.isclose(record["summary"][key][k], value, rel_tol=1e-10, abs_tol=1e-12)
               for k, value in expected.items()):
            raise ValueError("timing summary disagrees with raw observations")


def measure(fn, *, device, warmup, samples, inner, deadline):
    cuda = device.type == "cuda"
    if cuda:
        if torch.cuda.current_device() != device.index:
            raise ValueError("timed CUDA device must be the current device")
        stream = torch.cuda.current_stream(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream); end.record(stream); torch.cuda.synchronize(device)
    observations, output = [], None
    for _ in range(warmup + samples):
        if time.perf_counter() > deadline:
            raise RuntimeError("110-second experiment budget reached; no incomplete run is accepted")
        if cuda:
            torch.cuda.synchronize(device)
        before = time.perf_counter()
        if cuda:
            start.record(stream)
        for _ in range(inner):
            output = fn()
        if cuda:
            end.record(stream)
            torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - before) * 1000 / inner
        observations.append({"wall_ms": wall_ms, "cuda_event_ms": start.elapsed_time(end) / inner if cuda else None})
    record = _timing_record(observations[:warmup], observations[warmup:], inner=inner, cuda=cuda)
    validate_timing(record, samples=samples, warmup=warmup, cuda=cuda)
    return record, output.detach().cpu().clone()


def aba_statistics(arms):
    if any(arm["status"] != "complete" for arm in arms):
        return {"available": False, "reason": "candidate skipped; no paired speed ratio"}
    result = {"available": True}
    for key in ("wall_ms", "cuda_event_ms"):
        if arms[0]["timing"]["summary"][key] is None:
            result[key] = None
            continue
        a1, b, a2 = [[v[key] for v in arm["timing"]["hot"]] for arm in arms]
        if len({len(a1), len(b), len(a2)}) != 1 or min(b + a1 + a2) <= 0:
            raise ValueError("paired positive timing series required")
        reference = [(x + y) / 2 for x, y in zip(a1, a2)]
        result[key] = {"mean_A_over_B": sum(reference) / sum(b), "A2_over_A1": sum(a2) / sum(a1),
                       "paired_A_over_B": [x / y for x, y in zip(reference, b)]}
    return result


class RawKVHistory:
    """Bounded raw clones and chronological indices; no position/camera semantics."""
    def __init__(self, capacity=5, mode="list"):
        if type(capacity) is not int or capacity < 1 or mode not in ("list", "ring"):
            raise ValueError("positive capacity and list/ring mode required")
        self.capacity, self.mode = capacity, mode
        self.reset()

    def reset(self):
        self.next_index, self.spec = 0, None
        self._list, self._ring = [], [None] * self.capacity

    def __len__(self):
        return min(self.next_index, self.capacity)

    def history(self, chunk_index):
        if type(chunk_index) is not int or chunk_index != self.next_index:
            raise ValueError("history requires the next uncommitted index")
        indices = list(range(max(0, chunk_index - self.capacity), chunk_index))
        entries = self._list if self.mode == "list" else [self._ring[i % self.capacity] for i in indices]
        if [entry[0] for entry in entries] != indices:
            raise RuntimeError("raw history chronology violated")
        return tuple(entries)

    def commit(self, chunk_index, key, value):
        self.history(chunk_index)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError("raw Tensor keys and values required")
        spec = (tuple(key.shape), key.dtype, key.device)
        if (tuple(value.shape), value.dtype, value.device) != spec or self.spec not in (None, spec):
            raise ValueError("raw KV shape/dtype/device must stay fixed")
        entry = (chunk_index, key.detach().clone(), value.detach().clone())
        self.spec = spec
        if self.mode == "list":
            self._list.append(entry)
            if len(self._list) > self.capacity:
                self._list.pop(0)
        else:
            self._ring[chunk_index % self.capacity] = entry
        self.next_index += 1


def _arm(name, backend, timing):
    return {"name": name, "status": "complete", "backend": backend, "timing": timing}


def attention_case(device, options):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    generator = torch.Generator(device="cpu").manual_seed(42)
    q, k, v = [torch.randn(1, 8, 512, 64, generator=generator).to(device, torch.bfloat16) for _ in range(3)]
    qf, kf, vf = [x.float() for x in (q, k, v)]

    def explicit():
        return torch.softmax((qf @ kf.transpose(-2, -1)) / math.sqrt(qf.shape[-1]), dim=-1) @ vf

    def sdpa():
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    reference_timing, reference = measure(explicit, device=device, **options)
    arms, outputs = [], {}
    for name, backend, expected in (("A1", SDPBackend.MATH, "_scaled_dot_product_attention_math"),
                                    ("B", SDPBackend.FLASH_ATTENTION, "_scaled_dot_product_flash_attention"),
                                    ("A2", SDPBackend.MATH, "_scaled_dot_product_attention_math")):
        with sdpa_kernel([backend]):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    sdpa(); torch.cuda.synchronize(device)
            except (RuntimeError, NotImplementedError) as exc:
                if name != "B" or isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                arms.append({"name": name, "status": "skipped", "backend_requested": "FLASH_ATTENTION only",
                             "reason": str(exc), "warnings": [str(w.message) for w in caught]})
                continue
            # CPU operator tracing suffices to identify dispatch; no CUPTI or trace file.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                sdpa(); torch.cuda.synchronize(device)
            operators = sorted({event.key for event in profile.key_averages() if "scaled_dot_product" in event.key})
            if not any(expected in operator for operator in operators):
                raise RuntimeError(f"Backend dispatch not verified: {backend}: {operators}")
            timing, output = measure(sdpa, device=device, **options)
        arm = _arm(name, str(backend), timing)
        arm["actual_dispatch_operators"] = operators
        arms.append(arm); outputs[name] = output
    comparisons = {name + "_vs_fp32_reference": numerical(reference, output) for name, output in outputs.items()}
    comparisons["A2_vs_A1"] = numerical(outputs["A1"], outputs["A2"])
    if "B" in outputs:
        comparisons["B_vs_A1"] = numerical(outputs["A1"], outputs["B"])
    acceptance = attention_acceptance(comparisons, flash_executed="B" in outputs)
    return {"name": "attention", "status": acceptance["status"], "device": str(device),
            "tensors": {"q": descriptor(q), "k": descriptor(k), "v": descriptor(v), "q_fp32": descriptor(qf),
                        "reference_output_cpu_snapshot": descriptor(reference), "sdpa_output_cpu_snapshot": descriptor(outputs["A1"])},
            "input_sha256": {name: tensor_sha256(t) for name, t in zip(("q", "k", "v"), (q, k, v))},
            "fp32_reference": {"formula": "softmax(Q.float() @ K.float().T / sqrt(D)) @ V.float()",
                               "timing": reference_timing, "role": "numerical reference; not a same-dtype acceleration baseline"},
            "dropout_p": 0.0, "mask": None, "is_causal": False, "arms": arms,
            "fused_status": "executed" if "B" in outputs else "skipped_no_fallback",
            "comparisons": comparisons, "numerical_gate": acceptance, "paired_timing": aba_statistics(arms),
            "scope": "Single attention operator, fixed BF16 inputs; only MATH vs FLASH_ATTENTION changes; no complete DiT."}


def linear_fixture(width=512, depth=8, batch=64):
    """Seeded CPU fixture with version counters, including under an outer inference context."""
    generator = torch.Generator(device="cpu").manual_seed(43)
    with torch.inference_mode(False), torch.no_grad():
        model = torch.nn.Sequential(*[layer for _ in range(depth) for layer in (torch.nn.Linear(width, width), torch.nn.SiLU())])
        for parameter in model.parameters():
            parameter.normal_(0, 1 / math.sqrt(width), generator=generator)
        inputs = torch.randn(batch, width, generator=generator)
    return model.eval().requires_grad_(False), inputs


def graph_case(device, options):
    model, inputs = linear_fixture()
    model = model.to(device=device, dtype=torch.bfloat16)
    static_input = inputs.to(device, torch.bfloat16)
    output_buffer = torch.empty_like(static_input)
    versions = [p._version for p in model.parameters()]
    input_version = static_input._version
    output_pointer = output_buffer.data_ptr()

    def eager():
        output_buffer.copy_(model(static_input))
        return output_buffer

    a1_timing, a1 = measure(eager, device=device, **options)
    before = time.perf_counter()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(options["warmup"]):
            eager()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        eager()
    torch.cuda.synchronize(device)
    capture_seconds = time.perf_counter() - before

    def replay():
        graph.replay()
        return output_buffer

    b_timing, b = measure(replay, device=device, **options)
    # A stale output left by eager must not pass as a successful graph replay.
    output_buffer.fill_(float("nan")); replay(); torch.cuda.synchronize(device)
    overwrite = numerical(a1, output_buffer)
    a2_timing, a2 = measure(eager, device=device, **options)
    stable = versions == [p._version for p in model.parameters()] and static_input._version == input_version
    arms = [_arm("A1", "eager", a1_timing), _arm("B", "cuda_graph_replay", b_timing), _arm("A2", "eager", a2_timing)]
    comparisons = {"B_vs_A1": numerical(a1, b), "A2_vs_A1": numerical(a1, a2), "poison_overwrite_vs_A1": overwrite}
    passed = stable and output_buffer.data_ptr() == output_pointer and all(x["finite"] and x["bitwise"] for x in comparisons.values())
    return {"name": "graph", "status": "complete" if passed else "failed", "device": str(device),
            "tensors": {"input": descriptor(static_input), "output_buffer": descriptor(output_buffer),
                        "parameters": [descriptor(p) for p in model.parameters()]},
            "input_sha256": tensor_sha256(static_input), "architecture": "8 x (Linear(512,512,bias=True) + SiLU)",
            "output_buffer_reused": output_buffer.data_ptr() == output_pointer,
            "weights_and_input_versions_unchanged": stable, "capture_and_side_stream_warmup_seconds_excluded": capture_seconds,
            "arms": arms, "comparisons": comparisons, "paired_timing": aba_statistics(arms),
            "scope": "Fixed weights/input/addresses; output copy_ included in eager and captured graph. No dynamic KV, full DiT or request E2E."}


def kv_case(_device, options):
    cpu = torch.device("cpu")
    keys = [torch.arange(256, dtype=torch.float32).reshape(1, 2, 8, 16) + i for i in range(12)]
    values = [key * .5 + 3 for key in keys]
    arms, outputs, cleared = [], {}, []
    for name, mode in (("A1", "list"), ("B", "ring"), ("A2", "list")):
        cache = RawKVHistory(5, mode)

        def epoch():
            cache.reset()
            pieces = []
            try:
                for index, (key, value) in enumerate(zip(keys, values)):
                    for _, raw_key, raw_value in cache.history(index):
                        pieces.extend((raw_key.reshape(-1), raw_value.reshape(-1)))
                    cache.commit(index, key, value)
                for _, raw_key, raw_value in cache.history(len(keys)):
                    pieces.extend((raw_key.reshape(-1), raw_value.reshape(-1)))
                return torch.cat(pieces)
            finally:
                cache.reset()

        timing, output = measure(epoch, device=cpu, **options)
        cleared.append(len(cache) == 0 and cache.history(0) == ())
        outputs[name] = output; arms.append(_arm(name, mode, timing))
    comparisons = {name + "_vs_A1": numerical(outputs["A1"], outputs[name]) for name in ("B", "A2")}
    passed = all(cleared) and all(x["finite"] and x["bitwise"] for x in comparisons.values())
    return {"name": "kv", "status": "complete" if passed else "failed", "device": "cpu",
            "tensors": {"raw_key_per_commit": descriptor(keys[0]), "raw_value_per_commit": descriptor(values[0]),
                        "history_payload_trace": descriptor(outputs["A1"])},
            "capacity_entries": 5, "commits_per_epoch": 12,
            "retained_raw_storage_after_commit_upper_bound_bytes": 5 * 2 * keys[0].numel() * 4,
            "commit_temporary_raw_clone_bytes": 2 * keys[0].numel() * 4,
            "reset_cleared_each_arm": cleared, "arms": arms, "comparisons": comparisons,
            "paired_timing": aba_statistics(arms), "performance_optimization_claimed": False,
            "scope": "CPU data-structure contract only: raw clones, chronological reads, commit/evict/reset. No model-specific RoPE, camera, refresh or denoising semantics."}


def physical_gpu(device, record_hardware_id=False):
    properties = torch.cuda.get_device_properties(device)
    raw = getattr(properties, "uuid", None)
    method = "torch.cuda.get_device_properties.uuid"
    if raw is None:
        import ctypes
        class CUuuid(ctypes.Structure):
            _fields_ = [("value", ctypes.c_ubyte * 16)]
        driver = ctypes.CDLL("libcuda.so.1")
        index, value = ctypes.c_int(), CUuuid()
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
        get_uuid.argtypes = [ctypes.POINTER(CUuuid), ctypes.c_int]
        if driver.cuDeviceGet(ctypes.byref(index), device.index) or get_uuid(ctypes.byref(value), index):
            raise RuntimeError("Cannot verify physical CUDA device UUID")
        raw, method = bytes(value.value), "CUDA driver visible ordinal -> UUID"
    canonical = uuid.UUID(bytes=raw) if isinstance(raw, bytes) else uuid.UUID(str(raw).lower().removeprefix("gpu-"))
    identifier = "GPU-" + str(canonical)
    result = {"name": properties.name,
              "visible_device": str(device), "compute_capability": [properties.major, properties.minor],
              "total_memory_bytes": properties.total_memory}
    if record_hardware_id:
        result.update(uuid=identifier, uuid_source=method)
    try:
        inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version,pci.bus_id",
                                             "--format=csv,noheader,nounits"], text=True, timeout=10)
        rows = [line.split(",") for line in inventory.splitlines()]
        match = next(row for row in rows if row[0].strip().lower() == identifier.lower())
        result["driver_version"] = match[1].strip()
        if record_hardware_id:
            result["pci_bus_id"] = match[2].strip()
    except (OSError, subprocess.SubprocessError, StopIteration, IndexError) as exc:
        result["driver_inventory_unavailable"] = type(exc).__name__
    return result


def preflight_storage(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError("output exists; refusing overwrite")
    if not output.parent.is_dir():
        raise FileNotFoundError("output parent must already exist; no directory is created")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ValueError("output parent must be an existing writable directory")
    free = shutil.disk_usage(parent).free
    if free < 2**30:
        raise ValueError("retain at least 1 GiB free on the selected filesystem")
    return {"resolved_parent": str(parent), "free_bytes_before_run": free}


def write_receipt(path, receipt):
    payload = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(payload)


def run_lab(args, device, storage):
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    gpu = physical_gpu(device, record_hardware_id=args.record_hardware_id)
    if "RTX" not in gpu["name"] and not getattr(args, "allow_non_rtx", False):
        raise RuntimeError("This run is scoped to an RTX GPU; refusing a different GPU family")
    receipt = {"schema": "rtx-dit-lab.v1", "checked_at": datetime.now(timezone.utc).isoformat(),
               "tutorial_only": True, "complete_dit": False, "quality_assessed": False, "e2e_speedup_claimed": False,
               "source_sha256": source_sha, "seed_attention": 42, "seed_graph": 43, "storage": storage,
               "software": {"torch": torch.__version__, "python": sys.version, "cuda_runtime": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version(), "allow_tf32": False,
                            "float32_matmul_precision": "highest"}, "physical_gpu": gpu,
               "environment_paths": {name: os.environ.get(name) for name in ("TMPDIR", "CUDA_CACHE_PATH", "TORCH_HOME", "XDG_CACHE_HOME")},
               "hardware_ids_recorded": args.record_hardware_id,
               "samples_per_arm": args.samples, "warmup_per_arm": args.warmup, "iterations_per_observation": args.inner,
               "hot_timing_excludes": ["tensor/model setup", "CPU output snapshots", "backend probe/profiler",
                                       "graph side-stream warmup/capture", "poison overwrite validation", "numerical/hash audits", "receipt writing"],
               "api_references": ["https://docs.pytorch.org/docs/2.8/generated/torch.nn.attention.sdpa_kernel.html",
                                   "https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs"],
               "cases": [], "limitations": ["Small fixed-shape operators and a CPU storage contract only.",
                   "A/B/A is one sequential run; samples are not independent hardware/scene repetitions.",
                   "The output parent is checked; cache/temp environment paths are the operator's responsibility."]}
    if args.record_hardware_id:
        receipt["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    started = time.perf_counter()
    options = dict(warmup=args.warmup, samples=args.samples, inner=args.inner, deadline=started + 110)
    functions = {"attention": attention_case, "graph": graph_case, "kv": kv_case}
    names = list(functions) if args.case == "all" else [args.case]
    for name in names:
        try:
            if name != "kv":
                torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            result = functions[name](device, options)
            result["source_sha256"] = source_sha
            if args.record_hardware_id and name != "kv":
                result["physical_gpu_uuid"] = gpu["uuid"]
            peak = torch.cuda.max_memory_allocated(device) if name != "kv" else None
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["memory_peak"] = {"cuda_allocated_bytes": peak,
                                     "cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if name != "kv" else None,
                                     "cuda_allocation_target_bytes": 2 * 1024 ** 3,
                                     "cuda_target_met": peak < 2 * 1024 ** 3 if peak is not None else None,
                                     "process_peak_rss_cumulative_bytes": rss if sys.platform == "darwin" else rss * 1024,
                                     "rss_scope": "process cumulative high-water mark, not a per-case resettable peak",
                                     "cuda_scope": "stats reset before case; includes setup/audit and any preexisting allocator pool"}
            receipt["cases"].append(result)
        except Exception as exc:
            receipt["cases"].append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            break
    receipt["experiment_wall_seconds_including_setup_and_audits"] = time.perf_counter() - started
    receipt["skipped_arms"] = [{"case": c["name"], "arm": arm["name"], "reason": arm["reason"]}
                                for c in receipt["cases"] for arm in c.get("arms", []) if arm["status"] == "skipped"]
    receipt["technical_passed"] = len(receipt["cases"]) == len(names) and all(c["status"] == "complete" for c in receipt["cases"])
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", choices=("all", "attention", "graph", "kv"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-non-rtx", action="store_true",
                        help="allow non-RTX GPUs, including data-center devices")
    parser.add_argument("--record-hardware-id", action="store_true",
                        help="include GPU UUID, PCI bus ID and CUDA_VISIBLE_DEVICES in the local receipt")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--inner", type=int, default=10, help="calls in each timed observation")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        storage = preflight_storage(args.output)
        if not (1 <= args.warmup <= 20 and 30 <= args.samples <= 120 and 1 <= args.inner <= 50):
            raise ValueError("require warmup1..20, samples30..120, inner1..50")
        device = torch.device(args.device)
        if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA device and available GPU required; CPU-only KV tests are separate")
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # The fixed-input/weight audit needs real _version counters.
        with torch.no_grad():
            receipt = run_lab(args, device, storage)
        write_receipt(args.output, receipt)
        print(json.dumps({"output": str(args.output), "technical_passed": receipt["technical_passed"],
                          "quality_assessed": False, "skipped_arms": receipt["skipped_arms"],
                          "receipt_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))
        return 0 if receipt["technical_passed"] else 1
    except Exception as exc:
        print(json.dumps({"technical_passed": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
