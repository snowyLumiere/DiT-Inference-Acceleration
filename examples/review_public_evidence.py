#!/usr/bin/env python3
"""Recompute the reader edition's measurements with Python's standard library.

This checks saved observations, not original CUDA tensors or perceptual quality.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

DEFAULT = Path(__file__).resolve().parents[1] / 'docs/manuscript/evidence/measurements.json'


def close(actual, expected, label):
    if type(actual) not in (int, float) or not math.isfinite(actual):
        raise ValueError(label + ': invalid number')
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(label + ': recomputation mismatch')


def observations(timing, field, count=30):
    rows = timing['hot']
    if len(rows) != count or len(timing['warmup']) != 5:
        raise ValueError('requires all 5 warmup and all recorded hot observations')
    values = [r[field] for r in rows]
    warmup = [r[field] for r in timing['warmup']]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in values + warmup):
        raise ValueError('nonpositive or nonfinite observations')
    mean = statistics.mean(values)
    ordered = sorted(values)
    def percentile(p):
        pos = (len(ordered) - 1) * p
        low, high = math.floor(pos), math.ceil(pos)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
    expected = {'count': count, 'mean': mean, 'median': statistics.median(values),
                'min': min(values), 'max': max(values), 'p05': percentile(.05),
                'p95': percentile(.95), 'sample_stddev': statistics.stdev(values)}
    saved = timing['summary'][field]
    if 'cv' in saved:
        expected['cv'] = expected['sample_stddev'] / mean
    for key, value in expected.items():
        close(saved[key], value, field + ' ' + key)
    return mean


def recompute_ar(data):
    required = {'schema', 'benchmark_time', 'scope', 'quality_assessed',
                'production_e2e_claimed', 'provenance', 'hardware', 'software', 'results'}
    if set(data) != required:
        raise ValueError('unexpected AR measurement sections')
    if data['quality_assessed'] is not False or data['production_e2e_claimed'] is not False:
        raise ValueError('AR aggregates do not establish model quality or production E2E')
    provenance = data['provenance']
    if (not re.fullmatch(r'[0-9a-f]{64}', provenance.get('original_receipt_sha256', ''))
            or provenance.get('hot_observations_available') is not False
            or provenance.get('summary_only') is not True):
        raise ValueError('AR provenance must state summary-only scope and receipt hash')
    derived = provenance.get('derived_fields')
    if (not isinstance(derived, dict)
            or 'results.kv_cache.geometry.gather_output_bf16_bytes' not in derived):
        raise ValueError('AR export annotations must be declared in provenance.derived_fields')
    hardware = data['hardware']
    if (hardware['compute_capability'] != [9, 0]
            or hardware['benchmark_device_count'] != 1
            or hardware['benchmark_device'] != 'cuda:0'
            or hardware['visible_device_count'] < hardware['benchmark_device_count']):
        raise ValueError('AR benchmark device scope mismatch')
    results = data['results']
    if set(results) != {'kv_cache'}:
        raise ValueError('missing or unexpected AR case')
    kv = results['kv_cache']
    geometry = kv['geometry']
    dimensions = [kv[k] for k in ('num_layers', 'num_heads', 'head_dim',
                                  'tokens_per_chunk', 'window_size')]
    dimensions += [geometry[k] for k in ('block_size', 'pool_chunk_capacity')]
    if any(type(n) is not int or n <= 0 for n in dimensions):
        raise ValueError('invalid AR KV geometry')
    layers, heads, dim, chunk, window, block, capacity = dimensions
    if capacity < window:
        raise ValueError('AR pool smaller than active window')
    pages = math.ceil(chunk / block)
    slots = capacity * pages * block
    active_bytes = 2 * layers * window * chunk * heads * dim * 2
    elements = 2 * layers * slots * heads * dim
    scale_bytes = 2 * layers * capacity * pages * heads * 4
    for key, expected in {'blocks_per_chunk': pages, 'padded_tokens_per_chunk': pages * block,
                          'active_tokens': chunk * window, 'pool_token_slots': slots,
                          'active_bf16_payload_bytes': active_bytes,
                          'gather_output_bf16_bytes': active_bytes,
                          'fp8_scale_bytes': scale_bytes,
                          'fp8_pool_and_scale_bytes': elements + scale_bytes}.items():
        close(geometry[key], expected, 'KV ' + key)
    close(kv['vram_bf16_bytes'], elements * 2, 'KV BF16 pool payload')
    close(kv['vram_fp8_bytes'], elements, 'KV FP8 pool payload')
    close(kv['vram_savings_pct'], 50.0, 'KV payload saving')
    close(kv['paged_bf16_speedup'], kv['list_gather_ms'] / kv['paged_bf16_gather_ms'],
          'KV BF16 gather ratio')
    close(kv['paged_fp8_speedup'], kv['list_gather_ms'] / kv['paged_fp8_gather_ms'],
          'KV FP8 gather ratio')
    return {'passed': True, 'scope': '50-layer AR KV pool geometry and gather summary verification',
            'raw_hot_observations_available': False,
            'active_kv_bytes': active_bytes, 'gather_output_bf16_bytes': active_bytes,
            'pool_token_slots': slots,
            'bf16_pool_payload_bytes': elements * 2,
            'fp8_pool_and_scale_bytes': elements + scale_bytes,
            'gather_ratios': {'paged_bf16_vs_list': kv['paged_bf16_speedup'],
                              'paged_fp8_vs_list': kv['paged_fp8_speedup']}}


def recompute(data):
    if data['schema'] == 'dit.reader-ar-microbench.v1':
        return recompute_ar(data)
    if data['schema'] != 'dit.reader-measurements.v2':
        raise ValueError('wrong schema')
    allowed = {'schema', 'scope', 'export_note', 'quality_assessed',
               'production_e2e_claimed', 'rtx_lab', 'kv_transfer'}
    if set(data) != allowed:
        raise ValueError('unexpected measurement sections')
    if data['quality_assessed'] or data['production_e2e_claimed']:
        raise ValueError('saved numbers do not establish quality or production E2E')
    output = []
    for section in ('rtx_lab', 'kv_transfer'):
        metadata = data[section]
        required = {'original_receipt_sha256', 'source_sha256', 'technical_passed',
                    'software', 'hardware', 'cases'}
        if section == 'kv_transfer':
            required |= {'warmup_per_arm', 'samples_per_arm', 'setup_seconds', 'memory_peak'}
        if not required <= set(metadata):
            raise ValueError('missing required measurement metadata: ' + section)
        for key in ('original_receipt_sha256', 'source_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', metadata[key]):
                raise ValueError('invalid source or receipt hash')
        if metadata['technical_passed'] is not True:
            raise ValueError('measurement technical gate did not pass')
        if section == 'kv_transfer':
            if metadata['warmup_per_arm'] != 5 or metadata['samples_per_arm'] != 30:
                raise ValueError('wrong transfer observation contract')
            if not metadata['setup_seconds'] or not metadata['memory_peak']:
                raise ValueError('missing transfer setup or memory evidence')
        expected_cases = ({'attention', 'graph', 'kv'} if section == 'rtx_lab'
                          else {'pinning', 'submission', 'pin_each_read_negative'})
        cases = data[section]['cases']
        if len(cases) != len(expected_cases) or {c['name'] for c in cases} != expected_cases:
            raise ValueError('missing or duplicate measurement case')
        for case in data[section]['cases']:
            arms = {a['name']: a for a in case['arms']}
            if len(case['arms']) != 3 or set(arms) != {'A1', 'B', 'A2'}:
                raise ValueError('missing paired arm')
            for field in ('wall_ms', 'cuda_event_ms'):
                paired = case['paired_timing'].get(field)
                if paired is None:
                    if section == 'rtx_lab' and case['name'] == 'kv' and field == 'cuda_event_ms':
                        continue
                    raise ValueError('missing required timing metric')
                means = {k: observations(a['timing'], field) for k, a in arms.items()}
                ratio = (means['A1'] + means['A2']) / (2 * means['B'])
                drift = means['A2'] / means['A1']
                close(paired['mean_A_over_B'], ratio, case['name'] + ' ratio')
                close(paired['A2_over_A1'], drift, case['name'] + ' drift')
                if section == 'kv_transfer' or 'A2_drift_fraction' in paired:
                    close(paired['A2_drift_fraction'], drift - 1, case['name'] + ' drift fraction')
                ratios = paired['paired_A_over_B']
                if len(ratios) != 30:
                    raise ValueError('missing paired observations')
                for index, value in enumerate(ratios):
                    times = {k: a['timing']['hot'][index][field] for k, a in arms.items()}
                    close(value, (times['A1'] + times['A2']) / (2 * times['B']),
                          case['name'] + ' paired ratio')
                output.append({'case': section + '/' + case['name'], 'field': field,
                               'means': means, 'A_over_B': ratio, 'A2_over_A1': drift})
    payload = data['kv_transfer']['payload']
    shape = payload['shape_per_raw_k_or_v']
    if (len(shape) != 4 or any(type(x) is not int or x <= 0 for x in shape)
            or payload['dtype'] != 'torch.bfloat16'
            or type(payload['tensors_per_epoch']) is not int or payload['tensors_per_epoch'] <= 0):
        raise ValueError('invalid BF16 KV shape')
    per_tensor = math.prod(shape) * 2
    if per_tensor != payload['bytes_per_tensor']:
        raise ValueError('KV shape/byte mismatch')
    epoch = per_tensor * payload['tensors_per_epoch']
    if epoch != payload['h2d_payload_bytes_per_epoch']:
        raise ValueError('KV epoch byte mismatch')
    bandwidth = []
    for case in data['kv_transfer']['cases']:
        means = {a['name']: observations(a['timing'], 'wall_ms') for a in case['arms']}
        reference_ms = (means['A1'] + means['A2']) / 2
        bandwidth.append({'case': case['name'],
                          'reference_effective_GBps': epoch / (reference_ms * 1e6),
                          'candidate_effective_GBps': epoch / (means['B'] * 1e6),
                          'scope': 'Payload bytes / host wall time, including submission and synchronization'})
    graph = next(c for c in data['rtx_lab']['cases'] if c['name'] == 'graph')
    means = {a['name']: observations(a['timing'], 'wall_ms') for a in graph['arms']}
    saving = (means['A1'] + means['A2']) / 2 - means['B']
    capture_ms = graph['capture_and_side_stream_warmup_seconds_excluded'] * 1000
    return {'passed': True, 'scope': 'Saved statistics and geometry only; no GPU replay or quality assessment',
            'synthetic': output, 'kv_epoch_bytes': epoch, 'transfer_bandwidth': bandwidth,
            'graph_amortization': {'capture_and_warmup_ms': capture_ms,
                                   'wall_saving_ms_per_call': saving,
                                   'break_even_calls': math.ceil(capture_ms / saving) if saving > 0 else None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    try:
        result = recompute(json.loads(args.input.read_text()))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        parser.exit(1, f'Evidence verification failed: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
