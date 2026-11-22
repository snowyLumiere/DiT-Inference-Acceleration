import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('reader_evidence', Path(__file__).with_name('review_public_evidence.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReaderEvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(module.DEFAULT.read_text())

    def test_all_recorded_observations_recompute(self):
        report = module.recompute(self.data)
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['transfer_bandwidth']), 3)
        self.assertEqual(report['kv_epoch_bytes'], 722534400)

    def test_sm90_recorded_observations_recompute(self):
        report = module.recompute(json.loads(module.DEFAULT.with_name('measurements_sm90.json').read_text()))
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['transfer_bandwidth']), 3)
        self.assertEqual(report['kv_epoch_bytes'], 722534400)
        self.assertEqual(report['graph_amortization']['break_even_calls'], 285)

    def ar_data(self):
        return json.loads(module.DEFAULT.with_name('ar_microbench_sm90.json').read_text())

    def test_ar_geometry_and_gather_ratios(self):
        report = module.recompute(self.ar_data())
        self.assertTrue(report['passed'])
        self.assertEqual(report['active_kv_bytes'], 8601600000)
        self.assertEqual(report['gather_output_bf16_bytes'], 8601600000)
        self.assertEqual(report['pool_token_slots'], 8512)
        self.assertEqual(report['bf16_pool_payload_bytes'], 12202803200)
        self.assertEqual(report['fp8_pool_and_scale_bytes'], 6104380800)
        self.assertFalse(report['raw_hot_observations_available'])

    def test_ar_missing_case_and_wrong_pool_geometry_fail(self):
        for mutation in ('case', 'padding', 'scale'):
            data = self.ar_data()
            if mutation == 'case':
                data['results']['extra_unverified'] = {}
            elif mutation == 'padding':
                data['results']['kv_cache']['geometry']['pool_token_slots'] = 6000
            else:
                data['results']['kv_cache']['geometry']['fp8_scale_bytes'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.recompute(data)

    def test_ar_invalid_scope_flags_fail(self):
        for key, value in (
                ('quality_assessed', True),
                ('production_e2e_claimed', True)):
            data = self.ar_data()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.recompute(data)

    def test_ar_undeclared_export_annotation_fails(self):
        data = self.ar_data()
        del data['provenance']['derived_fields']
        with self.assertRaises(ValueError):
            module.recompute(data)

    def test_missing_transfer_metadata_fails_for_both_devices(self):
        for name in ('measurements.json', 'measurements_sm90.json'):
            for key in ('source_sha256', 'memory_peak', 'samples_per_arm',
                        'warmup_per_arm', 'setup_seconds', 'technical_passed'):
                data = json.loads(module.DEFAULT.with_name(name).read_text())
                del data['kv_transfer'][key]
                with self.subTest(name=name, key=key), self.assertRaisesRegex(ValueError, 'metadata'):
                    module.recompute(data)

    def test_corrupt_percentiles_and_paired_ratios_fail(self):
        for mutation in ('p95', 'paired_ratio'):
            data = copy.deepcopy(self.data)
            case = data['rtx_lab']['cases'][0]
            if mutation == 'p95':
                case['arms'][0]['timing']['summary']['wall_ms']['p95'] *= 2
            else:
                case['paired_timing']['wall_ms']['paired_A_over_B'][0] *= 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.recompute(data)

    def test_removed_slow_observation_is_rejected(self):
        self.data['kv_transfer']['cases'][2]['arms'][1]['timing']['hot'].pop()
        with self.assertRaises(ValueError):
            module.recompute(self.data)

    def test_changed_measurement_without_recomputed_summary_is_rejected(self):
        self.data['rtx_lab']['cases'][0]['arms'][1]['timing']['hot'][0]['wall_ms'] *= 2
        with self.assertRaisesRegex(ValueError, 'recomputation mismatch'):
            module.recompute(self.data)

    def test_changed_payload_without_matching_geometry_is_rejected(self):
        self.data['kv_transfer']['payload']['h2d_payload_bytes_per_epoch'] += 2
        with self.assertRaisesRegex(ValueError, 'KV epoch byte mismatch'):
            module.recompute(self.data)

    def test_duplicate_paired_arm_is_rejected(self):
        case = self.data['rtx_lab']['cases'][0]
        case['arms'].append(copy.deepcopy(case['arms'][0]))
        with self.assertRaisesRegex(ValueError, 'missing paired arm'):
            module.recompute(self.data)

    def test_perceptual_claim_cannot_be_added_to_statistics(self):
        self.data['quality_assessed'] = True
        with self.assertRaises(ValueError):
            module.recompute(self.data)

    def test_extra_measurement_section_is_rejected(self):
        self.data['unreviewed_experiment'] = {'time_ms': 1.0}
        with self.assertRaisesRegex(ValueError, 'unexpected measurement sections'):
            module.recompute(self.data)

    def test_removed_case_is_rejected(self):
        self.data['rtx_lab']['cases'].clear()
        with self.assertRaisesRegex(ValueError, 'missing or duplicate measurement case'):
            module.recompute(self.data)

    def test_removed_timing_metrics_are_rejected(self):
        self.data['rtx_lab']['cases'][0]['paired_timing'] = {}
        with self.assertRaisesRegex(ValueError, 'missing required timing metric'):
            module.recompute(self.data)

    def test_wrong_payload_dtype_is_rejected(self):
        self.data['kv_transfer']['payload']['dtype'] = 'torch.float32'
        with self.assertRaisesRegex(ValueError, 'invalid BF16 KV shape'):
            module.recompute(self.data)


if __name__ == '__main__':
    unittest.main()
