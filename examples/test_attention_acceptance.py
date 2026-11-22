"""Exercise the actual fixture acceptance policy without importing PyTorch."""
import ast
import copy
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/rtx_dit_lab.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def source_function(name):
    node = next(item for item in TREE.body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {"math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


acceptance = source_function("attention_acceptance")


def case_status(comparisons, flash_executed):
    """Use the case's actual gate call and returned status, without CUDA work."""
    function = next(item for item in TREE.body
                    if isinstance(item, ast.FunctionDef) and item.name == "attention_case")
    assignment = next(item for item in function.body
                      if isinstance(item, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == "acceptance"
                          for target in item.targets))
    returned = next(item.value for item in function.body if isinstance(item, ast.Return))
    status = next(value for key, value in zip(returned.keys, returned.values)
                  if isinstance(key, ast.Constant) and key.value == "status")
    namespace = {"attention_acceptance": acceptance, "comparisons": comparisons,
                 "outputs": {"B": object()} if flash_executed else {}}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return eval(compile(ast.Expression(status), str(SOURCE), "eval"), namespace)


def recorded_comparisons(filename="measurements.json"):
    data = json.loads((ROOT / "docs/manuscript/evidence" / filename).read_text(encoding="utf-8"))

    def find(value):
        if isinstance(value, dict):
            if value.get("name") == "attention" and "comparisons" in value:
                return value["comparisons"]
            children = value.values()
        elif isinstance(value, list):
            children = value
        else:
            return None
        for child in children:
            result = find(child)
            if result is not None:
                return result
        return None

    result = find(data)
    if result is None:
        raise AssertionError("saved attention comparisons missing")
    return result


class AttentionAcceptanceTests(unittest.TestCase):
    def test_both_saved_bf16_comparisons_pass(self):
        for filename in ("measurements.json", "measurements_sm90.json"):
            with self.subTest(filename=filename):
                result = acceptance(recorded_comparisons(filename), flash_executed=True)
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["passed"])
                self.assertTrue(all(result["checks"].values()))

    def test_finite_zero_flash_output_is_rejected(self):
        comparisons = recorded_comparisons()
        # A1=A2=FP32=[1,1], B=[0,0]: finite but 100% relative L2 error.
        for name in ("B_vs_fp32_reference", "B_vs_A1"):
            comparisons[name].update(finite=True, bitwise=False, max_abs=1.0, relative_l2=1.0)
        result = acceptance(comparisons, flash_executed=True)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])

    def test_each_required_metric_has_an_enforced_upper_bound(self):
        for name in recorded_comparisons():
            for metric in ("max_abs", "relative_l2"):
                with self.subTest(comparison=name, metric=metric):
                    comparisons = recorded_comparisons()
                    comparisons[name][metric] = 0.005001
                    self.assertFalse(acceptance(comparisons, flash_executed=True)["passed"])

    def test_limits_are_inclusive(self):
        comparisons = recorded_comparisons()
        comparisons["B_vs_A1"].update(max_abs=0.005, relative_l2=0.005)
        self.assertTrue(acceptance(comparisons, flash_executed=True)["passed"])

    def test_nonfinite_undefined_and_invalid_metrics_are_rejected(self):
        for metric in ("max_abs", "relative_l2"):
            for value in (float("nan"), float("inf"), None, -0.001, True):
                with self.subTest(metric=metric, value=value):
                    comparisons = recorded_comparisons()
                    comparisons["B_vs_A1"][metric] = value
                    self.assertFalse(acceptance(comparisons, flash_executed=True)["passed"])
        comparisons = recorded_comparisons()
        comparisons["B_vs_A1"]["finite"] = False
        self.assertFalse(acceptance(comparisons, flash_executed=True)["passed"])

    def test_missing_comparison_cannot_pass(self):
        for name in recorded_comparisons():
            with self.subTest(comparison=name):
                comparisons = recorded_comparisons()
                del comparisons[name]
                self.assertFalse(acceptance(comparisons, flash_executed=True)["passed"])

    def test_math_repeat_must_remain_bitwise_equal(self):
        comparisons = recorded_comparisons()
        comparisons["A2_vs_A1"]["bitwise"] = False
        self.assertFalse(acceptance(comparisons, flash_executed=True)["passed"])

    def test_unavailable_flash_is_skipped_not_passed(self):
        comparisons = {key: value for key, value in recorded_comparisons().items()
                       if not key.startswith("B_")}
        result = acceptance(comparisons, flash_executed=False)
        self.assertEqual(result["status"], "skipped")
        self.assertFalse(result["passed"])
        self.assertTrue(result["baseline_passed"])
        self.assertFalse(result["flash_assessed"])

    def test_skipped_backend_does_not_create_a_paired_ratio(self):
        statistics = source_function("aba_statistics")
        result = statistics([{"status": "complete"}, {"status": "skipped"}, {"status": "complete"}])
        self.assertFalse(result["available"])
        self.assertNotIn("wall_ms", result)
        self.assertNotIn("cuda_event_ms", result)

    def test_failed_math_baseline_is_failed_even_when_flash_is_unavailable(self):
        comparisons = recorded_comparisons()
        comparisons["A1_vs_fp32_reference"]["relative_l2"] = 1.0
        self.assertEqual(acceptance(comparisons, flash_executed=False)["status"], "failed")

    def test_gate_failure_or_skip_reaches_the_run_receipt(self):
        function = next(item for item in TREE.body
                        if isinstance(item, ast.FunctionDef) and item.name == "run_lab")
        assignment = next(item for item in function.body
                          if isinstance(item, ast.Assign) and any(
                              isinstance(target, ast.Subscript)
                              and isinstance(target.slice, ast.Constant)
                              and target.slice.value == "technical_passed"
                              for target in item.targets))
        predicate = compile(ast.Expression(assignment.value), str(SOURCE), "eval")
        good = recorded_comparisons()
        bad = copy.deepcopy(good)
        bad["B_vs_A1"]["relative_l2"] = 1.0
        for comparisons, executed, expected_status, expected in (
                (good, True, "complete", True), (bad, True, "failed", False),
                (good, False, "skipped", False)):
            with self.subTest(status=expected_status):
                status = case_status(comparisons, executed)
                self.assertEqual(status, expected_status)
                self.assertEqual(eval(predicate, {"receipt": {"cases": [{"status": status}]},
                                                  "names": ["attention"]}), expected)

    def test_acceptance_does_not_modify_comparison_measurements(self):
        comparisons = recorded_comparisons()
        original = copy.deepcopy(comparisons)
        acceptance(comparisons, flash_executed=True)
        self.assertEqual(comparisons, original)


if __name__ == "__main__":
    unittest.main()
