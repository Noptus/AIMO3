import json
import unittest
from pathlib import Path

import pandas as pd


NOTEBOOK_PATH = Path("kaggle_kernel_submission_44_50/aimo3_submission.ipynb")


class Notebook4450LocalGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.runtime_cell = "".join(notebook["cells"][14]["source"])
        cls.predict_cell = "".join(notebook["cells"][15]["source"])

    def _build_runtime(self) -> dict[str, object]:
        class _PL:
            @staticmethod
            def DataFrame(data):
                return pd.DataFrame(data)

        globals_dict: dict[str, object] = {
            "os": __import__("os"),
            "re": __import__("re"),
            "time": __import__("time"),
            "gc": __import__("gc"),
            "Counter": __import__("collections").Counter,
            "pd": pd,
            "pl": _PL,
            "Optional": __import__("typing").Optional,
            "CFG": type("CFG", (), {"notebook_limit": 17_400}),
            "OPTIONAL_IMPORT_ERRORS": {},
            "load_harmony_encoding": object(),
            "OpenAI": object(),
            "Conversation": object(),
        }
        exec("from __future__ import annotations\n" + self.runtime_cell, globals_dict)
        exec("from __future__ import annotations\n" + self.predict_cell, globals_dict)
        return globals_dict

    def test_sample_passthrough_avoids_solver_init(self) -> None:
        runtime = self._build_runtime()
        calls = {"count": 0}

        def _fake_get_solver():
            calls["count"] += 1
            return None

        runtime["_get_solver"] = _fake_get_solver
        predict = runtime["predict"]

        output = predict(
            pd.Series(["000aaa", "111bbb", "222ccc"], name="id"),
            pd.Series(["sample one", "sample two", "sample three"], name="problem"),
        )
        if hasattr(output, "to_pandas"):
            output = output.to_pandas()

        self.assertEqual(calls["count"], 0)
        self.assertEqual(output["answer"].tolist(), [0, 0, 0])
        self.assertEqual(output["id"].tolist(), ["000aaa", "111bbb", "222ccc"])

        debug_rows = runtime["DEBUG_ROWS"]
        self.assertEqual(len(debug_rows), 3)
        self.assertTrue(
            all(str(row.get("source", "")) == "sample_validation_passthrough" for row in debug_rows)
        )

    def test_non_sample_safe_mode_returns_valid_range_answers(self) -> None:
        runtime = self._build_runtime()
        runtime["_get_solver"] = lambda: None
        predict = runtime["predict"]

        output = predict(
            pd.Series(["abc123", "def456"], name="id"),
            pd.Series(
                [
                    "Compute remainder when 123456 is divided by 97.",
                    "A hard geometry question with no obvious direct parser.",
                ],
                name="problem",
            ),
        )
        if hasattr(output, "to_pandas"):
            output = output.to_pandas()

        self.assertEqual(list(output.columns), ["id", "answer"])
        self.assertEqual(len(output), 2)
        self.assertTrue(((output["answer"] >= 0) & (output["answer"] <= 99_999)).all())

    def test_sm60_preflight_selects_deepseek_and_skips_gpt_oss(self) -> None:
        runtime = self._build_runtime()

        runtime["CFG"] = type(
            "CFG",
            (),
            {
                "notebook_limit": 17_400,
                "force_model_family": "auto",
                "disable_gpt_oss_on_sm_lt": 80,
                "prefer_small_model_below_gb": 28,
            },
        )
        runtime["BOOTSTRAP_DIAGNOSTICS"] = {}
        runtime["_check_mounts"] = lambda: (True, "ok")
        runtime["_get_gpu_info"] = lambda: {
            "ok": True,
            "major": 6,
            "minor": 0,
            "total_gb": 16.0,
            "raw": "cuda_capability_6_0",
        }
        runtime["_check_gpu_capability"] = lambda gpu_info=None: (True, "ok")
        runtime["_discover_model_path"] = (
            lambda gpu_info=None: (None, "primary_incompatible:/kaggle/input/models/gpt-oss:quant=mxfp4:sm=60")
        )
        runtime["_discover_fallback_model_path"] = (
            lambda gpu_info=None: ("/kaggle/input/models/deepseek-math-7b-instruct", "ok_fallback")
        )
        runtime["_check_tool_runtime_ready"] = lambda require_harmony=True: (False, "harmony_unavailable")
        runtime["_check_fallback_runtime_ready"] = lambda: (True, "ok")
        runtime["_apply_resource_profile"] = lambda gpu, path: {}

        preflight = runtime["_run_startup_preflight"]()
        self.assertTrue(preflight["ok"])
        self.assertEqual(preflight["gpu_sm"], 60)
        self.assertEqual(preflight["selected_model_family"], "deepseek")
        self.assertEqual(preflight["backend"], "deepseek_transformers")
        self.assertIn("gpu_sm_60_lt_80", str(preflight.get("primary_blocked_reason", "")))
        skipped = preflight.get("incompatible_models_skipped", [])
        self.assertTrue(any("quant=mxfp4" in str(item) for item in skipped))

    def test_get_solver_does_not_attempt_gpt_oss_on_sm60(self) -> None:
        runtime = self._build_runtime()

        runtime["solver"] = None
        runtime["SAFE_MODE_REASON"] = ""
        runtime["STARTUP_PREFLIGHT"] = {
            "ok": True,
            "selected_model_family": "deepseek",
            "selected_model_path": "/kaggle/input/models/deepseek-math-7b-instruct",
            "gpu_info": {"major": 6, "minor": 0, "ok": True},
        }

        calls = {"gpt": 0, "deepseek": 0}

        class _NeverGpt:
            def __init__(self, *_args, **_kwargs):
                calls["gpt"] += 1
                raise AssertionError("GPT-OSS path must not be used on sm_60 deepseek selection")

        class _DummyDeepSeek:
            def __init__(self, *_args, **_kwargs):
                calls["deepseek"] += 1
                self.runtime_status = "active:deepseek_transformers:cuda"

        runtime["AIMO3Solver"] = _NeverGpt
        runtime["AIMO3FallbackSolver"] = _DummyDeepSeek

        solver = runtime["_get_solver"]()
        self.assertIsNotNone(solver)
        self.assertEqual(calls["gpt"], 0)
        self.assertEqual(calls["deepseek"], 1)


if __name__ == "__main__":
    unittest.main()
