import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("kaggle_kernel_submission_44_50/aimo3_submission.ipynb")


class Notebook4450ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = notebook["cells"]
        cls.full_code = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in cls.cells
            if cell.get("cell_type") == "code"
        )

    def _cell_code(self, index: int) -> str:
        return "".join(self.cells[index].get("source", []))

    def test_cfg_env_controls_present(self) -> None:
        cfg_code = self._cell_code(8)
        for knob in [
            "AIMO_NOTEBOOK_LIMIT_SEC",
            "AIMO_END_BUFFER_SEC",
            "AIMO_MAX_PROBLEM_SEC",
            "AIMO_MIN_PROBLEM_SEC",
            "AIMO_STRICT_SUBMISSION_MODE",
            "AIMO_FORCE_MODEL_FAMILY",
            "AIMO_DISABLE_GPT_OSS_ON_SM_LT",
            "AIMO_DEEPSEEK_ATTEMPTS_HIGH",
            "AIMO_DEEPSEEK_ATTEMPTS_MED",
            "AIMO_DEEPSEEK_ATTEMPTS_LOW",
            "AIMO_DEEPSEEK_VERIFY_TOP_K",
        ]:
            self.assertIn(knob, cfg_code)

    def test_bootstrap_lock_and_diagnostics_present(self) -> None:
        bootstrap_code = self._cell_code(3)
        for symbol in [
            "BOOTSTRAP_DIAGNOSTICS",
            "_BOOTSTRAP_LOCK_PATH",
            "fcntl.flock",
            "AIMO_BOOTSTRAP_EXTRA_PACKAGES",
            "AIMO_FORCE_BOOTSTRAP_LOCAL",
        ]:
            self.assertIn(symbol, bootstrap_code)

    def test_preflight_and_safe_mode_helpers_present(self) -> None:
        runtime_code = self._cell_code(14)
        for symbol in [
            "def _check_mounts(",
            "def _discover_model_path(",
            "def _discover_fallback_model_path(",
            "def _check_gpu_capability(",
            "def _check_tool_runtime_ready(",
            "def _run_startup_preflight(",
            "def _hashed_fallback(",
            "def _get_solver(",
            "'selected_model_path'",
            "'selected_model_family'",
            "'gpu_sm'",
            "'incompatible_models_skipped'",
            "'model_status'",
        ]:
            self.assertIn(symbol, runtime_code)

    def test_debug_contract_columns_present(self) -> None:
        runtime_code = self._cell_code(14)
        for col in [
            "'id'",
            "'answer'",
            "'source'",
            "'model_status'",
            "'time_left_s'",
            "'tool_calls'",
            "'tool_errors'",
            "'candidate_count'",
            "'vote_margin'",
        ]:
            self.assertIn(col, runtime_code)

    def test_predict_signature_and_sample_passthrough_present(self) -> None:
        predict_code = self._cell_code(15)
        self.assertIn(
            "def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None)",
            predict_code,
        )
        self.assertIn("sample_validation_passthrough", predict_code)
        self.assertIn("safe_mode_hash_fallback", predict_code)

    def test_local_gateway_submission_and_debug_outputs_present(self) -> None:
        final_code = self._cell_code(16)
        self.assertIn("run_local_gateway", final_code)
        self.assertIn("/kaggle/working/submission.parquet", final_code)
        self.assertIn("/kaggle/working/submission_debug_sources.csv", final_code)
        self.assertIn("/kaggle/working/runtime_health.json", final_code)
        self.assertIn("def _validate_submission_parquet(", final_code)
        self.assertIn("def _write_debug_csv(", final_code)
        self.assertIn("def _local_solver_warmup_check(", final_code)
        for key in [
            "'selected_model_family'",
            "'gpu_sm'",
            "'backend'",
            "'incompatible_models_skipped'",
            "'solver_warmup_ok'",
        ]:
            self.assertIn(key, final_code)

    def test_competition_rerun_env_parser_not_bool_cast(self) -> None:
        runtime_code = self._cell_code(14)
        self.assertIn("def _env_is_true(", runtime_code)
        self.assertNotIn("bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))", self.full_code)


if __name__ == "__main__":
    unittest.main()
