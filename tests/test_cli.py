import unittest
import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory

from aimo3.cli import (
    FORCED_MODEL,
    _assert_preflight_ready_for_submit,
    _assert_kernel_metadata_safe,
    _build_solver_from_args,
    _build_sweep_trials,
    _profile_overrides,
    _validate_kernel_runtime_health,
    build_parser,
)
from aimo3.langgraph_solver import is_langgraph_available


class CliProfileTests(unittest.TestCase):
    def test_default_orchestrator_is_classic(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "solve",
                "--input-csv",
                "examples/sample_problems.csv",
            ]
        )
        self.assertEqual(args.orchestrator, "classic")

    def test_autonomous_profile_defaults_to_10_minute_full_time_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "solve",
                "--input-csv",
                "examples/sample_problems.csv",
                "--profile",
                "autonomous120b",
            ]
        )

        overrides = _profile_overrides(args)
        self.assertEqual(overrides["per_problem_time_sec"], 600)
        self.assertTrue(overrides["force_full_problem_time"])
        self.assertGreaterEqual(int(overrides["stage_time_reserve_sec"]), 45)
        self.assertTrue(bool(overrides["force_tool_round_for_unverified"]))
        self.assertGreaterEqual(int(overrides["min_time_for_attempt_sec"]), 25)
        self.assertGreaterEqual(int(overrides["min_time_for_stage_sec"]), 10)
        self.assertGreaterEqual(int(overrides["parallel_attempt_workers"]), 4)
        self.assertGreaterEqual(int(overrides["parallel_code_workers"]), 4)

    def test_autonomous_profile_respects_explicit_time_and_force_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "solve",
                "--input-csv",
                "examples/sample_problems.csv",
                "--profile",
                "autonomous120b",
                "--per-problem-time-sec",
                "900",
                "--no-force-full-problem-time",
            ]
        )

        solver = _build_solver_from_args(args)
        self.assertEqual(solver.config.per_problem_time_sec, 900)
        self.assertFalse(solver.config.force_full_problem_time)

    def test_parallel_worker_flags_are_applied(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "solve",
                "--input-csv",
                "examples/sample_problems.csv",
                "--parallel-attempt-workers",
                "3",
                "--parallel-code-workers",
                "5",
            ]
        )

        solver = _build_solver_from_args(args)
        self.assertEqual(solver.config.parallel_attempt_workers, 3)
        self.assertEqual(solver.config.parallel_code_workers, 5)

    def test_force_tool_round_flag_is_applied(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "solve",
                "--input-csv",
                "examples/sample_problems.csv",
                "--force-tool-round-for-unverified",
                "--stage-time-reserve-sec",
                "33",
            ]
        )

        solver = _build_solver_from_args(args)
        self.assertTrue(solver.config.force_tool_round_for_unverified)
        self.assertEqual(solver.config.stage_time_reserve_sec, 33)

    def test_benchmark_sweep_parser_and_trial_catalog(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "benchmark-sweep",
                "--reference-csv",
                "reference/ai-mathematical-olympiad-progress-prize-3/reference.csv",
                "--trial-set",
                "quick",
            ]
        )
        trials = _build_sweep_trials(args)
        self.assertGreaterEqual(len(trials), 1)
        self.assertLessEqual(len(trials), 4)
        self.assertEqual(trials[0]["name"], "base")

    def test_kaggle_kernel_pipeline_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "kaggle-kernel-pipeline",
                "--competition",
                "ai-mathematical-olympiad-progress-prize-3",
            ]
        )
        self.assertEqual(args.kernel_dir, "kaggle_kernel_submission")
        self.assertEqual(args.required_output_file, "submission.parquet")
        self.assertTrue(args.strict_output_validation)
        self.assertTrue(args.skip_if_daily_limit_reached)

    def test_kaggle_kernel_preflight_44_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "kaggle-kernel-preflight-44",
                "--competition",
                "ai-mathematical-olympiad-progress-prize-3",
            ]
        )
        self.assertEqual(args.notebook_dir, "kaggle_kernel_submission_44_50")
        self.assertEqual(args.stage, "all")
        self.assertEqual(args.required_output_file, "submission.parquet")
        self.assertTrue(args.strict_runtime_health)

    def test_kaggle_submit_44_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "kaggle-submit-44",
                "--competition",
                "ai-mathematical-olympiad-progress-prize-3",
            ]
        )
        self.assertEqual(args.notebook_dir, "kaggle_kernel_submission_44_50")
        self.assertEqual(args.output_file_name, "submission.parquet")
        self.assertEqual(args.max_preflight_age_minutes, 240)
        self.assertTrue(args.skip_if_daily_limit_reached)

    def test_kernel_metadata_safe_requires_inference_server_markers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_file = root / "submission.py"
            code_file.write_text("print('hello')\n", encoding="utf-8")
            metadata = {
                "id": "owner/slug",
                "code_file": "submission.py",
                "competition_sources": ["ai-mathematical-olympiad-progress-prize-3"],
                "enable_internet": False,
            }
            with self.assertRaises(ValueError):
                _assert_kernel_metadata_safe(
                    metadata=metadata,
                    kernel_root=root,
                    competition="ai-mathematical-olympiad-progress-prize-3",
                )

    def test_kernel_metadata_safe_accepts_inference_server_markers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_file = root / "submission.py"
            code_file.write_text(
                "AIMO3InferenceServer\nrun_local_gateway\nKAGGLE_IS_COMPETITION_RERUN\n",
                encoding="utf-8",
            )
            metadata = {
                "id": "owner/slug",
                "code_file": "submission.py",
                "competition_sources": ["ai-mathematical-olympiad-progress-prize-3"],
                "enable_internet": False,
                "model_sources": ["danielhanchen/gpt-oss-120b/transformers/default/1"],
            }
            _assert_kernel_metadata_safe(
                metadata=metadata,
                kernel_root=root,
                competition="ai-mathematical-olympiad-progress-prize-3",
            )

    def test_langgraph_orchestrator_builder(self) -> None:
        if not is_langgraph_available():
            self.skipTest("langgraph is not installed")

        parser = build_parser()
        args = parser.parse_args(
            [
                "solve",
                "--input-csv",
                "examples/sample_problems.csv",
                "--orchestrator",
                "langgraph",
            ]
        )
        solver = _build_solver_from_args(args)
        self.assertEqual(type(solver).__name__, "LangGraphAIMO3Solver")

    def test_model_policy_forces_gpt_oss_120b(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "solve",
                "--input-csv",
                "examples/sample_problems.csv",
                "--model",
                "openai/gpt-oss-20b",
            ]
        )
        solver = _build_solver_from_args(args)
        self.assertEqual(solver.client.model, FORCED_MODEL)

    def test_validate_kernel_runtime_health_requires_debug_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                _validate_kernel_runtime_health(output_dir=tmpdir, strict=True)

    def test_validate_kernel_runtime_health_rejects_disabled_model_status(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission_debug_sources.csv").write_text(
                "id,answer,source,model_status,time_left_s,tool_calls,tool_errors,candidate_count,vote_margin\n"
                "000aaa,0,pattern_solver,disabled:model_load_failed,100,1,0,2,0.4\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _validate_kernel_runtime_health(output_dir=root, strict=True)

    def test_validate_kernel_runtime_health_accepts_healthy_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission_debug_sources.csv").write_text(
                "id,answer,source,model_status,time_left_s,tool_calls,tool_errors,candidate_count,vote_margin\n"
                "abc,123,hf_sc_votes12_b1_tool3,ready:transformers:/kaggle/input/models/x,100,2,0,3,1.7\n",
                encoding="utf-8",
            )
            (root / "runtime_health.json").write_text(
                (
                    '{"solver_warmup_ok": true, "reason": "ok", '
                    '"selected_model_family": "deepseek", "gpu_sm": 60, '
                    '"backend": "active:deepseek_transformers:cuda"}'
                ),
                encoding="utf-8",
            )
            _validate_kernel_runtime_health(output_dir=root, strict=True)

    def test_validate_kernel_runtime_health_allows_sample_passthrough(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission_debug_sources.csv").write_text(
                "id,answer,source,model_status,time_left_s,tool_calls,tool_errors,candidate_count,vote_margin\n"
                "000aaa,0,sample_validation_passthrough,disabled:uninitialized,100,0,0,0,0.0\n"
                "111bbb,0,sample_validation_passthrough,disabled:uninitialized,100,0,0,0,0.0\n",
                encoding="utf-8",
            )
            (root / "runtime_health.json").write_text(
                (
                    '{"solver_warmup_ok": true, "reason": "ok", '
                    '"selected_model_family": "deepseek", "gpu_sm": 60, '
                    '"backend": "active:deepseek_transformers:cuda"}'
                ),
                encoding="utf-8",
            )
            _validate_kernel_runtime_health(output_dir=root, strict=True)

    def test_validate_kernel_runtime_health_rejects_missing_required_columns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission_debug_sources.csv").write_text(
                "id,answer,source,model_status\nabc,1,solver,active\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _validate_kernel_runtime_health(output_dir=root, strict=True)

    def test_validate_kernel_runtime_health_rejects_sample_without_runtime_health(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission_debug_sources.csv").write_text(
                "id,answer,source,model_status,time_left_s,tool_calls,tool_errors,candidate_count,vote_margin\n"
                "000aaa,0,sample_validation_passthrough,disabled:uninitialized,100,0,0,0,0.0\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _validate_kernel_runtime_health(output_dir=root, strict=True)

    def test_validate_kernel_runtime_health_rejects_non_sample_without_runtime_health(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission_debug_sources.csv").write_text(
                "id,answer,source,model_status,time_left_s,tool_calls,tool_errors,candidate_count,vote_margin\n"
                "abc,123,hf_sc_votes12_b1_tool3,ready:transformers:/kaggle/input/models/x,100,2,0,3,1.7\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _validate_kernel_runtime_health(output_dir=root, strict=True)

    def test_validate_kernel_runtime_health_rejects_safe_mode_source(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission_debug_sources.csv").write_text(
                "id,answer,source,model_status,time_left_s,tool_calls,tool_errors,candidate_count,vote_margin\n"
                "abc,42,safe_mode_hash_fallback,ready:transformers,100,0,0,1,0.2\n",
                encoding="utf-8",
            )
            (root / "runtime_health.json").write_text(
                (
                    '{"solver_warmup_ok": true, "reason": "ok", '
                    '"selected_model_family": "deepseek", "gpu_sm": 60, '
                    '"backend": "active:deepseek_transformers:cuda"}'
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _validate_kernel_runtime_health(output_dir=root, strict=True)

    def test_validate_kernel_runtime_health_rejects_fatal_log_markers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission_debug_sources.csv").write_text(
                "id,answer,source,model_status,time_left_s,tool_calls,tool_errors,candidate_count,vote_margin\n"
                "abc,123,hf_sc_votes12_b1_tool3,ready:transformers:/kaggle/input/models/x,100,2,0,3,1.7\n",
                encoding="utf-8",
            )
            (root / "runtime_health.json").write_text(
                (
                    '{"solver_warmup_ok": true, "reason": "ok", '
                    '"selected_model_family": "deepseek", "gpu_sm": 60, '
                    '"backend": "active:deepseek_transformers:cuda"}'
                ),
                encoding="utf-8",
            )
            (root / "vllm_server.log").write_text(
                "startup failed: quantization unsupported on this gpu\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _validate_kernel_runtime_health(output_dir=root, strict=True)

    def test_assert_preflight_ready_for_submit_accepts_fresh_status(self) -> None:
        status = {
            "ok": True,
            "stage": "all",
            "notebook_dir": "kaggle_kernel_submission_44_50",
            "competition": "ai-mathematical-olympiad-progress-prize-3",
            "finished_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
        _assert_preflight_ready_for_submit(
            status=status,
            notebook_dir="kaggle_kernel_submission_44_50",
            competition="ai-mathematical-olympiad-progress-prize-3",
            max_age_minutes=30,
        )

    def test_assert_preflight_ready_for_submit_rejects_stale_status(self) -> None:
        stale = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=12)
        status = {
            "ok": True,
            "stage": "all",
            "notebook_dir": "kaggle_kernel_submission_44_50",
            "competition": "ai-mathematical-olympiad-progress-prize-3",
            "finished_at": stale.replace(microsecond=0).isoformat(),
        }
        with self.assertRaises(RuntimeError):
            _assert_preflight_ready_for_submit(
                status=status,
                notebook_dir="kaggle_kernel_submission_44_50",
                competition="ai-mathematical-olympiad-progress-prize-3",
                max_age_minutes=30,
            )


if __name__ == "__main__":
    unittest.main()
