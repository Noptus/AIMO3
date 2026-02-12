import unittest

from aimo3.cli import _build_solver_from_args, _build_sweep_trials, _profile_overrides, build_parser


class CliProfileTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
