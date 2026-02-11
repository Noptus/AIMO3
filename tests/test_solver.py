import unittest

from aimo3.solver import AIMO3Solver, SolverConfig


class DummyClient:
    def __init__(self, responses):
        self.responses = responses
        self.i = 0

    def generate(self, *, system_prompt, user_prompt, temperature, max_tokens):
        response = self.responses[self.i % len(self.responses)]
        self.i += 1
        return response


class SolverTests(unittest.TestCase):
    def test_solver_aggregates_candidates(self) -> None:
        responses = [
            "Reasoning. FINAL_ANSWER: 42",
            "```python\nprint(42)\n```\nFINAL_ANSWER: 42",
            "Reasoning. FINAL_ANSWER: 17",
            "```python\nprint(42)\n```\nFINAL_ANSWER: 42",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(attempts=4, temperatures=(0.2,), min_consensus=2, early_stop_attempt=3)
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Find x modulo 100000", problem_id=1)
        self.assertEqual(result.predicted_answer, 42)

    def test_solver_agentic_tool_round_uses_sandbox_observation(self) -> None:
        responses = [
            "Let's compute.\n```python\nprint(42)\n```",
            "From tool output, the value is FINAL_ANSWER: 42",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=1,
            temperatures=(0.2,),
            early_stop_attempt=99,
            agentic_tool_rounds=1,
            final_extractor_passes=0,
            verification_attempts=0,
            consistency_audit_attempts=0,
            geometry_recheck_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
            small_answer_guard_attempts=0,
            fallback_guess_attempts=0,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Compute final value.", problem_id=101)
        self.assertEqual(result.predicted_answer, 42)
        self.assertEqual(result.debug_summary["agentic_candidates"], 1)

    def test_solver_agentic_stateful_python_persists_variables(self) -> None:
        responses = [
            "Try code.\n```python\nx = 40\nprint(x)\n```",
            "Continue with one more check.\n```python\nprint(x + 2)\n```",
            "Now finalize. FINAL_ANSWER: 42",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=1,
            temperatures=(0.2,),
            early_stop_attempt=99,
            agentic_tool_rounds=2,
            agentic_stateful_python=True,
            final_extractor_passes=0,
            verification_attempts=0,
            consistency_audit_attempts=0,
            geometry_recheck_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
            small_answer_guard_attempts=0,
            fallback_guess_attempts=0,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Compute final value.", problem_id=102)
        self.assertEqual(result.predicted_answer, 42)
        self.assertEqual(result.debug_summary["agentic_candidates"], 1)
        self.assertEqual(result.debug_summary["max_agent_rounds"], 2)

    def test_solver_selector_stage_influences_final_pick(self) -> None:
        responses = [
            "Reasoning. FINAL_ANSWER: 10",
            "Reasoning. FINAL_ANSWER: 20",
            "Verifier reasoning. FINAL_ANSWER: 20",
            "Selector reasoning. FINAL_ANSWER: 20",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=2,
            temperatures=(0.2,),
            early_stop_attempt=99,
            verification_attempts=1,
            verification_top_k=2,
            selector_attempts=1,
            selector_top_k=2,
            sparse_recovery_attempts=0,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Compute the final integer answer.", problem_id=2)
        self.assertEqual(result.predicted_answer, 20)
        self.assertEqual(result.debug_summary["selector_candidates"], 1)

    def test_solver_penalizes_problem_echo_answers(self) -> None:
        responses = [
            "Reasoning. FINAL_ANSWER: 123",
            "Reasoning. FINAL_ANSWER: 456",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=2,
            temperatures=(0.2,),
            early_stop_attempt=99,
            verification_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("The statement includes 123 explicitly; find the true answer.", problem_id=3)
        self.assertEqual(result.predicted_answer, 456)

    def test_geometry_recheck_stage_influences_pick(self) -> None:
        responses = [
            "Reasoning. FINAL_ANSWER: 111",
            "Reasoning. FINAL_ANSWER: 222",
            "Geometry recheck. FINAL_ANSWER: 222",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=2,
            temperatures=(0.2,),
            early_stop_attempt=99,
            verification_attempts=0,
            geometry_recheck_attempts=1,
            geometry_top_k=2,
            selector_attempts=0,
            sparse_recovery_attempts=0,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("In triangle ABC with circumcircle and tangent lines, find the integer.", problem_id=4)
        self.assertEqual(result.predicted_answer, 222)
        self.assertEqual(result.debug_summary["geometry_recheck_candidates"], 1)

    def test_consistency_audit_stage_influences_pick(self) -> None:
        responses = [
            "Reasoning. FINAL_ANSWER: 10",
            "Reasoning. FINAL_ANSWER: 20",
            "Consistency audit. FINAL_ANSWER: 20",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=2,
            temperatures=(0.2,),
            early_stop_attempt=99,
            verification_attempts=0,
            consistency_audit_attempts=1,
            consistency_audit_top_k=2,
            geometry_recheck_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
            small_answer_guard_attempts=0,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Compute the final integer answer.", problem_id=9)
        self.assertEqual(result.predicted_answer, 20)
        self.assertEqual(result.debug_summary["consistency_audit_candidates"], 1)

    def test_adversarial_probe_stage_influences_pick(self) -> None:
        responses = [
            "Reasoning. FINAL_ANSWER: 10",
            "Reasoning. FINAL_ANSWER: 20",
            "Adversarial probe. FINAL_ANSWER: 20",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=2,
            temperatures=(0.2,),
            early_stop_attempt=99,
            verification_attempts=0,
            consistency_audit_attempts=0,
            adversarial_probe_attempts=1,
            adversarial_probe_top_k=2,
            geometry_recheck_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
            small_answer_guard_attempts=0,
            fallback_guess_attempts=0,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Compute the final integer answer.", problem_id=11)
        self.assertEqual(result.predicted_answer, 20)
        self.assertEqual(result.debug_summary["adversarial_probe_candidates"], 1)

    def test_small_answer_guard_escapes_trivial_collapse(self) -> None:
        responses = [
            "Reasoning. FINAL_ANSWER: 0",
            "Reasoning. FINAL_ANSWER: 1",
            "Guarded re-check. FINAL_ANSWER: 777",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=2,
            temperatures=(0.2,),
            early_stop_attempt=99,
            verification_attempts=0,
            geometry_recheck_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
            small_answer_guard_attempts=1,
            small_answer_guard_top_k=2,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Find the required integer value.", problem_id=5)
        self.assertEqual(result.predicted_answer, 777)
        self.assertEqual(result.debug_summary["small_guard_candidates"], 1)

    def test_small_answer_guard_triggers_with_single_trivial_candidate(self) -> None:
        responses = [
            "Reasoning. FINAL_ANSWER: 0",
            "Guarded re-check. FINAL_ANSWER: 42",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=1,
            temperatures=(0.2,),
            early_stop_attempt=99,
            verification_attempts=0,
            geometry_recheck_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
            small_answer_guard_attempts=1,
            small_answer_guard_top_k=1,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Compute the integer.", problem_id=6)
        self.assertEqual(result.predicted_answer, 42)

    def test_small_answer_guard_salvages_when_no_answer_is_parsed(self) -> None:
        responses = [
            "I cannot conclude from this step.",
            "Guarded re-check. FINAL_ANSWER: 31415",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=1,
            temperatures=(0.2,),
            early_stop_attempt=99,
            verification_attempts=0,
            geometry_recheck_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
            small_answer_guard_attempts=1,
            small_answer_guard_top_k=1,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Find the answer modulo 100000.", problem_id=7)
        self.assertEqual(result.predicted_answer, 31415)

    def test_fallback_guess_runs_when_no_answer_and_guard_disabled(self) -> None:
        responses = [
            "No extractable integer here.",
            "Fallback estimate. FINAL_ANSWER: 27182",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=1,
            temperatures=(0.2,),
            early_stop_attempt=99,
            agentic_tool_rounds=0,
            final_extractor_passes=0,
            verification_attempts=0,
            geometry_recheck_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
            small_answer_guard_attempts=0,
            fallback_guess_attempts=1,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Compute the integer.", problem_id=8)
        self.assertEqual(result.predicted_answer, 27182)
        self.assertEqual(result.debug_summary["fallback_guess_candidates"], 1)

    def test_fallback_guess_runs_on_unsupported_trivial_only_answers(self) -> None:
        responses = [
            "Reasoning. FINAL_ANSWER: 1",
            "Fallback estimate. FINAL_ANSWER: 1234",
        ]
        client = DummyClient(responses)
        cfg = SolverConfig(
            attempts=1,
            temperatures=(0.2,),
            early_stop_attempt=99,
            agentic_tool_rounds=0,
            verification_attempts=0,
            consistency_audit_attempts=0,
            geometry_recheck_attempts=0,
            selector_attempts=0,
            sparse_recovery_attempts=0,
            small_answer_guard_attempts=0,
            fallback_guess_attempts=1,
        )
        solver = AIMO3Solver(client, config=cfg)

        result = solver.solve("Compute the integer.", problem_id=10)
        self.assertEqual(result.predicted_answer, 1234)
        self.assertEqual(result.debug_summary["fallback_guess_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
