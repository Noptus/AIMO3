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


if __name__ == "__main__":
    unittest.main()
