import unittest

from aimo3.langgraph_solver import LangGraphAIMO3Solver, is_langgraph_available
from aimo3.solver import SolverConfig


class _ScriptedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, *, system_prompt, user_prompt, temperature, max_tokens):
        if self.calls >= len(self._responses):
            return "FINAL_ANSWER: 0"
        out = self._responses[self.calls]
        self.calls += 1
        return out


@unittest.skipUnless(is_langgraph_available(), "langgraph is not installed")
class LangGraphSolverTests(unittest.TestCase):
    def test_tool_followup_flow_produces_answer(self) -> None:
        client = _ScriptedClient(
            responses=[
                """
I'll verify quickly.
```python
print(42)
```
""",
                "Using the tool check, we conclude FINAL_ANSWER: 42",
            ]
        )

        solver = LangGraphAIMO3Solver(
            client=client,
            config=SolverConfig(
                attempts=1,
                temperatures=(0.2,),
                max_tokens=256,
                max_code_blocks_per_attempt=1,
                agentic_tool_rounds=1,
                min_consensus=1,
                early_stop_attempt=1,
                early_stop_ratio=0.5,
            ),
        )

        result = solver.solve("Compute a checked integer answer.", problem_id="p1")
        self.assertEqual(result.predicted_answer, 42)
        self.assertEqual(len(result.candidates), 1)
        self.assertIn(result.candidates[0].stage, {"agent", "initial"})

    def test_no_signal_uses_deterministic_fallback(self) -> None:
        client = _ScriptedClient(responses=["No definitive output."])

        solver = LangGraphAIMO3Solver(
            client=client,
            config=SolverConfig(
                attempts=1,
                temperatures=(0.2,),
                max_tokens=128,
                agentic_tool_rounds=0,
                default_answer=7,
            ),
        )

        result = solver.solve("Hard statement with no parseable final line.", problem_id="p2")
        self.assertTrue(0 <= result.predicted_answer <= 99_999)
        self.assertEqual(len(result.candidates), 1)


if __name__ == "__main__":
    unittest.main()
