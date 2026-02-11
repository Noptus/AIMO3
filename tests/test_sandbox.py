import unittest

from aimo3.sandbox import SandboxPolicy, execute_python


class SandboxTests(unittest.TestCase):
    def test_executes_safe_code(self) -> None:
        result = execute_python("print(2 + 3)")
        self.assertTrue(result.success)
        self.assertIn("5", result.stdout)

    def test_blocks_unsafe_import(self) -> None:
        result = execute_python("import os\nprint('x')")
        self.assertFalse(result.success)
        self.assertEqual(result.exception_type, "CodeSafetyError")

    def test_respects_timeout(self) -> None:
        policy = SandboxPolicy(timeout_sec=1)
        result = execute_python("while True:\n    pass", policy=policy)
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
