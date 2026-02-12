import unittest
from unittest.mock import patch

from aimo3.client import OpenAICompatChatClient


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class ClientRetryTests(unittest.TestCase):
    def test_retries_transient_524_then_succeeds(self) -> None:
        client = OpenAICompatChatClient(
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
            api_key="dummy",
            timeout_sec=10,
            max_retries=1,
        )

        calls = [
            _DummyResponse(524, {"error": {"message": "timeout"}}, text="timeout"),
            _DummyResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Reasoning\nFINAL_ANSWER: 42",
                            }
                        }
                    ]
                },
            ),
        ]

        def _fake_post(*args, **kwargs):
            return calls.pop(0)

        with patch("requests.post", side_effect=_fake_post):
            out = client.generate(
                system_prompt="sys",
                user_prompt="usr",
                temperature=0.2,
                max_tokens=128,
            )

        self.assertIn("FINAL_ANSWER", out)


if __name__ == "__main__":
    unittest.main()
