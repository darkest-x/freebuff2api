import unittest

from freebuff2api.codebuff import CodebuffClient
from freebuff2api.config import Settings


class QueuedSessionClient(CodebuffClient):
    def __init__(self) -> None:
        super().__init__(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        self.calls = []
        self.responses = [
            {
                "status": "queued",
                "instanceId": "queued-instance",
                "model": "moonshotai/kimi-k2.6",
                "position": 0,
                "queueDepth": 0,
                "estimatedWaitMs": 0,
            },
            {
                "status": "active",
                "instanceId": "queued-instance",
                "model": "moonshotai/kimi-k2.6",
                "expiresAt": "2026-05-23T16:04:31.177Z",
                "remainingMs": 3_000_000,
            },
        ]

    async def _json(self, method, path, *, body=None, headers=None):
        self.calls.append((method, path))
        return self.responses.pop(0)


class CodebuffClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_session_polls_queued_session_until_active(self) -> None:
        client = QueuedSessionClient()
        try:
            session = await client.create_session("moonshotai/kimi-k2.6")
        finally:
            await client.aclose()

        self.assertEqual(session.instance_id, "queued-instance")
        self.assertEqual(session.model, "moonshotai/kimi-k2.6")
        self.assertEqual(
            client.calls,
            [
                ("POST", "/api/v1/freebuff/session"),
                ("GET", "/api/v1/freebuff/session"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
