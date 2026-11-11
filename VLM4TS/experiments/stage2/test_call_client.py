"""call() builds one retrying client and reuses it; no network."""
import os
import unittest
from unittest.mock import MagicMock, patch

import run_stage2_review_hfz_v2 as runner


class CallClientTests(unittest.TestCase):
    def test_client_is_retrying_and_reused(self):
        runner._CLIENT = None
        fake = MagicMock()
        fake.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="{}"))]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), patch("dotenv.load_dotenv"), \
                patch("openai.OpenAI", return_value=fake) as factory:
            self.assertEqual(runner.call("p", "aW1n"), "{}")
            runner.call("p", "aW1n")
        self.assertEqual(factory.call_count, 1)
        self.assertGreaterEqual(factory.call_args.kwargs["max_retries"], 5)
        self.assertLessEqual(factory.call_args.kwargs["timeout"], 300)
        runner._CLIENT = None


if __name__ == "__main__":
    unittest.main()
