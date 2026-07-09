import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from config import BACKOFF_BASE_SECONDS, MAX_RETRIES, MODEL_NAME, TEMPERATURE

load_dotenv()  # picks up OPENAI_API_KEY from sanity/.env if present


@dataclass
class VLMResponse:
    status: str
    raw_response: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float = 0.0
    error: str | None = None


class VLMClient:
    def __init__(self, model_name: str = MODEL_NAME):
        self.vlm_model = model_name
        self.client = OpenAI()

    def _call_once(self, prompt: str, img_b64: str) -> VLMResponse | None:
        payload = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
            ],
        }]
        start = time.perf_counter()
        try:
            resp = self.client.responses.create(
                model=self.vlm_model,
                input=payload,
                temperature=TEMPERATURE,
            )
            latency = time.perf_counter() - start
            usage = getattr(resp, "usage", None)
            return VLMResponse(
                status="OK",
                raw_response=(getattr(resp, "output_text", "") or "").strip(),
                input_tokens=getattr(usage, "input_tokens", None) if usage else None,
                output_tokens=getattr(usage, "output_tokens", None) if usage else None,
                total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                latency_seconds=latency,
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            return VLMResponse(status="API_ERROR", latency_seconds=latency, error=str(exc))

    def call(self, prompt: str, img_b64: str) -> VLMResponse:
        last = None
        for attempt in range(MAX_RETRIES):
            last = self._call_once(prompt, img_b64)
            if last and last.status == "OK":
                return last
            if last and last.error and _is_non_retryable(last.error):
                return last
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
        return last or VLMResponse(status="API_ERROR", error="unknown_api_error")


def _is_non_retryable(error: str) -> bool:
    text = error.lower()
    return any(
        marker in text
        for marker in [
            "insufficient_quota",
            "invalid_api_key",
            "incorrect api key",
            "billing",
        ]
    )


class MockVLMClient:
    def call(self, prompt: str, img_b64: str, answer: str = "maintained") -> VLMResponse:
        raw = (
            '{"answer": "%s", "reason": "Mock response for pipeline validation.", '
            '"confidence": 0.99}'
        ) % answer
        return VLMResponse(
            status="OK",
            raw_response=raw,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_seconds=0.0,
        )
