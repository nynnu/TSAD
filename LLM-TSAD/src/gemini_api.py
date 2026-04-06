"""
Gemini API wrapper for LLM-TSAD.
google-genai 패키지(신버전) 사용.
"""
import os
import base64
import random
import time
import yaml
from io import BytesIO
from PIL import Image
from loguru import logger

try:
    import yaml
    credentials = yaml.safe_load(open("credentials.yml")) or {}
except Exception:
    credentials = {}


def _get_gemini_client(api_key=None):
    from google import genai
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY 환경변수를 설정하세요.")
    return genai.Client(api_key=key)


def convert_openai_to_gemini(openai_request):
    """OpenAI 포맷 요청을 Gemini contents 리스트로 변환."""
    contents = []
    for message in openai_request["messages"]:
        parts = []
        for content in message["content"]:
            if isinstance(content, str):
                parts.append(content)
            elif content["type"] == "text":
                parts.append(content["text"])
            elif content["type"] == "image_url":
                image_url = content["image_url"]["url"]
                if image_url.startswith("data:image"):
                    base64_str = image_url.split(",")[1]
                    img_data = base64.b64decode(base64_str)
                    img = Image.open(BytesIO(img_data))
                else:
                    import requests as req
                    resp = req.get(image_url)
                    img = Image.open(BytesIO(resp.content))
                parts.append(img)
        contents.extend(parts)

    temperature = openai_request.get("temperature", 0.4)
    return {"contents": contents, "temperature": temperature}


_SLEEP_SEC = 7  # 무료 10 RPM 대응


def send_gemini_request(gemini_request, model, api_key=None):
    """Gemini API 호출 (429 에러 시 최대 3회 재시도)."""
    from google.genai import types

    client = _get_gemini_client(api_key)
    logger.debug(f"Gemini 호출: model={model}")

    contents = gemini_request["contents"]
    temperature = gemini_request.get("temperature", 0.4)

    for attempt in range(3):
        try:
            time.sleep(_SLEEP_SEC)  # Rate limit 준수
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 60 * (attempt + 1)
                logger.warning(f"Rate limit (429). {wait}초 대기 후 재시도 ({attempt+1}/3)...")
                time.sleep(wait)
            else:
                logger.error(f"Gemini 오류: {e}")
                raise
    raise RuntimeError("Gemini API 최대 재시도 초과")
