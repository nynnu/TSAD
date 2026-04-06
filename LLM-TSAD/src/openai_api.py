import os
from loguru import logger
from openai import OpenAI
import yaml
import random
from gemini_api import convert_openai_to_gemini, send_gemini_request


credentials = yaml.safe_load(open("credentials.yml"))


def openai_client(
    model,
    api_key=None,
    base_url="https://api.openai.com/v1"
):
    if api_key is None:
        assert model in credentials, f"Model {model} not found in credentials"
        # Randomly select an API key if multiple are provided
        if "round-robin" in credentials[model]:
            num_keys = len(credentials[model]["round-robin"])
            rand_idx = random.randint(0, num_keys - 1)
            credential = credentials[model]["round-robin"][rand_idx]
        else:
            credential = credentials[model]
        api_key = credential["api_key"]
        if "base_url" in credential:
            base_url = credential["base_url"]
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    logger.debug(
        f"API key: ****{api_key[-4:]}, endpoint: {base_url}"
    )
    
    return client


def send_openai_request(
    openai_request,
    model,
    api_key=None,
    base_url="https://api.openai.com/v1"
):
    import time
    if "gemini" in model:
        return send_gemini_request(
            convert_openai_to_gemini(openai_request),
            model,
            api_key=api_key
        )
    client = openai_client(model, api_key=api_key, base_url=base_url)

    for attempt in range(3):
        try:
            time.sleep(3)  # TPM rate limit 대응
            response = client.chat.completions.create(
                model=model, **openai_request
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limit. {wait}초 대기 후 재시도 ({attempt+1}/3)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("OpenAI API 최대 재시도 초과")
