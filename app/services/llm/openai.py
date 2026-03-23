from collections.abc import AsyncGenerator

from openai import AsyncOpenAI
import httpx

from app.config import get_settings
from app.services.llm.base import BaseLLM, MEDICAL_SYSTEM_PROMPT

LLM_TIMEOUT = httpx.Timeout(timeout=60.0, connect=10.0)


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key.get_secret_value(),
            timeout=LLM_TIMEOUT,
            max_retries=2,
        )
        self.model = model

    async def generate(self, prompt: str, context: str = "",
                       system_prompt: str | None = None) -> str:
        messages = self._build_messages(prompt, context,
                                        system_prompt or MEDICAL_SYSTEM_PROMPT)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    async def stream(self, prompt: str, context: str = "") -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, context, MEDICAL_SYSTEM_PROMPT)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def _build_messages(prompt: str, context: str,
                        system_prompt: str) -> list[dict]:
        if not context:
            user_content = prompt
        else:
            user_content = (
                f"Medical Report Data:\n{context}\n\n"
                "IMPORTANT: Reply in the SAME language as the user's question below, "
                "NOT in the language of the report data above.\n\n"
                f"Question: {prompt}"
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
