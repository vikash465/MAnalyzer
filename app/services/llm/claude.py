from collections.abc import AsyncGenerator

from anthropic import AsyncAnthropic
import httpx

from app.config import get_settings
from app.services.llm.base import BaseLLM, MEDICAL_SYSTEM_PROMPT

LLM_TIMEOUT = httpx.Timeout(timeout=60.0, connect=10.0)


class ClaudeLLM(BaseLLM):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        settings = get_settings()
        self.client = AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key.get_secret_value(),
            timeout=LLM_TIMEOUT,
            max_retries=2,
        )
        self.model = model

    async def generate(self, prompt: str, context: str = "",
                       system_prompt: str | None = None) -> str:
        sys = system_prompt or MEDICAL_SYSTEM_PROMPT
        messages = self._build_messages(prompt, context)
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=sys,
            messages=messages,
        )
        return response.content[0].text

    async def stream(self, prompt: str, context: str = "") -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, context)
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=MEDICAL_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    @staticmethod
    def _build_messages(prompt: str, context: str) -> list[dict]:
        if not context:
            content = prompt
        else:
            content = (
                f"Medical Report Data:\n{context}\n\n"
                "IMPORTANT: Reply in the SAME language as the user's question below, "
                "NOT in the language of the report data above.\n\n"
                f"Question: {prompt}"
            )
        return [{"role": "user", "content": content}]
