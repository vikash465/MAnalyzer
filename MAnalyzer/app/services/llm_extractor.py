import json
import logging
import re

from app.models.schemas import MedicalReport
from app.services.llm.base import BaseLLM, EXTRACTION_SYSTEM_PROMPT
from app.services.llm import get_llm

logger = logging.getLogger(__name__)


async def extract_report_with_llm(
    raw_text: str,
    provider: str | None = None,
    llm: BaseLLM | None = None,
) -> MedicalReport:
    """Send raw report text to the LLM and parse the structured JSON response
    into a ``MedicalReport``.

    Falls back to an empty report (with ``raw_text`` preserved) if the LLM
    response cannot be parsed.
    """
    if llm is None:
        llm = get_llm(provider=provider)
    prompt = (
        "Extract all medical data from the following report text.\n\n"
        "--- BEGIN REPORT ---\n"
        f"{raw_text}\n"
        "--- END REPORT ---"
    )

    response_text = await llm.generate(
        prompt=prompt,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
    )

    report = _parse_llm_json(response_text, raw_text)
    report.raw_text = raw_text
    return report


def _parse_llm_json(text: str, raw_text: str) -> MedicalReport:
    """Robustly parse the LLM's JSON output into a MedicalReport."""
    cleaned = _strip_markdown_fences(text).strip()

    try:
        data = json.loads(cleaned)
        return MedicalReport.model_validate(data)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("LLM JSON parse failed (%s), trying to extract JSON block", exc)

    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return MedicalReport.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("Could not parse extracted JSON block: %s", exc)

    logger.error("LLM extraction failed entirely. Returning empty report.")
    return MedicalReport(raw_text=raw_text)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else 3
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
