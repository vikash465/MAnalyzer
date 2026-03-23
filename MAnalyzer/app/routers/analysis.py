import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.models.schemas import AnalysisRequest, AnalysisResponse
from app.services.session import get_report_session
from app.services.analyzer import build_llm_context
from app.services.llm import get_llm_from_request

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(tags=["analysis"])


# ── Report-based analysis (requires uploaded report) ──

@router.post("/reports/{session_id}/analyze", response_model=AnalysisResponse)
@limiter.limit("20/minute")
async def analyze_report(request: Request, session_id: str, req: AnalysisRequest):
    session = get_report_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a report first.")

    settings = get_settings()
    context = build_llm_context(session.report, max_results=settings.context_max_records)

    try:
        llm = get_llm_from_request(request, provider=req.provider)
        reply = await llm.generate(prompt=req.message, context=context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("LLM generate failed for session %s", session_id)
        raise HTTPException(status_code=502, detail="AI analysis failed. Please try again.")

    return AnalysisResponse(
        reply=reply,
        provider=req.provider or settings.default_llm_provider,
        disclaimer=settings.medical_disclaimer if settings.disclaimer_enabled else "",
    )


@router.post("/reports/{session_id}/analyze/stream")
@limiter.limit("20/minute")
async def analyze_report_stream(request: Request, session_id: str, req: AnalysisRequest):
    session = get_report_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a report first.")

    settings = get_settings()
    context = build_llm_context(session.report, max_results=settings.context_max_records)

    try:
        llm = get_llm_from_request(request, provider=req.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    async def event_generator():
        try:
            async for chunk in llm.stream(prompt=req.message, context=context):
                yield chunk
        except Exception:
            logger.exception("LLM stream error for session %s", session_id)
            yield "\n\n[Error: AI response interrupted. Please try again.]"

    return StreamingResponse(event_generator(), media_type="text/plain")


# ── General symptom chat (no report needed) ──

@router.post("/chat/stream")
@limiter.limit("20/minute")
async def general_chat_stream(request: Request, req: AnalysisRequest):
    try:
        llm = get_llm_from_request(request, provider=req.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    async def event_generator():
        try:
            async for chunk in llm.stream(prompt=req.message):
                yield chunk
        except Exception:
            logger.exception("LLM stream error in general chat")
            yield "\n\n[Error: AI response interrupted. Please try again.]"

    return StreamingResponse(event_generator(), media_type="text/plain")
