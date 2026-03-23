import logging

from fastapi import APIRouter, UploadFile, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.models.schemas import UploadReportResponse, ReportSummary
from app.services.parser import extract_text
from app.services.llm_extractor import extract_report_with_llm
from app.services.llm import get_llm_from_request
from app.services.session import create_report_session, get_report_session
from app.services.analyzer import get_abnormal_results

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/upload", response_model=UploadReportResponse)
@limiter.limit("10/minute")
async def upload_report(request: Request, file: UploadFile):
    settings = get_settings()

    filename = file.filename or "unnamed"
    ext = f".{filename.rsplit('.', 1)[-1]}" if "." in filename else ""

    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Accepted: {settings.allowed_extensions}",
        )

    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds maximum upload size.")

    try:
        raw_text = extract_text(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))

    try:
        user_llm = get_llm_from_request(request)
        report = await extract_report_with_llm(raw_text, llm=user_llm)
    except Exception:
        logger.exception("LLM extraction failed for %s", filename)
        raise HTTPException(
            status_code=502,
            detail="Failed to extract report data. Please try again later.",
        )

    session = create_report_session(filename=filename, report=report)
    abnormal = get_abnormal_results(report)

    return UploadReportResponse(
        session_id=session.session_id,
        filename=session.filename,
        report_type=report.report_type.value,
        patient=report.patient,
        result_count=len(report.results),
        abnormal_count=len(abnormal),
        preview=report.results[:5],
    )


@router.get("/{session_id}/summary", response_model=ReportSummary)
async def report_summary(session_id: str):
    session = get_report_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a report first.")

    settings = get_settings()
    report = session.report
    abnormal = get_abnormal_results(report)

    return ReportSummary(
        session_id=session.session_id,
        filename=session.filename,
        report_type=report.report_type.value,
        patient=report.patient,
        total_tests=len(report.results),
        normal_count=len(report.results) - len(abnormal),
        abnormal_count=len(abnormal),
        all_results=report.results,
        disclaimer=settings.medical_disclaimer if settings.disclaimer_enabled else "",
    )
