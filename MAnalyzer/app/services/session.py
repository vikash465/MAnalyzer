import secrets
import time
import threading
from dataclasses import dataclass, field

from app.models.schemas import MedicalReport


@dataclass
class ReportSession:
    session_id: str
    filename: str
    report: MedicalReport = field(default_factory=MedicalReport)
    created_at: float = field(default_factory=time.time)


_store: dict[str, ReportSession] = {}
_lock = threading.Lock()


def create_report_session(filename: str, report: MedicalReport) -> ReportSession:
    sid = secrets.token_hex(16)
    session = ReportSession(session_id=sid, filename=filename, report=report)
    with _lock:
        _store[sid] = session
    return session


def get_report_session(session_id: str) -> ReportSession | None:
    with _lock:
        session = _store.get(session_id)
    if session is None:
        return None
    from app.config import get_settings
    ttl = get_settings().session_ttl_minutes * 60
    if time.time() - session.created_at > ttl:
        remove_session(session_id)
        return None
    return session


def remove_session(session_id: str) -> None:
    with _lock:
        _store.pop(session_id, None)


def cleanup_expired_sessions() -> int:
    from app.config import get_settings
    ttl = get_settings().session_ttl_minutes * 60
    now = time.time()
    expired = []
    with _lock:
        for sid, session in _store.items():
            if now - session.created_at > ttl:
                expired.append(sid)
        for sid in expired:
            del _store[sid]
    return len(expired)
