import base64
import logging

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        from app.config import get_settings
        key = get_settings().encryption_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "ENCRYPTION_KEY is not set. Generate one with: "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        _fernet = Fernet(key.encode())
    return _fernet


def encrypt_value(plain_text: str) -> str:
    token = _get_fernet().encrypt(plain_text.encode())
    return base64.urlsafe_b64encode(token).decode()


def decrypt_value(encrypted: str) -> str:
    try:
        token = base64.urlsafe_b64decode(encrypted.encode())
        return _get_fernet().decrypt(token).decode()
    except (InvalidToken, Exception):
        logger.warning("Failed to decrypt value — invalid or expired token")
        raise ValueError("Invalid or expired encrypted token")
