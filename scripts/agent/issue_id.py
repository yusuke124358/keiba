import hashlib
import re


def normalize_message(message: str) -> str:
    text = (message or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def build_issue_id(source: str, file_path: str, line: str, message: str) -> str:
    payload = "|".join([source or "", file_path or "", str(line or ""), normalize_message(message)])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"iss_{digest}"
