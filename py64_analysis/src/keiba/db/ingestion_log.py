"""
Ingestionログ管理

取り込んだファイルの履歴を管理し、再現性・監査性を確保する
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Integer, BigInteger, DateTime, Text
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from .models import Base


class RawIngestionLog(Base):
    """Raw取り込みログ"""
    __tablename__ = "raw_ingestion_log"
    
    id = Column(BigInteger, primary_key=True)
    file_path = Column(String(500), nullable=False, unique=True)
    data_spec = Column(String(20), nullable=False)
    file_size = Column(BigInteger)
    record_count = Column(Integer)
    
    # 処理結果
    ra_count = Column(Integer, default=0)
    se_count = Column(Integer, default=0)
    o1_count = Column(Integer, default=0)
    o2_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    
    # メタ
    ingested_at = Column(DateTime, nullable=False)
    processed_at = Column(DateTime)
    status = Column(String(20), default="pending")  # pending, success, error
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.now)


def log_ingestion(
    session: Session,
    file_path: str,
    data_spec: str,
    file_size: int,
    record_count: int,
    result: dict,
    status: str = "success",
    error_message: Optional[str] = None,
) -> None:
    """取り込みログを記録"""
    stmt = insert(RawIngestionLog).values(
        file_path=file_path,
        data_spec=data_spec,
        file_size=file_size,
        record_count=record_count,
        ra_count=result.get("ra", 0),
        se_count=result.get("se", 0),
        o1_count=result.get("o1", 0),
        o2_count=result.get("o2", 0),
        error_count=result.get("errors", 0),
        ingested_at=datetime.now(),
        processed_at=datetime.now(),
        status=status,
        error_message=error_message,
    ).on_conflict_do_update(
        index_elements=["file_path"],
        set_={
            "processed_at": datetime.now(),
            "status": status,
            "ra_count": result.get("ra", 0),
            "se_count": result.get("se", 0),
            "o1_count": result.get("o1", 0),
            "o2_count": result.get("o2", 0),
            # ★追加: 再取り込み時の監査ログを完全にするため
            "error_count": result.get("errors", 0),
            "record_count": record_count,
            "file_size": file_size,
            "error_message": error_message,
        }
    )
    session.execute(stmt)
    session.commit()

