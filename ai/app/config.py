from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables."""

    model_path: Path = Field(
        default=Path(
            os.getenv(
                "MODEL_PATH",
                "models/license_plate/best.pt",
            )
        ),
        description="YOLO 가중치 파일 경로 (MODEL_BASE_DIR 기준 상대 경로 가능)",
    )
    model_base_dir: Path = Field(
        default=Path(os.getenv("MODEL_BASE_DIR", Path(__file__).resolve().parents[2])),
        description="모델 및 리소스가 위치한 최상단 디렉터리",
    )
    tessdata_dir: Path = Field(
        default=Path(os.getenv("TESSDATA_DIR", "tessdata")),
        description="Tesseract 학습 데이터 저장 경로",
    )
    detection_conf: float = Field(
        default=float(os.getenv("DETECTION_CONF", "0.5")),
        description="YOLO 추론 최소 confidence",
    )
    max_workers: int = Field(
        default=int(os.getenv("MAX_WORKERS", "4")),
        description="ThreadPoolExecutor 워커 수 (OCR 전용)",
    )

    class Config:
        frozen = True


settings = Settings()
