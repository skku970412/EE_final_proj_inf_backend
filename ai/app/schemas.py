from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    text: str = Field(..., description="OCR 후보 문자열")
    score: int = Field(..., description="정규식 일치도 점수 (0~3)")
    bin: str = Field(..., description="전처리 방법명")
    psm: int = Field(..., description="Tesseract PSM 모드")


class RecognitionResponse(BaseModel):
    plate: str = Field(..., description="최종 번호판 (하이픈 제거)")
    score: int = Field(..., description="최종 후보 점수")
    raw_text: str = Field(..., description="Tesseract RAW 텍스트")
    bbox: Tuple[float, float, float, float] = Field(..., description="YOLO 바운딩 박스 (x1,y1,x2,y2)")
    confidence: float = Field(..., description="YOLO confidence")
    candidates: List[Candidate] = Field(default_factory=list, description="OCR 후보 목록")
    timing: dict = Field(default_factory=dict, description="파이프라인 처리 시간(ms)")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "v1"
    detail: Optional[str] = None

