from __future__ import annotations

import traceback
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .model import PlateRecognizer
from .schemas import HealthResponse, RecognitionResponse


@lru_cache(maxsize=1)
def get_recognizer() -> PlateRecognizer:
    return PlateRecognizer()


app = FastAPI(
    title="License Plate Recognition Service",
    version="1.0.0",
    description="YOLOv8 + Tesseract 기반 번호판 인식 API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", response_model=HealthResponse, tags=["system"])
async def healthz() -> HealthResponse:
    return HealthResponse()


@app.post(
    "/v1/recognize",
    response_model=RecognitionResponse,
    tags=["license-plate"],
    summary="이미지에서 번호판 인식",
)
async def recognize(image: UploadFile = File(...)) -> RecognitionResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="이미지 파일만 업로드 가능합니다.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    recognizer = get_recognizer()
    try:
        result = await run_in_threadpool(recognizer.predict, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="번호판 인식 중 오류가 발생했습니다.") from exc

    return RecognitionResponse(**result)


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

