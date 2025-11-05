from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings

settings = get_settings()

app = FastAPI(
    title="Backend ⇄ Plate Recognition Proxy",
    version="1.0.0",
    description="메인 백엔드가 번호판 인식 AI 서비스와 통신할 수 있도록 중계하는 FastAPI 서버",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", tags=["system"])
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "service": "backend-proxy"}


def _recognize_endpoint() -> str:
    base = settings.ai_service_url.rstrip("/")
    return f"{base}/v1/recognize"


async def _call_ai_service(image_file: UploadFile, payload: bytes) -> Dict[str, Any]:
    filename = image_file.filename or "upload.jpg"
    content_type = image_file.content_type or "application/octet-stream"
    files = {"image": (filename, payload, content_type)}

    timeout = httpx.Timeout(settings.request_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        last_error: Exception | None = None
        for attempt in range(settings.request_retries + 1):
            try:
                response = await client.post(_recognize_endpoint(), files=files)
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as exc:
                last_error = exc
                if attempt == settings.request_retries:
                    raise HTTPException(
                        status_code=502,
                        detail="AI 서비스와 통신 실패 (연결/타임아웃)",
                    ) from exc
                await asyncio.sleep(0.5 * (attempt + 1))
            except httpx.HTTPStatusError as exc:
                last_error = exc
                detail = "AI 서비스 오류"
                try:
                    data = exc.response.json()
                    detail = data.get("detail", detail)
                except json.JSONDecodeError:
                    if exc.response.text:
                        detail = exc.response.text[:200]
                raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc

        if last_error is not None:
            raise HTTPException(status_code=502, detail=str(last_error)) from last_error

        raise HTTPException(status_code=502, detail="AI 서비스 응답을 받을 수 없습니다.")


@app.post(
    "/api/license-plates",
    tags=["license-plate"],
    summary="번호판 인식 요청 (AI 서비스 프록시)",
)
async def recognize_plate(image: UploadFile = File(...)) -> Dict[str, Any]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="이미지 파일만 허용됩니다.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일은 처리할 수 없습니다.")

    return await _call_ai_service(image, data)
