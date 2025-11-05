# License Plate Inference Service

이 디렉터리는 `yeonsup_차량.py`에 구현돼 있던 YOLOv8 + Tesseract 파이프라인을
운영 환경에서 재사용할 수 있도록 **API 서버 형태로 모듈화**한 예시입니다.

## 아키텍처 개요

- **FastAPI**: HTTP 서버 및 스웨거 문서 제공
- **Ultralytics YOLOv8**: 번호판 영역 탐지
- **Tesseract OCR**: 번호판 문자열 인식
- **Pydantic**: 요청/응답 스키마 검증
- **Uvicorn**: ASGI 서버

### 동작 흐름
1. 서버 기동 시 `PlateRecognizer`가 모델 가중치(.pt)와 Tesseract 데이터를 초기화합니다.
2. `POST /v1/recognize` 엔드포인트로 이미지가 업로드되면
   - YOLO로 번호판 바운딩 박스를 탐지
   - 투시 보정 및 전처리
   - Tesseract로 문자열 후보를 평가
3. 최고 점수를 얻은 7자리 번호판 문자열과 메타데이터(JSON)를 반환합니다.

## 실행 방법

```bash
cd ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 실행 후 `http://localhost:8000/docs`에서 OpenAPI 페이지 확인
- `POST /v1/recognize`에 이미지 파일 업로드 → 번호판 문자열 반환

### 3) 확장(Scalability)

- **멀티 워커**: `gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4`
- **컨테이너 오토스케일링**: Docker Swarm / Kubernetes에서 HPA, Cluster Autoscaler 적용
- **비동기 요청**: FastAPI 엔드포인트에서 `run_in_threadpool`을 사용하여 CPU‑bound OCR을 스레드 풀에 위임
- **큐 기반**: 요청 폭주 시 Redis + Celery/RQ로 비동기 처리

> GPU/CPU 자원이 많은 노드에 감지/인식을 배치하고, 메인 백엔드 서버는 HTTP로만 연동하면 손쉽게 수평 확장이 가능합니다.

## API 개요

| Method | Path              | 설명                     |
|--------|-------------------|--------------------------|
| GET    | `/healthz`        | 헬스체크                 |
| POST   | `/v1/recognize`   | 이미지 업로드 → 번호판 인식 결과 |

### 요청 예 (cURL)

```bash
curl -X POST http://localhost:8000/v1/recognize \
     -F "image=@/path/to/plate.jpg"
```

### 응답 예

```json
{
  "plate": "43너9652",
  "score": 3,
  "bbox": [1263.0, 1739.0, 1878.8, 1967.7],
  "confidence": 0.914,
  "candidates": [
    {"text": "43너9652", "score": 3, "bin": "OTSU", "psm": 8},
    {"text": "43너9652", "score": 3, "bin": "OTSU", "psm": 7}
  ],
  "timing": {
    "detection_ms": 35.1,
    "ocr_ms": 18.7,
    "total_ms": 64.0
  }
}
```

## 디렉터리 구조

```
ai/
├── README.md
├── requirements.txt
└── app
    ├── __init__.py
    ├── config.py
    ├── main.py
    ├── model.py
    ├── schemas.py
    └── utils.py
```

- `config.py` : 환경 변수/경로 관리
- `model.py` : YOLO+Tesseract 초기화 및 로직
- `main.py`  : FastAPI 애플리케이션
- `schemas.py`: Pydantic 데이터 모델
- `utils.py` : 공통 헬퍼 함수

## 운영 팁

- **모델, tessdata 위치**는 `MODEL_PATH`, `MODEL_BASE_DIR`, `TESSDATA_PREFIX` 환경 변수로 조정합니다.
- **로그/모니터링**: `uvicorn` access log, Prometheus FastAPI metrics, Sentry 등 연동 가능.
- **보안**: 사내망에서만 접근하거나, API GW/인증 토큰을 붙여 사용하세요.

---

이 구조를 기반으로 메인 백엔드 서버는 HTTP 요청 한 번으로 번호판 문자열을 받아 올 수 있으며, 필요 시 여러 인퍼런스 인스턴스를 수평 확장하여 처리량을 확보할 수 있습니다.
