# Back-End ↔ Plate Recognition 통합 샘플

이 디렉터리는 AI 인퍼런스 코드(`ai/`, YOLOv8 + Tesseract 기반 번호판 인식 API`)와
별도의 **백엔드 프록시 서버**를 한 프로젝트 안에서 함께 운용하는 구조를 제공합니다.

- **AI 인퍼런스 서버**: `ai` FastAPI 앱(포트 8001)
- **백엔드 프록시**: 업로드 이미지를 AI 서버에 전달 후 결과 반환(포트 8000)
- **샘플 클라이언트**: 통합 흐름 검증용 CLI (`sample_client.py`)

> ⚠️ 현재 학교 서버 환경에서는 Docker 사용이 제한되어 있으므로, Python 가상환경 + Uvicorn 방식으로 실행합니다.

## 1. 의존성 설치

```bash
cd /home/work/llama_young
python -m venv back_end_detect_resolve_plate/venv
source back_end_detect_resolve_plate/venv/bin/activate
pip install -r back_end_detect_resolve_plate/ai/requirements.txt
pip install -r back_end_detect_resolve_plate/requirements.txt
```

추가로 시스템 패키지 `tesseract-ocr`가 설치되어 있어야 합니다.

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

## 2. 한번에 실행하기

AI 인퍼런스 + 백엔드 프록시를 동시에 띄우려면 아래 스크립트를 사용합니다. 기본값은 AI(8001) → 백엔드(8000) 순으로 기동하며, `Ctrl+C`로 모두 종료됩니다.

```bash
cd /home/work/llama_young/back_end_detect_resolve_plate
./scripts/run_all.sh
```

스크립트는 `./venv` 또는 `../.venv` 순으로 가상환경을 자동 활성화합니다. 필요 시 실행 전에 `AI_PORT`, `BACKEND_PORT`, `MODEL_BASE_DIR`, `MODEL_PATH`, `TESSDATA_DIR`, `VENV_PATH` 등을 지정할 수 있습니다.

### 포함된 리소스

- `models/license_plate/best.pt`: YOLOv8 번호판 가중치
- `tessdata/kor.traineddata`, `eng.traineddata`, `osd.traineddata`: Tesseract 언어 데이터

## 4. 통합 테스트

샘플 이미지가 있다면 아래 CLI로 전체 흐름을 검증할 수 있습니다.

```bash
python back_end_detect_resolve_plate/sample_client.py \
    /path/to/sample_plate.jpg \
    --url http://127.0.0.1:8000/api/license-plates
```

성공 시 인식된 번호판과 후보 목록이 JSON 형식으로 출력됩니다. 현재 제공된 가중치에서는 OCR이 빈 문자열을 반환할 수 있으니, ROI 전처리나 가중치 교체를 검토하세요.

## 5. 환경 변수 정리

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `AI_SERVICE_URL` | `http://127.0.0.1:8001` | 백엔드 프록시가 호출할 AI 서비스 URL |
| `AI_REQUEST_TIMEOUT` | `30.0` | 백엔드에서 AI 서비스 호출 시 타임아웃(초) |
| `AI_REQUEST_RETRIES` | `2` | 실패 시 재시도 횟수 |
| `MODEL_BASE_DIR` | `/home/work/llama_young` | AI 서비스가 모델·tessdata를 찾을 기본 경로 |
| `MODEL_PATH` | `models/license_plate/best.pt` | YOLO 가중치 상대 경로(또는 절대 경로) |
| `TESSDATA_DIR` | `tessdata` | tessdata 상대 경로 |
| `PORT` | `8001` | `run_ai_service.sh` 실행 시 사용할 포트 |

필요 시 `.env` 파일을 직접 만들어 `export $(cat .env | xargs)` 형태로 적용해도 됩니다.

## 6. SSH 포워딩 팁

학교 서버(8001)와 로컬 백엔드(8000)를 분리해서 돌릴 경우:

```bash
# 로컬 PC에서 실행
ssh -i ./id_container \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p 53523 \
    -N -L 8001:127.0.0.1:8001 \
    work@127.0.0.1
```

이후 로컬 백엔드는 `http://127.0.0.1:8001`로 AI 서비스에 접속하면 됩니다.

---

이 구조를 이용하면 메인 백엔드 코드와 AI 인퍼런스를 독립적으로 배포 및 확장할 수 있으며, 필요 시 AI 서버만 교체하거나 수평 확장할 수 있습니다.
# EE_final_proj_inf_backend
