# Local Inference Test (FastAPI + Backend) Summary

## 서비스 기동
- 프로젝트: `/home/work/llama_young/back_end_detect_resolve_plate`
- 명령: `VENV_PATH=./venv ./scripts/run_all.sh > run_all.log 2>&1 &`
- 상태: AI 서버(8001)가 정상 기동, 백엔드는 기존 uvicorn 프로세스가 포트 8000을 잡고 있어서 기동 대기 상태. AI 단독으로는 health 체크 200 OK 확인.

## 직접 API 호출
- 명령:
  ```bash
  curl -X POST http://127.0.0.1:8001/v1/recognize \
       -F "image=@../차량번호판인식_yolov8/samples/example.jpg" \
       -o response.json
  ```
- 결과 파일(`response.json`):
  ```json
  {
    "plate": "",
    "score": 0,
    "raw_text": "",
    "bbox": [19.0, 0.0, 2630.0, 2937.0],
    "confidence": 0.6946427822113037,
    "candidates": [... 모두 score=0 ...],
    "timing": {
      "detection_ms": 1195.24,
      "ocr_ms": 3900.33,
      "total_ms": 5354.01
    }
  }
  ```
- 관찰: YOLO가 큰 박스로 전체 차량을 감지한 뒤 OCR 후보 전부 빈 문자열 → 기존과 동일하게 번호판 텍스트가 나오지 않음.

## 포트 충돌 메모
- run_all.sh 실행 시 포트 8000이 이미 다른 uvicorn 인스턴스에 의해 사용 중이면 백엔드가 더 이상 기동하지 못함.
- 해결책: `lsof -t -i :8000,:8001`로 기존 프로세스를 종료 후 스크립트 재기동.

## 다음 조치 제안
1. OCR 실패 원인을 파악하기 위해 ROI 시각화 및 전처리 파라미터 조정 검토.
2. 백엔드 프록시를 함께 쓰려면 기존 uvicorn PID를 종료하고 run_all.sh를 다시 실행하거나, 백엔드 포트를 다른 값으로 변경.
