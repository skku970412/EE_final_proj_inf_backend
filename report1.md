# Plate Recognition Async Wrapper Update

## Summary
- `차량번호판인식_yolov8/yeonsup_차량.py`를 FastAPI 백엔드에서 재사용할 수 있도록 비동기 친화적인 래퍼로 전환했습니다.
- 로컬 `venv`와 `tessdata`를 우선 사용하도록 환경 정리를 유지하면서, 기본 `MODEL_BASE_DIR`·`MODEL_PATH`를 스크립트 폴더 기준으로 자동 설정합니다.
- `ai.app.model.PlateRecognizer`를 직접 사용해 YOLO + Tesseract 파이프라인을 공유하고, `recognize_bytes_async`, `recognize_path_async`, `recognize_and_callback` API를 제공하도록 정리했습니다.
- CLI에 `--callback`(POST 콜백), `--indent` 옵션을 추가하고, 요약 정보를 출력한 뒤 JSON 응답을 그대로 보여줍니다.

## How to Use
- CLI: `python yeonsup_차량.py [이미지경로] --callback http://localhost:8000/hook`  
  (이미지 경로를 생략하면 `samples/example.jpg`를 사용하며, 콜백은 선택 사항)
- Python 코드에서:  
  ```python
  from yeonsup_차량 import recognize_bytes_async
  result = await recognize_bytes_async(image_bytes)
  ```
- 콜백은 `requests.post(url, json=result)`로 동작하므로, 수신 서버는 JSON 본문을 받도록 준비하면 됩니다.

## Verification
- `python yeonsup_차량.py --help` 실행으로 새 CLI가 정상적으로 구동되고 환경 정리/설정이 적용되는 것까지 확인했습니다.
- 실제 추론은 YOLO 가중치 용량과 시간 문제 때문에 이번 작업에서 돌려보지 못했습니다. 가중치(`runs/license_plate_yolov8n/weights/best.pt`)가 준비되어 있으면 동일 명령으로 바로 검증 가능합니다.

## Next Steps
1. YOLO 가중치와 Tesseract 리소스가 있는 환경에서 `python yeonsup_차량.py samples/example.jpg`로 엔드 투 엔드 추론을 검증하세요.
2. 백엔드 프록시(`back_end_detect_resolve_plate`)에서 새 async 함수들을 import하여, 로컬 처리와 AI 서비스 중 원하는 경로를 선택적으로 사용하도록 연동할 수 있습니다.
3. 콜백 URL을 실제 메인 백엔드 엔드포인트로 설정하고, 응답 스키마가 기대대로 맞는지 통합 테스트를 수행하면 좋습니다.
