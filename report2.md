# Plate Recognition E2E Run (samples/example.jpg)

## Setup
- 위치: `/home/work/llama_young/차량번호판인식_yolov8`
- 명령어: `python yeonsup_차량.py samples/example.jpg`
- 사전 준비: `sudo apt-get install -y tesseract-ocr` (OCR 바이너리 미설치로 인한 오류 해결)
- 모델 가중치: `runs/license_plate_yolov8n/weights/best.pt`
- Tesseract 데이터: `tessdata/kor.traineddata` 등 자동 다운로드 확인

## 결과
- 추론 요약: `번호판: 03두2902 | score=3 | conf=0.543 | time_ms=detection_ms=826.2/ocr_ms=10860.0/total_ms=11773.7`
- JSON 출력(발췌):
  ```json
  {
    "plate": "03두2902",
    "score": 3,
    "confidence": 0.5434941649436951,
    "bbox": [486.0, 1748.0, 2194.0, 2595.0],
    "timing": {
      "detection_ms": 826.18,
      "ocr_ms": 10860.00,
      "total_ms": 11773.69
    }
  }
  ```
- 후보 OCR 조합 중 `OTSU` + `psm=8`이 최종 번호판 문자열을 일치시키며, 다른 조합은 잡음 또는 부분 문자열만 반환함.

## 관찰 사항
- 최초 실행 시 `pytesseract.TesseractNotFoundError`가 발생했으며, 시스템 패키지 `tesseract-ocr` 설치 후 정상 동작함.
- YOLO 추론(~0.8초)에 비해 Tesseract 기반 OCR 단계가 상대적으로 오래 걸림(~10.9초). 멀티스레드 실행 또는 후보 축소로 최적화 가능성 있음.
- 출력된 bbox는 원본 이미지 내 번호판 위치 `[x1, y1, x2, y2]`를 의미하므로, 필요한 경우 시각화에 활용 가능.

## 다음 단계 제안
1. 동일 스크립트를 비동기 백엔드 워커에 붙여 실서비스 요청을 처리할 때도 동일 결과가 나오는지 확인.
2. OCR 시간을 줄이기 위해 후보 바이너리·PSM 조합을 조정하거나, Tesseract 호출을 병렬 처리하는 방안 검토.
3. 다양한 샘플 이미지로 추가 테스트를 수행해 추론 일관성과 실패 케이스를 파악. (테스트 이미지가 늘어나면 `report3.md`, `report4.md` 등으로 기록 권장)
