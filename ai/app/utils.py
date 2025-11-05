from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Iterable

import cv2
import numpy as np


TESSDATA_LOCK = Lock()


def ensure_site_packages(venv_dir: Path) -> None:
    """Remove foreign site-packages so that bundled venv is preferred."""

    site_candidates = sorted((venv_dir / "lib").glob("python*/site-packages"))
    if not site_candidates:
        return

    site_packages_path = site_candidates[0].resolve()
    venv_root = site_packages_path.parent.parent.resolve()

    foreign_paths: list[str] = []
    for entry in list(sys.path):
        if (
            ("site-packages" in entry or "dist-packages" in entry)
            and not entry.startswith(str(venv_root))
        ):
            foreign_paths.append(entry)
            sys.path.remove(entry)
    if foreign_paths:
        print("외부 site-packages 제거:", *foreign_paths, sep="\n  ")

    if str(site_packages_path) not in sys.path:
        sys.path.insert(0, str(site_packages_path))

    bin_path = (venv_dir / "bin").resolve()
    current_path = os.environ.get("PATH", "")
    if current_path:
        parts = current_path.split(":")
        if str(bin_path) not in parts:
            os.environ["PATH"] = f"{bin_path}:{current_path}"
    else:
        os.environ["PATH"] = str(bin_path)


def ensure_tessdata(tessdata_dir: Path, langs: Iterable[str]) -> None:
    """Download/copy Tesseract language data if missing."""

    with TESSDATA_LOCK:
        tessdata_dir.mkdir(parents=True, exist_ok=True)

        default_tessdata = Path("/usr/share/tesseract-ocr/4.00/tessdata")
        for lang in langs:
            dst = tessdata_dir / f"{lang}.traineddata"
            if dst.exists():
                continue
            if lang == "kor":
                url = (
                    "https://github.com/tesseract-ocr/tessdata_best/raw/main/kor.traineddata"
                )
                print("한국어 tessdata 다운로드 중...", url)
                urllib.request.urlretrieve(url, dst)
                print("한국어 tessdata 다운로드 완료:", dst)
            else:
                src = default_tessdata / f"{lang}.traineddata"
                if src.exists():
                    shutil.copy2(src, dst)

        os.environ["TESSDATA_PREFIX"] = str(tessdata_dir.resolve())
        os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
        print("TESSDATA_PREFIX 설정 완료:", os.environ["TESSDATA_PREFIX"])


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB numpy array."""

    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("이미지 디코딩에 실패했습니다.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

