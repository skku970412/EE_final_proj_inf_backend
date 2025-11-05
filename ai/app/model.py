from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from itertools import combinations

from .config import settings
from .utils import decode_image, ensure_site_packages, ensure_tessdata

YOLO_LOCK = Lock()


@dataclass
class DetectionResult:
    bbox: Tuple[float, float, float, float]
    confidence: float
    crop_rgb: np.ndarray


def _is_hangul_char(ch: str) -> bool:
    return "\uAC00" <= ch <= "\uD7A3"


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    dp = list(range(len_b + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            temp = dp[j]
            if ca == cb:
                dp[j] = prev
            else:
                dp[j] = min(prev, dp[j - 1], dp[j]) + 1
            prev = temp
    return dp[len_b]


class PlateRecognizer:
    """Wrap YOLO detection + Tesseract OCR pipeline."""

    def __init__(self) -> None:
        base_dir = settings.model_base_dir

        venv_dir = base_dir / "venv"
        if venv_dir.exists():
            ensure_site_packages(venv_dir)

        ensure_tessdata(base_dir / settings.tessdata_dir, langs=("kor", "eng", "osd"))

        model_path = (
            settings.model_path
            if settings.model_path.is_absolute()
            else base_dir / settings.model_path
        )
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO 가중치를 찾을 수 없습니다: {model_path}")

        with YOLO_LOCK:
            self.model = YOLO(str(model_path))
        self.confidence = settings.detection_conf

    # ------------------------- Detection ------------------------- #

    def _detect(self, rgb: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        results = self.model.predict(
            source=rgb, conf=self.confidence, verbose=False, imgsz=640
        )
        elapsed = (time.perf_counter() - start) * 1000
        if not results:
            raise ValueError("YOLO 결과가 비어 있습니다.")

        result = results[0]
        if not len(result.boxes):
            raise ValueError("탐지된 번호판이 없습니다.")

        # 최고 confidence 박스 선택
        idx = int(np.argmax(result.boxes.conf.cpu().numpy()))
        box = result.boxes.xyxy.cpu().numpy()[idx]
        conf = float(result.boxes.conf.cpu().numpy()[idx])

        x1, y1, x2, y2 = map(int, box)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("탐지된 박스에서 이미지를 잘라낼 수 없습니다.")

        return DetectionResult(
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            confidence=conf,
            crop_rgb=crop,
        ), elapsed

    # ------------------------- Warp helpers ------------------------- #

    def _warp_plate_from_aligned(
        self,
        bgr: np.ndarray,
        target_w: int = 900,
        inner: float = 0.08,
        canny: Tuple[int, int] = (40, 140),
        area_frac: Tuple[float, float] = (0.06, 0.98),
        ar_range: Tuple[float, float] = (2.2, 7.5),
        eps: float = 0.03,
        dilate_iter: int = 2,
    ):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)

        med = np.median(gray)
        auto_low = int(max(0, 0.66 * med))
        auto_high = int(min(255, 1.33 * med))
        low, high = (min(canny[0], auto_low), max(canny[1], auto_high))

        edges = cv2.Canny(gray, low, high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), dilate_iter)

        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = bgr.shape[:2]
        area_img = h * w

        best = None
        best_score = -1.0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            area = cv2.contourArea(approx)
            frac = area / (area_img + 1e-6)
            if not (area_frac[0] <= frac <= area_frac[1]):
                continue

            pts = approx.reshape(-1, 2).astype(np.float32)
            sums = pts.sum(axis=1)
            diffs = np.diff(pts, axis=1).ravel()
            tl = pts[np.argmin(sums)]
            br = pts[np.argmax(sums)]
            tr = pts[np.argmin(diffs)]
            bl = pts[np.argmax(diffs)]
            quad = np.array([tl, tr, bl, br], np.float32)

            w1 = hypot(*(tr - tl))
            w2 = hypot(*(br - bl))
            h1 = hypot(*(bl - tl))
            h2 = hypot(*(br - tr))
            max_w = max(w1, w2)
            max_h = max(h1, h2)
            if max_h <= 1:
                continue
            ar = max_w / max_h
            if not (ar_range[0] <= ar <= ar_range[1]):
                continue

            score = frac * ar
            if score > best_score:
                best_score = score
                best = quad

        if best is None:
            ys, xs = np.where(edges > 0)
            if len(xs) == 0:
                return bgr, False, edges, None
            rect = cv2.minAreaRect(np.column_stack((xs, ys)))
            best = cv2.boxPoints(rect).astype(np.float32)

        target_h = int(target_w / 4.5)
        dst = np.float32(
            [[0, 0], [target_w - 1, 0], [0, target_h - 1], [target_w - 1, target_h - 1]]
        )
        M = cv2.getPerspectiveTransform(best, dst)
        warped = cv2.warpPerspective(bgr, M, (target_w, target_h))
        pad = int(inner * min(target_w, target_h))
        warped = warped[pad : target_h - pad, pad : target_w - pad]

        return warped, True, edges, best

    def _inner_warp_from_quad(
        self,
        img_bgr: np.ndarray,
        quad_src: np.ndarray,
        inner_ratio: float = 0.08,
        target_w: int = 900,
    ):
        target_h = int(target_w / 4.5)
        dst_rect = np.float32(
            [[0, 0], [target_w - 1, 0], [0, target_h - 1], [target_w - 1, target_h - 1]]
        )
        H_src2dst = cv2.getPerspectiveTransform(quad_src, dst_rect)
        H_dst2src = np.linalg.inv(H_src2dst)

        r = float(inner_ratio)
        inner_dst = np.float32(
            [
                [r * target_w, r * target_h],
                [(1 - r) * target_w - 1, r * target_h],
                [r * target_w, (1 - r) * target_h - 1],
                [(1 - r) * target_w - 1, (1 - r) * target_h - 1],
            ]
        )
        inner_src = cv2.perspectiveTransform(inner_dst[None, :, :], H_dst2src)[
            0
        ].astype(np.float32)

        M = cv2.getPerspectiveTransform(inner_src, dst_rect)
        warped = cv2.warpPerspective(img_bgr, M, (target_w, target_h))
        return warped, inner_src

    # ------------------------- OCR helpers ------------------------- #

    def _prepare_ocr_inputs(self, plate_bgr: np.ndarray):
        def inner_crop_xy(img_bgr: np.ndarray, ratio_x: float = 0.02, ratio_y: float = 0.0):
            h, w = img_bgr.shape[:2]
            pad_x = int(ratio_x * w)
            pad_y = int(ratio_y * h)
            x_start = max(0, pad_x)
            x_end = max(x_start + 1, w - pad_x)
            y_start = max(0, pad_y)
            y_end = max(y_start + 1, h - pad_y)
            return img_bgr[y_start:y_end, x_start:x_end]

        def pad_white(img_bgr: np.ndarray, ratio: float = 0.025):
            h, w = img_bgr.shape[:2]
            p = int(ratio * max(h, w))
            return cv2.copyMakeBorder(
                img_bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )

        roi = inner_crop_xy(plate_bgr, ratio_x=0.02, ratio_y=0.0)
        roi_pad = pad_white(roi, ratio=0.025)

        gray_pad = cv2.cvtColor(roi_pad, cv2.COLOR_BGR2GRAY)
        gray_pad = cv2.resize(gray_pad, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
        gray_pad = cv2.createCLAHE(2.0, (8, 8)).apply(gray_pad)
        _, otsu_pad = cv2.threshold(gray_pad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        gray = cv2.cvtColor(roi_pad, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

        _, b1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        b2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        b3 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 4
        )
        k3 = np.ones((3, 3), np.uint8)
        b4 = cv2.morphologyEx(b2, cv2.MORPH_CLOSE, k3, 1)

        bins = [
            ("OTSU_PAD", otsu_pad),
            ("OTSU", b1),
            ("ADP31_C2", b2),
            ("ADP41_C4", b3),
            ("CLOSE_ADP", b4),
        ]

        return roi_pad, bins

    def _run_ocr(self, roi_pad: np.ndarray, bins) -> Dict[str, Any]:
        plate_re = re.compile(r"\b\d{2,3}[가-힣]-?\d{4}\b")
        allowed_kor = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자하허호"
        subs = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "D": "0", "—": "-"})
        candidates: List[Dict[str, Any]] = []

        best: Dict[str, Any] = {"txt": "", "score": -1, "raw": "", "name": "", "img": None, "psm": 7}

        for name, b in bins:
            for psm in (7, 8, 6, 13):
                cfg = (
                    f"--oem 1 --psm {psm} -l kor+eng "
                    f"-c tessedit_do_invert=0 -c load_system_dawg=0 -c load_freq_dawg=0 "
                    f"-c tessedit_char_whitelist={allowed_kor}0123456789-"
                )
                raw = pytesseract.image_to_string(b, config=cfg)
                txt = re.sub(r"[^0-9가-힣- ]", "", raw.translate(subs)).strip().replace(" ", "")
                score = 3 if plate_re.fullmatch(txt) else (2 if plate_re.search(txt) else 0)
                candidates.append({"text": txt, "score": score, "bin": name, "psm": psm})

                if score > best["score"]:
                    best.update(txt=txt, score=score, raw=raw.strip(), name=name, img=b, psm=psm)
                elif score == best["score"] and score > 0:
                    candidate_len = len(txt.replace("-", ""))
                    best_len = len(best["txt"].replace("-", "")) if best["txt"] else 0
                    if candidate_len and (not best_len or candidate_len < best_len):
                        best.update(txt=txt, score=score, raw=raw.strip(), name=name, img=b, psm=psm)

        refined = self._refine_plate_text(candidates)
        if refined:
            candidates.append(
                {
                    "text": refined["plate"],
                    "score": 3,
                    "bin": "POST",
                    "psm": -1,
                }
            )
            best.update(
                txt=refined["plate"],
                score=max(best["score"], 3),
                raw=refined.get("source_text", refined["plate"]),
                name=refined.get("source_bin", best.get("name", "")),
                refined_from=refined.get("source_text", ""),
            )

        return {"best": best, "candidates": candidates}

    # ------------------------- Public API ------------------------- #

    def predict(self, data: bytes) -> Dict[str, Any]:
        rgb = decode_image(data)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        detection, det_ms = self._detect(rgb)
        timings["detection_ms"] = det_ms

        flat, ok, _, quad = self._warp_plate_from_aligned(bgr=detection.crop_rgb)
        if not ok or quad is None:
            raise ValueError("번호판 투시 변환에 실패했습니다.")

        flat_cropped_bgr, _ = self._inner_warp_from_quad(
            detection.crop_rgb.copy(), quad.astype(np.float32), inner_ratio=0.08, target_w=900
        )

        roi_pad, bins = self._prepare_ocr_inputs(flat_cropped_bgr)
        ocr_start = time.perf_counter()
        ocr_result = self._run_ocr(roi_pad, bins)
        timings["ocr_ms"] = (time.perf_counter() - ocr_start) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000

        best = ocr_result["best"]
        plate = best["txt"].replace("-", "") if best["score"] > 0 else ""

        return {
            "plate": plate,
            "score": best["score"],
            "raw_text": best["raw"],
            "bbox": detection.bbox,
            "confidence": detection.confidence,
            "candidates": ocr_result["candidates"],
            "timing": timings,
        }

    @staticmethod
    def _generate_variants(base: str) -> List[Tuple[str, int]]:
        clean = re.sub(r"[^0-9가-힣]", "", base)
        if not clean:
            return []
        length = len(clean)
        max_remove = min(2, length)
        variants: Dict[str, int] = {clean: 0}
        for k in range(1, max_remove + 1):
            for remove_idx in combinations(range(length), k):
                variant = "".join(clean[i] for i in range(length) if i not in remove_idx)
                if variant:
                    prev = variants.get(variant)
                    if prev is None or k < prev:
                        variants[variant] = k
        additional: Dict[str, int] = {}
        for variant, cost in list(variants.items()):
            if variant and variant[0] != "0":
                pref_variant = "0" + variant
                pref_cost = cost + 1
                if pref_variant not in variants or pref_cost < variants[pref_variant]:
                    additional[pref_variant] = pref_cost
        variants.update(additional)
        return [(variant, cost) for variant, cost in variants.items()]

    @staticmethod
    def _finalize_variant(variant: str) -> List[Tuple[str, int]]:
        clean = "".join(ch for ch in variant if ch.isdigit() or _is_hangul_char(ch))
        if not clean:
            return []
        hangul_positions = [idx for idx, ch in enumerate(clean) if _is_hangul_char(ch)]
        results: List[Tuple[str, int]] = []
        seen: set[str] = set()
        for idx in hangul_positions:
            hangul = clean[idx]
            left_digits_full = "".join(ch for ch in clean[:idx] if ch.isdigit())
            right_digits_full = "".join(ch for ch in clean[idx + 1 :] if ch.isdigit())
            if not right_digits_full:
                continue

            left_options: List[Tuple[str, int]] = []
            if len(left_digits_full) >= 2:
                left_options.append((left_digits_full[-2:], max(0, len(left_digits_full) - 2)))
                if len(left_digits_full) >= 3:
                    left_options.append((left_digits_full[-3:], max(0, len(left_digits_full) - 3)))
            else:
                left_options.append((left_digits_full.rjust(2, "0"), 2 - len(left_digits_full)))

            if len(right_digits_full) < 4:
                continue

            for start in range(len(right_digits_full) - 3):
                segment = right_digits_full[start : start + 4]
                drop = start + (len(right_digits_full) - (start + 4))
                for left, left_penalty in left_options:
                    candidate = left + hangul + segment
                    extra_cost = left_penalty + drop
                    if candidate not in seen:
                        seen.add(candidate)
                        results.append((candidate, extra_cost))

        return results

    def _refine_plate_text(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not candidates:
            return None

        base_entries: List[Dict[str, Any]] = []
        hangul_chars: set[str] = set()
        digits_only_strings: List[str] = []

        for cand in candidates:
            raw_text = cand.get("text") or ""
            clean_text = re.sub(r"[^0-9가-힣]", "", raw_text)
            if not clean_text:
                continue
            score = int(cand.get("score", 0))
            entry = {
                "clean": clean_text,
                "raw": raw_text,
                "score": score,
                "bin": cand.get("bin"),
                "psm": cand.get("psm"),
            }
            base_entries.append(entry)
            if any(_is_hangul_char(ch) for ch in clean_text):
                hangul_chars.update(ch for ch in clean_text if _is_hangul_char(ch))
            else:
                digits_only_strings.append(clean_text)

        if not base_entries:
            return None

        for hangul in hangul_chars:
            for digits in digits_only_strings:
                if len(digits) < 3:
                    continue
                upper_bound = min(len(digits), 4)
                for pos in range(1, upper_bound):
                    combined = digits[:pos] + hangul + digits[pos:]
                    base_entries.append(
                        {"clean": combined, "raw": combined, "score": 0, "bin": "SYN", "psm": -1}
                    )

        plate_pattern = re.compile(r"^\d{2,3}[가-힣]\d{4}$")

        best_pick: Dict[str, Any] | None = None
        best_priority: Tuple[int, int, int, int, str] | None = None

        for entry in base_entries:
            variants = self._generate_variants(entry["clean"])
            for variant, base_cost in variants:
                for candidate_text, extra_cost in self._finalize_variant(variant):
                    if not plate_pattern.fullmatch(candidate_text):
                        continue
                    edit_cost = _edit_distance(candidate_text, entry["clean"])
                    total_cost = base_cost * 10 + extra_cost * 5 + edit_cost
                    priority = (
                        -entry["score"],
                        total_cost,
                        edit_cost,
                        len(candidate_text),
                        candidate_text,
                    )
                    if best_priority is None or priority < best_priority:
                        best_priority = priority
                        best_pick = {
                            "plate": candidate_text,
                            "source_text": entry.get("raw", candidate_text),
                            "source_bin": entry.get("bin"),
                            "total_cost": total_cost,
                        }

        return best_pick
