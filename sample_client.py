from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict

import httpx


async def send_request(image_path: Path, url: str, timeout: float) -> Dict[str, Any]:
    content_type, _ = mimetypes.guess_type(image_path.name)
    content_type = content_type or "application/octet-stream"

    async with httpx.AsyncClient(timeout=timeout) as client:
        with image_path.open("rb") as file_handle:
            response = await client.post(
                url,
                files={"image": (image_path.name, file_handle, content_type)},
            )
        response.raise_for_status()
        return response.json()


async def _async_main(args: argparse.Namespace) -> None:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    try:
        result = await send_request(image_path, args.url, args.timeout)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise SystemExit(
            f"요청 실패(status={exc.response.status_code}): {detail}"
        ) from exc
    except httpx.RequestError as exc:
        raise SystemExit(f"백엔드 연결 실패: {exc}") from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="백엔드 프록시를 통해 번호판 인식 결과를 받아옵니다.",
    )
    parser.add_argument(
        "image",
        help="전송할 번호판 이미지 경로",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/api/license-plates",
        help="백엔드 프록시 엔드포인트 URL (기본: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="요청 타임아웃(초)",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(_async_main(parse_args()))


if __name__ == "__main__":
    main()
