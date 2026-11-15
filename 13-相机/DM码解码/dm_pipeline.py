#!/usr/bin/env python3
"""Standalone DataMatrix/QR image pipeline.

This script mirrors the important stages of the SROS dml-lib path in one file:
  image -> grayscale/invert/resize/pad -> candidate ROIs -> perspective warp
  -> decode -> confidence/pose -> JSON result and optional debug image.

DataMatrix decoding requires pylibdmtx plus the native libdmtx library. If it is
not available, the script still runs and tries OpenCV's QRCodeDetector. The
candidate detector is intentionally dependency-light; replace detect_candidates
with a model inference block when the private SROS detector model is available.

Examples:
  python dm_pipeline.py QRcode_detection.png
  python dm_pipeline.py image.png --dmtx-size 40 --camera-json camera.json
  python dm_pipeline.py image.png --debug-dir debug --json result.json

camera.json example:
  {"fx": 1200, "fy": 1200, "cx": 640, "cy": 360,
   "width": 0.04, "height": 0.04}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
except Exception:
    dmtx_decode = None


@dataclass
class Candidate:
    points: np.ndarray
    source: str
    score: float = 0.0


@dataclass
class Detection:
    value: str
    confidence: float
    bbox: list[int]
    corners: list[list[float]]
    source: str
    pose_camera: Optional[dict[str, float]] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-file DM/QR recognition pipeline")
    p.add_argument("image", type=Path, help="input image")
    p.add_argument("--json", dest="json_path", type=Path, help="write result JSON")
    p.add_argument("--debug-dir", type=Path, help="write grayscale/ROI/debug images")
    p.add_argument("--dmtx-size", type=float, default=None, help="DM physical side length in metres")
    p.add_argument("--camera-json", type=Path, help="camera.json containing fx,fy,cx,cy")
    p.add_argument("--scale", type=float, default=1.0, help="image scale before detection")
    p.add_argument("--padding", type=int, default=32, help="border padding in pixels")
    p.add_argument("--no-invert", action="store_true", help="do not try inverted images")
    return p.parse_args()


def load_camera(path: Optional[Path]) -> Optional[dict[str, float]]:
    if not path:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("fx", "fy", "cx", "cy")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"camera JSON missing: {', '.join(missing)}")
    return {k: float(v) for k, v in data.items()}


def preprocess(image: np.ndarray, scale: float, padding: int, invert: bool) -> tuple[np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # Preserve edge candidates, like the padding in DmCodeReader.
    gray = cv2.copyMakeBorder(gray, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    work = cv2.bitwise_not(gray) if invert else gray
    return gray, work


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(s)], points[np.argmin(d)], points[np.argmax(s)], points[np.argmax(d)]], dtype=np.float32)


def _quad_from_contour(contour: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    area = abs(cv2.contourArea(approx))
    if area < 100:
        return None
    return _order_points(approx.reshape(4, 2))


def detect_candidates(gray: np.ndarray) -> list[Candidate]:
    """Find square/rectangular candidates without the private SROS model."""
    candidates: list[Candidate] = []
    for threshold_mode in ("adaptive", "otsu"):
        if threshold_mode == "adaptive":
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 7)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
            quad = _quad_from_contour(contour)
            if quad is None:
                continue
            x, y, w, h = cv2.boundingRect(quad.astype(np.float32))
            ratio = min(w, h) / max(w, h)
            if ratio < 0.45:
                continue
            score = float(ratio * min(1.0, cv2.contourArea(contour) / max(1.0, w * h)))
            candidates.append(Candidate(quad, "contour", score))
    # De-duplicate candidates whose centers overlap.
    unique: list[Candidate] = []
    for c in sorted(candidates, key=lambda x: x.score, reverse=True):
        center = c.points.mean(axis=0)
        if all(np.linalg.norm(center - u.points.mean(axis=0)) > 12 for u in unique):
            unique.append(c)
    return unique[:30]


def warp_candidate(gray: np.ndarray, points: np.ndarray, size: int = 320) -> np.ndarray:
    src = _order_points(points)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)


def decode_roi(roi: np.ndarray) -> list[tuple[str, float, str]]:
    results: list[tuple[str, float, str]] = []
    variants = [("normal", roi), ("inverted", cv2.bitwise_not(roi))]
    # DataMatrix decoding through libdmtx; OpenCV QR is a useful fallback.
    if dmtx_decode is not None:
        for label, img in variants:
            try:
                decoded = dmtx_decode(img, timeout=200, max_count=8)
            except Exception:
                decoded = []
            for item in decoded:
                raw = bytes(item.data).decode("utf-8", errors="replace").strip("\\x00\\r\\n")
                if raw:
                    results.append((raw, 0.95, f"libdmtx-{label}"))
    qr = cv2.QRCodeDetector()
    for label, img in variants:
        try:
            text, points, _ = qr.detectAndDecode(img)
        except Exception:
            text, points = "", None
        if text:
            results.append((text, 0.85, f"opencv-qr-{label}"))
    return results


def estimate_pose(corners: np.ndarray, camera: Optional[dict[str, float]], size_m: Optional[float]) -> Optional[dict[str, float]]:
    if camera is None or not size_m:
        return None
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    half = size_m / 2.0
    object_points = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], dtype=np.float32)
    image_points = _order_points(corners).astype(np.float32)
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, k, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return {"x_m": float(tvec[0]), "y_m": float(tvec[1]), "z_m": float(tvec[2]),
            "roll_rad": float(roll), "pitch_rad": float(pitch), "yaw_rad": float(yaw)}


def draw_debug(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    out = image.copy()
    for d in detections:
        pts = np.array(d.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 0, 255), 3)
        x, y, _, _ = d.bbox
        cv2.putText(out, f"{d.value} [{d.confidence:.2f}]", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {args.image}")
    camera = load_camera(args.camera_json)
    gray, work = preprocess(image, args.scale, args.padding, invert=False)
    candidates = detect_candidates(work)
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.debug_dir / "01_gray.png"), gray)
        cv2.imwrite(str(args.debug_dir / "02_preprocessed.png"), work)
    detections: list[Detection] = []
    for i, candidate in enumerate(candidates):
        roi = warp_candidate(gray, candidate.points)
        decoded = decode_roi(roi)
        if args.debug_dir:
            cv2.imwrite(str(args.debug_dir / f"roi_{i:03d}.png"), roi)
        if not decoded:
            continue
        value, confidence, source = max(decoded, key=lambda x: x[1])
        x, y, w, h = cv2.boundingRect(candidate.points.astype(np.float32))
        detections.append(Detection(value=value, confidence=confidence, bbox=[x, y, w, h],
                                    corners=candidate.points.round(3).tolist(), source=source,
                                    pose_camera=estimate_pose(candidate.points, camera, args.dmtx_size)))
    result: dict[str, Any] = {
        "image": str(args.image),
        "status": "SUCCESS" if detections else "NOT_FOUND",
        "decoder_available": dmtx_decode is not None,
        "candidate_count": len(candidates),
        "detections": [asdict(d) for d in detections],
        "notes": [
            "This is a standalone approximation of the SROS dml-lib pipeline.",
            "The private neural detector and SROS coordinate transforms are not embedded.",
        ],
    }
    if args.debug_dir:
        cv2.imwrite(str(args.debug_dir / "03_result.png"), draw_debug(image, detections))
        result["debug_dir"] = str(args.debug_dir)
    return result


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_path:
        args.json_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
