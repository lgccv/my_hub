#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@dataclass
class DetectionConfig:
    code_type: str = "FRACTAL_MARKER_2L"
    marker_size: float = 0.252
    filter_window: int = 10
    enable_undistort: bool = True
    fmlib_undistort: bool = False
    pyramid_detection: bool = False
    reproject_error_threshold_px: float | None = None


@dataclass
class CameraParams:
    camera_name: str
    intrinsics: list[float]
    distortion: list[float]
    loaded: bool = True


@dataclass
class FmObservation:
    code_str: str
    R_cam_marker: np.ndarray
    t_cam_marker: np.ndarray
    confidence: float = 0.0


@dataclass
class FrameMarkerObservation:
    frame_index: int
    marker_code: str
    R_cam_fm: np.ndarray
    t_cam_fm: np.ndarray


@dataclass
class CameraObservationSet:
    camera_id: str
    marker_code: str
    frames: list[FrameMarkerObservation] = field(default_factory=list)
    R_cam_fm_fused: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t_cam_fm_fused: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    num_valid_frames: int = 0
    num_outliers: int = 0


@dataclass
class CameraExtrinsicResult:
    camera_id: str
    marker_code: str
    success: bool
    error_msg: str = ""
    num_observations: int = 0
    num_outliers: int = 0
    R_base_camera_link: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t_base_camera_link: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    R_base_camera_optical: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t_base_camera_optical: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_open3d() -> None:
    if o3d is None:
        raise RuntimeError("Python package 'open3d' is required to run this example.")


def read_image_with_open3d(image_path: str | Path):
    if o3d is not None:
        image = o3d.io.read_image(str(image_path))
        if np.asarray(image).size == 0:
            return None
        return image
    if cv2 is None:
        raise RuntimeError("Python package 'open3d' or 'cv2' is required to run this example.")
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        return None
    return image


def collect_image_paths(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        return []
    paths: list[Path] = []
    for child in p.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(child)
    return sorted(paths, key=lambda item: str(item))


def collect_open3d_readable_image_paths(path: str | Path) -> list[Path]:
    readable: list[Path] = []
    for image_path in collect_image_paths(path):
        if read_image_with_open3d(image_path) is not None:
            readable.append(image_path)
    return readable


def compose_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def normalize_quat_xyzw(q: Iterable[float]) -> np.ndarray:
    quat = np.asarray(list(q), dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def quat_xyzw_from_matrix(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quat_xyzw([x, y, z, w])


def matrix_from_quat_xyzw(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_from_quat_wxyz(w: float, x: float, y: float, z: float) -> np.ndarray:
    return matrix_from_quat_xyzw([x, y, z, w])


def slerp_quat_xyzw(lhs: np.ndarray, rhs: np.ndarray, t: float) -> np.ndarray:
    lhs = normalize_quat_xyzw(lhs)
    rhs = normalize_quat_xyzw(rhs)
    dot = float(np.dot(lhs, rhs))
    if dot < 0.0:
        rhs = -rhs
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return normalize_quat_xyzw(lhs + t * (rhs - lhs))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    scale_lhs = math.sin(theta_0 - theta) / sin_theta_0
    scale_rhs = math.sin(theta) / sin_theta_0
    return normalize_quat_xyzw(scale_lhs * lhs + scale_rhs * rhs)


def fuse_quats_slerp_like_cpp(quats: list[np.ndarray]) -> np.ndarray:
    q_avg = normalize_quat_xyzw(quats[0])
    for i, q in enumerate(quats[1:], start=1):
        q = normalize_quat_xyzw(q)
        if float(np.dot(q, q_avg)) < 0.0:
            q = -q
        q_avg = slerp_quat_xyzw(q_avg, q, 1.0 / float(i + 1))
    return normalize_quat_xyzw(q_avg)


def fuse_quats_linear_like_cpp(quats: list[np.ndarray]) -> np.ndarray:
    q_ref = normalize_quat_xyzw(quats[0])
    q_sum = np.zeros(4, dtype=np.float64)
    for q in quats:
        q = normalize_quat_xyzw(q)
        if float(np.dot(q_ref, q)) < 0.0:
            q = -q
        q_sum += q
    return normalize_quat_xyzw(q_sum / float(len(quats)))


def detect_translation_outliers(translations: list[np.ndarray], threshold_m: float, min_size: int) -> list[bool]:
    is_outlier = [False] * len(translations)
    if len(translations) < min_size:
        return is_outlier
    mean_t = np.mean(np.asarray(translations, dtype=np.float64), axis=0)
    for i, t in enumerate(translations):
        if float(np.linalg.norm(t - mean_t)) > threshold_m:
            is_outlier[i] = True
    return is_outlier


def format_vec(values: Iterable[float], precision: int = 9) -> str:
    return ", ".join(f"{float(v):.{precision}f}" for v in values)


def format_yaml_float(value: float) -> str:
    return f"{float(value):.17g}"


def _parse_fm_observation_items(items: Iterable[dict]) -> list[FmObservation]:
    observations: list[FmObservation] = []
    for item in items:
        q_xyzw = item["q"]
        observations.append(
            FmObservation(
                code_str=item["code"],
                confidence=float(item.get("confidence", 0.0)),
                t_cam_marker=np.asarray(item["t"], dtype=np.float64),
                R_cam_marker=matrix_from_quat_xyzw(q_xyzw),
            )
        )
    return observations


def parse_fm_observations(stdout: str) -> list[FmObservation]:
    payload = json.loads(stdout)
    return _parse_fm_observation_items(payload.get("observations", []))


def parse_fm_frame_observations(stdout: str) -> dict[int, list[FmObservation]]:
    payload = json.loads(stdout)
    if "frames" not in payload:
        frame_index = int(payload.get("frame_index", 0))
        return {frame_index: _parse_fm_observation_items(payload.get("observations", []))}

    frames: dict[int, list[FmObservation]] = {}
    for frame in payload.get("frames", []):
        frame_index = int(frame["frame_index"])
        frames[frame_index] = _parse_fm_observation_items(frame.get("observations", []))
    return frames


def find_default_detector_binary() -> Path | None:
    for env_name in ("CAMERA_FM_DETECT_EXAMPLE", "SP_CAMERA_FM_DETECT_EXAMPLE"):
        value = os.environ.get(env_name)
        if value:
            path = Path(value)
            if path.is_file() and os.access(path, os.X_OK):
                return path

    root = repo_root()
    candidates = [
        root / "build" / "example" / "camera_fm_detect_example",
        root / "cmake-build-release" / "example" / "camera_fm_detect_example",
        root / "cmake-build-debug" / "example" / "camera_fm_detect_example",
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


class FmCliDetector:
    """Open3D Python image loading plus the repo's native FM detector executable."""

    def __init__(
        self,
        camera_params: CameraParams,
        detection_cfg: DetectionConfig,
        detector_binary: str | Path | None = None,
    ) -> None:
        self.camera_params = camera_params
        self.detection_cfg = detection_cfg
        self.detector_binary = Path(detector_binary) if detector_binary else find_default_detector_binary()

    def detect(self, image_path: str | Path, frame_index: int = -1) -> list[FmObservation] | None:
        del frame_index
        if read_image_with_open3d(image_path) is None:
            return None
        if self.detector_binary is None:
            raise RuntimeError(
                "FM detector backend not found. Build camera_fm_detect_example or set "
                "CAMERA_FM_DETECT_EXAMPLE=/path/to/camera_fm_detect_example."
            )
        cmd = [
            str(self.detector_binary),
            "--camera-id",
            self.camera_params.camera_name,
            "--single",
            str(image_path),
            "--json",
        ]
        env = os.environ.copy()
        if self.detection_cfg.reproject_error_threshold_px is not None:
            env["FM_REPROJECT_ERROR_TH"] = str(self.detection_cfg.reproject_error_threshold_px)
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
        if proc.returncode != 0:
            details = (proc.stderr + "\n" + proc.stdout).strip()
            raise RuntimeError(f"FM detector backend failed for {image_path}: {details}")
        return parse_fm_observations(proc.stdout)


def print_observations(frame_index: int, observations: list[FmObservation]) -> None:
    print(f"[FRAME {frame_index}] detected={len(observations)}")
    for obs in observations:
        q = quat_xyzw_from_matrix(obs.R_cam_marker)
        print(
            f"  code={obs.code_str} conf={obs.confidence:.9f} "
            f"t=[{format_vec(obs.t_cam_marker, 9)}] q=[{format_vec(q, 9)}]"
        )


def write_base_jig_yaml(
    output_path: str | Path,
    R_base_fm0: np.ndarray,
    t_base_fm0: np.ndarray,
    num_frames_used: int,
    num_outliers: int,
    num_frames_processed: int,
    num_frames_accepted: int,
) -> None:
    q = quat_xyzw_from_matrix(R_base_fm0)
    content = (
        "calibration_type: base_jig_localization\n"
        "result:\n"
        "  T_base_fm0:\n"
        "    quat:\n"
        f"      x: {format_yaml_float(q[0])}\n"
        f"      y: {format_yaml_float(q[1])}\n"
        f"      z: {format_yaml_float(q[2])}\n"
        f"      w: {format_yaml_float(q[3])}\n"
        "    xyz:\n"
        f"      x: {format_yaml_float(t_base_fm0[0])}\n"
        f"      y: {format_yaml_float(t_base_fm0[1])}\n"
        f"      z: {format_yaml_float(t_base_fm0[2])}\n"
        f"  num_frames_used: {num_frames_used}\n"
        f"  num_outliers: {num_outliers}\n"
        f"  num_frames_processed: {num_frames_processed}\n"
        f"  num_frames_accepted: {num_frames_accepted}\n"
    )
    Path(output_path).write_text(content, encoding="utf-8")


def write_multi_cam_yaml(output_path: str | Path, cameras: list[CameraExtrinsicResult]) -> None:
    lines = [
        "calibration_type: multi_cam_extrinsic",
        f"timestamp: {int(time.time())}",
        "success: true",
        "result:",
        "  cameras:",
    ]
    for cam in cameras:
        q = quat_xyzw_from_matrix(cam.R_base_camera_link)
        lines.extend(
            [
                "    - camera_id: " + cam.camera_id,
                "      marker_code: " + cam.marker_code,
                f"      num_observations: {cam.num_observations}",
                f"      num_outliers: {cam.num_outliers}",
                "      success: true",
                "      T_base_camera_link:",
                "        quat:",
                f"          x: {format_yaml_float(q[0])}",
                f"          y: {format_yaml_float(q[1])}",
                f"          z: {format_yaml_float(q[2])}",
                f"          w: {format_yaml_float(q[3])}",
                "        xyz:",
                f"          x: {format_yaml_float(cam.t_base_camera_link[0])}",
                f"          y: {format_yaml_float(cam.t_base_camera_link[1])}",
                f"          z: {format_yaml_float(cam.t_base_camera_link[2])}",
            ]
        )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
