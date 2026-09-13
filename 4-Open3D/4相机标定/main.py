#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from camera_calibration_open3d_common import (
    CameraExtrinsicResult,
    CameraObservationSet,
    CameraParams,
    DetectionConfig,
    FmObservation,
    FrameMarkerObservation,
    collect_image_paths,
    detect_translation_outliers,
    format_vec,
    fuse_quats_linear_like_cpp,
    matrix_from_quat_wxyz,
    parse_fm_frame_observations,
    quat_xyzw_from_matrix,
    write_multi_cam_yaml,
)


@dataclass
class MarkerPreset:
    marker_id: str
    camera_id: str
    t_jig_fm: np.ndarray
    q_jig_fm_wxyz: tuple[float, float, float, float]


@dataclass
class MarkerOnJigConfig:
    marker_id: str
    marker_code: str
    R_fm0_fm: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t_fm0_fm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class CameraBindingConfig:
    camera_id: str
    marker_code: str
    camera_params: CameraParams


@dataclass
class MultiCamConfig:
    detection_cfg: DetectionConfig
    R_base_fm0: np.ndarray
    t_base_fm0: np.ndarray
    jig_markers: dict[str, MarkerOnJigConfig] = field(default_factory=dict)
    cameras: list[CameraBindingConfig] = field(default_factory=list)
    outlier_translation_m: float = 0.05
    min_frames_per_camera: int = 10


def build_detection_config() -> DetectionConfig:
    return DetectionConfig(
        code_type="FRACTAL_MARKER_2L",
        marker_size=0.252,
        filter_window=10,
        enable_undistort=True,
        fmlib_undistort=False,
        pyramid_detection=False,
        reproject_error_threshold_px=20.0,
    )

# 写死并返回fm0标定板在机器人base坐标系下的位姿
def build_base_fm0_pose() -> tuple[np.ndarray, np.ndarray]:
    t_base_fm0 = np.array(
        [0.79303114144469611, -0.0016941422208361923, 0.1897624219271071],
        dtype=np.float64,
    )
    R_base_fm0 = matrix_from_quat_wxyz(
        0.50152001280199776,
        0.49629413218682672,
        -0.49581701090982316,
        -0.50629566738068887,
    )
    return R_base_fm0, t_base_fm0


def build_camera_intrinsics() -> dict[str, CameraParams]:
    intrinsics = [
        478.77838134765625,
        0.0,
        324.2511291503906,
        0.0,
        478.9640808105469,
        240.33447265625,
        0.0,
        0.0,
        1.0,
    ]
    distortion = [
        -0.017814787104725838,
        0.005230664741247892,
        0.00024115070118568838,
        0.00015762064140290022,
        0.0,
    ]
    intr_map: dict[str, CameraParams] = {}
    for key in ("front_camera", "left_camera", "right_camera", "back_camera"):
        intr_map[key] = CameraParams(key, intrinsics[:], distortion[:], True)
    return intr_map


def map_camera_id_to_intrinsics_key(camera_id: str) -> str:
    return {
        "cam_front": "front_camera",
        "cam_left": "left_camera",
        "cam_right": "right_camera",
        "cam_back": "back_camera",
    }.get(camera_id, "")


def map_camera_id_to_image_folder(camera_id: str) -> str:
    return {
        "cam_front": "front_image",
        "cam_left": "left_image",
        "cam_right": "right_image",
        "cam_back": "back_image",
    }.get(camera_id, "")


def build_fixture_and_bindings(
    code_type: str,
    intr_map: dict[str, CameraParams],
) -> tuple[dict[str, MarkerOnJigConfig], list[CameraBindingConfig]] | None:
    markers = [
        MarkerPreset("fm0", "cam_front", np.array([0.795, 0.0, 0.185]), (-0.5, -0.5, 0.5, 0.5)),
        MarkerPreset("fm1", "cam_right", np.array([0.0, -0.795, 0.185]), (0.0, 0.0, 0.70710678, 0.70710678)),
        MarkerPreset("fm2", "cam_back", np.array([-0.795, 0.0, 0.185]), (0.5, 0.5, 0.5, 0.5)),
        MarkerPreset("fm3", "cam_left", np.array([0.0, 0.795, 0.185]), (0.70710678, 0.70710678, 0.0, 0.0)),
    ]  # 4个marker在夹具jig坐标系下的位置和姿态

    marker_map: dict[str, MarkerPreset] = {}
    seen_cameras: set[str] = set()
    for marker in markers:
        if marker.camera_id in seen_cameras:
            return None
        seen_cameras.add(marker.camera_id)
        marker_map[marker.marker_id] = marker
    if "fm0" not in marker_map:
        return None

    fm0 = marker_map["fm0"]
    R_jig_fm0 = matrix_from_quat_wxyz(*fm0.q_jig_fm_wxyz)   # 表示fm0在jig里的旋转
    R_fm0_jig = R_jig_fm0.T     # 从jig转到fm0

    jig_markers: dict[str, MarkerOnJigConfig] = {}
    cameras: list[CameraBindingConfig] = []
    for marker_id in sorted(marker_map):
        marker = marker_map[marker_id]
        R_jig_fm = matrix_from_quat_wxyz(*marker.q_jig_fm_wxyz)
        marker_code = f"{code_type}_{marker.marker_id[2:]}"

        marker_cfg = MarkerOnJigConfig(
            marker_id=marker.marker_id,
            marker_code=marker_code,
            R_fm0_fm=R_fm0_jig @ R_jig_fm,
            t_fm0_fm=R_fm0_jig @ (marker.t_jig_fm - fm0.t_jig_fm),
        )
        jig_markers[marker_code] = marker_cfg

        intr_key = map_camera_id_to_intrinsics_key(marker.camera_id)
        if intr_key not in intr_map:
            return None
        cameras.append(CameraBindingConfig(marker.camera_id, marker_code, intr_map[intr_key]))
    # jig_markers:每个 marker在标定夹具上，相对于基准marker fm0,在哪里、朝向是什么？
    # cameras: 这个相机应该看哪个marker
    return jig_markers, cameras


def build_config() -> MultiCamConfig | None:
    R_base_fm0, t_base_fm0 = build_base_fm0_pose()
    cfg = MultiCamConfig(
        detection_cfg=build_detection_config(),
        R_base_fm0=R_base_fm0,
        t_base_fm0=t_base_fm0,
        outlier_translation_m=0.05,
        min_frames_per_camera=10,
    )
    built = build_fixture_and_bindings(cfg.detection_cfg.code_type, build_camera_intrinsics())
    if built is None:
        return None
    cfg.jig_markers, cfg.cameras = built
    return cfg


def load_buffered_frames(
    cameras: list[CameraBindingConfig],
    image_base_dir: str,
    max_frames: int,
) -> dict[str, list[Path]]:
    frames_by_camera: dict[str, list[Path]] = {}
    for cam in cameras:
        folder = map_camera_id_to_image_folder(cam.camera_id)
        if not folder:
            continue
        frames: list[Path] = []
        for image_path in collect_image_paths(Path(image_base_dir) / folder):
            if max_frames > 0 and len(frames) >= max_frames:
                break
            frames.append(image_path)
        frames_by_camera[cam.camera_id] = frames
    return frames_by_camera


def validate_buffered_input(
    cameras: list[CameraBindingConfig],
    frames_by_camera: dict[str, list[Path]],
    min_frames_per_camera: int,
) -> bool:
    for cam in cameras:
        if cam.camera_id not in frames_by_camera:
            print(f"[ERROR] Missing buffered data for camera: {cam.camera_id}", file=sys.stderr)
            return False
        count = len(frames_by_camera[cam.camera_id])
        if count < min_frames_per_camera:
            print(
                f"[ERROR] Camera {cam.camera_id} has insufficient frames: "
                f"{count} < {min_frames_per_camera}",
                file=sys.stderr,
            )
            return False
    return True


def find_precomputed_result_file(result_dir: Path, cam: CameraBindingConfig) -> Path | None:
    candidates = [
        result_dir / f"{cam.camera_id}.json",
        result_dir / f"{map_camera_id_to_image_folder(cam.camera_id)}.json",
        result_dir / f"{map_camera_id_to_intrinsics_key(cam.camera_id)}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_precomputed_fm_results(
    cameras: list[CameraBindingConfig],
    result_dir: str | Path,
) -> dict[str, dict[int, list[FmObservation]]] | None:
    result_root = Path(result_dir)
    results: dict[str, dict[int, list[FmObservation]]] = {}
    for cam in cameras:
        result_file = find_precomputed_result_file(result_root, cam)
        if result_file is None:
            print(f"[ERROR] Missing precomputed FM result for camera: {cam.camera_id}", file=sys.stderr)
            return None
        try:
            results[cam.camera_id] = parse_fm_frame_observations(result_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            print(f"[ERROR] Failed to load precomputed FM result {result_file}: {exc}", file=sys.stderr)
            return None
    return results


def build_runtime_states(
    cfg: MultiCamConfig,
    precomputed_results: dict[str, dict[int, list[FmObservation]]],
):
    states = {}
    for cam in cfg.cameras:
        states[cam.camera_id] = {
            "config": cam,
            "observations": CameraObservationSet(cam.camera_id, cam.marker_code),
            "precomputed_results": precomputed_results[cam.camera_id],
        }
    return states


def process_frame(state, image_path: Path, frame_index: int) -> bool:
    del image_path
    observations = state["precomputed_results"].get(frame_index, [])
    if not observations:
        return False

    marker_code_for_geometry = state["config"].marker_code
    selected = None
    for obs in observations:
        if obs.code_str == marker_code_for_geometry:
            selected = obs
            break
    if selected is None:
        selected = max(observations, key=lambda obs: obs.confidence)

    state["observations"].frames.append(
        FrameMarkerObservation(
            frame_index=frame_index,
            marker_code=marker_code_for_geometry,
            R_cam_fm=selected.R_cam_marker,
            t_cam_fm=selected.t_cam_marker,
        )
    )
    return True


def fuse_camera_observations(obs: CameraObservationSet, outlier_translation_m: float) -> bool:
    translations = [frame.t_cam_fm for frame in obs.frames]
    is_outlier = detect_translation_outliers(translations, outlier_translation_m, min_size=2)
    obs.num_outliers = sum(1 for item in is_outlier if item)

    valid_quats = []
    valid_translations = []
    for frame, outlier in zip(obs.frames, is_outlier):
        if outlier:
            continue
        valid_quats.append(quat_xyzw_from_matrix(frame.R_cam_fm))
        valid_translations.append(frame.t_cam_fm)
    if not valid_quats:
        return False

    obs.num_valid_frames = len(valid_quats)
    q_avg = fuse_quats_linear_like_cpp(valid_quats)
    obs.R_cam_fm_fused = matrix_from_quat_wxyz(q_avg[3], q_avg[0], q_avg[1], q_avg[2])
    obs.t_cam_fm_fused = np.mean(np.asarray(valid_translations, dtype=np.float64), axis=0)
    return True


def get_R_link_to_optical() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def compute_camera_extrinsic(
    camera_id: str,
    obs: CameraObservationSet,
    cfg: MultiCamConfig,
) -> CameraExtrinsicResult:
    result = CameraExtrinsicResult(
        camera_id=camera_id,
        marker_code=obs.marker_code,
        success=False,
        num_observations=obs.num_valid_frames,
        num_outliers=obs.num_outliers,
    )
    marker_cfg = cfg.jig_markers.get(obs.marker_code)
    if marker_cfg is None:
        result.error_msg = f"marker config not found: {obs.marker_code}"
        return result

    R_fm_cam = obs.R_cam_fm_fused.T
    t_fm_cam = -R_fm_cam @ obs.t_cam_fm_fused
    result.R_base_camera_optical = cfg.R_base_fm0 @ marker_cfg.R_fm0_fm @ R_fm_cam
    result.t_base_camera_optical = (
        cfg.R_base_fm0 @ (marker_cfg.R_fm0_fm @ t_fm_cam + marker_cfg.t_fm0_fm)
        + cfg.t_base_fm0
    )

    R_opt_link = get_R_link_to_optical().T
    result.R_base_camera_link = result.R_base_camera_optical @ R_opt_link
    result.t_base_camera_link = result.t_base_camera_optical
    result.success = True
    return result


def default_fm_result_dir() -> str:
    return str(Path(__file__).resolve().parent / "data" / "fm_code_json")


def run_calibration(image_base_dir: str, output_path: str, fm_result_dir: str) -> bool:
    cfg = build_config()
    if cfg is None:
        print("[ERROR] Failed to build fixture and camera bindings", file=sys.stderr)
        return False

    frames_by_camera = load_buffered_frames(cfg.cameras, image_base_dir, 50)
    if not validate_buffered_input(cfg.cameras, frames_by_camera, cfg.min_frames_per_camera):
        return False

    precomputed_results = load_precomputed_fm_results(cfg.cameras, fm_result_dir)
    if precomputed_results is None:
        print(
            "[ERROR] Generate JSON with camera_fm_detect_example --folder --json first, "
            "then pass the directory via --fm-result-dir.",
            file=sys.stderr,
        )
        return False
    print(f"[INFO] Using precomputed FM results from: {fm_result_dir}")

    states = build_runtime_states(cfg, precomputed_results)
    for cam in cfg.cameras:
        accepted = 0
        for i, image_path in enumerate(frames_by_camera[cam.camera_id]):
            if process_frame(states[cam.camera_id], image_path, i):
                accepted += 1
        print(
            f"[INFO] Camera {cam.camera_id} processed={len(frames_by_camera[cam.camera_id])} "
            f"accepted={accepted}"
        )

    results: list[CameraExtrinsicResult] = []
    for camera_id in sorted(states):
        obs = states[camera_id]["observations"]
        if len(obs.frames) < cfg.min_frames_per_camera:
            print(f"[ERROR] Calibration failed: insufficient frames for camera: {camera_id}", file=sys.stderr)
            return False
        if not fuse_camera_observations(obs, cfg.outlier_translation_m):
            print(f"[ERROR] Calibration failed: fusion failed for camera: {camera_id}", file=sys.stderr)
            return False
        cam_result = compute_camera_extrinsic(camera_id, obs, cfg)
        if not cam_result.success:
            print(
                f"[ERROR] Calibration failed: extrinsic computation failed for camera: {camera_id}",
                file=sys.stderr,
            )
            return False
        results.append(cam_result)

    print("[INFO] success=true")
    for cam in results:
        q = quat_xyzw_from_matrix(cam.R_base_camera_link)
        print(
            f"  camera={cam.camera_id} marker={cam.marker_code} "
            f"obs={cam.num_observations} outliers={cam.num_outliers} "
            f"t=[{format_vec(cam.t_base_camera_link, 6)}] q=[{format_vec(q, 6)}]"
        )
    write_multi_cam_yaml(output_path, results)
    print(f"[INFO] Result saved to: {output_path}")
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage="%(prog)s [image_base_dir] [output_result.yaml]")
    parser.add_argument(
        "image_base_dir",
        nargs="?",
        default="/Users/jodocls/Desktop/code/my_hub/4-Open3D/4相机标定/data/fm_code",
        type=str,
    )
    parser.add_argument(
        "output_result",
        nargs="?",
        default="./multi_cam_result.yaml",
        type=str,
    )
    parser.add_argument(
        "--fm-result-dir",
        default=default_fm_result_dir(),
        type=str,
        help="Directory containing precomputed C++ FM JSON results, e.g. cam_front.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return 0 if run_calibration(args.image_base_dir, args.output_result, args.fm_result_dir) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


# 如果要得到6D位姿，为什么不用点云,而是用FM码库