#!/usr/bin/env python3
"""Calibrate a fixed Azure Kinect to the RM75 base frame.

The ChArUco board must be rigidly attached to link_7. This program only reads
RM75 feedback; it never enables the robot or sends a motion command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


TELEOP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TELEOP_ROOT.parent
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

from utils.eye_to_hand_calibration import (  # noqa: E402
    rotation_angle_rad,
    solve_eye_to_hand,
)
from utils.wuji_fingertip_kinematics import WujiFingertipKinematics  # noqa: E402


WINDOW_TITLE = "RM75 eye-to-hand calibration  [SPACE] capture  [Q] solve"
DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(data), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def board_definition(args):
    return {
        "dictionary": args.dictionary,
        "squares_x": args.squares_x,
        "squares_y": args.squares_y,
        "square_length_m": args.square_length,
        "marker_length_m": args.marker_length,
        "legacy_pattern": False,
    }


def make_board(definition):
    try:
        dictionary_id = DICTIONARIES[definition["dictionary"]]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ArUco dictionary {definition['dictionary']!r}"
        ) from exc
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (int(definition["squares_x"]), int(definition["squares_y"])),
        float(definition["square_length_m"]),
        float(definition["marker_length_m"]),
        dictionary,
    )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(definition.get("legacy_pattern", False)))
    return board


def generate_board(path, definition, pixels_per_square=200):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".png":
        raise ValueError("The generated board path must end in .png")
    if pixels_per_square < 50:
        raise ValueError("pixels_per_square must be at least 50")
    board = make_board(definition)
    size = (
        int(definition["squares_x"]) * pixels_per_square,
        int(definition["squares_y"]) * pixels_per_square,
    )
    image = board.generateImage(size, marginSize=0, borderBits=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write board image: {path}")
    metadata = {
        **definition,
        "image_file": path.name,
        "image_size_px": list(size),
        "printed_width_m": definition["squares_x"]
        * definition["square_length_m"],
        "printed_height_m": definition["squares_y"]
        * definition["square_length_m"],
        "print_instruction": "Print at 100%/actual size; disable fit-to-page.",
    }
    sidecar = path.with_suffix(".json")
    atomic_write_json(sidecar, metadata)
    return path, sidecar, metadata


def _observation_arrays(samples):
    corners = [
        np.asarray(sample["charuco_corners_px"], np.float32).reshape(-1, 1, 2)
        for sample in samples
    ]
    ids = [
        np.asarray(sample["charuco_ids"], np.int32).reshape(-1, 1)
        for sample in samples
    ]
    return corners, ids


def _estimate_camera_poses(board, corners, ids, camera_matrix, distortion):
    chessboard_corners = np.asarray(board.getChessboardCorners(), np.float64)
    poses = []
    errors = []
    for image_corners, image_ids in zip(corners, ids):
        object_points = chessboard_corners[image_ids.reshape(-1)]
        image_points = image_corners.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            poses.append(None)
            errors.append(float("inf"))
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = np.asarray(tvec).reshape(3)
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        residuals = projected.reshape(-1, 2) - image_points
        errors.append(float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))))
        poses.append(pose)
    return poses, np.asarray(errors, dtype=np.float64)


def _refine_intrinsics(board, corners, ids, image_size, initial_matrix):
    result = cv2.aruco.calibrateCameraCharuco(
        corners,
        ids,
        board,
        tuple(int(value) for value in image_size),
        np.asarray(initial_matrix, np.float64).copy(),
        np.zeros((5, 1), dtype=np.float64),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )
    rms, matrix, distortion = result[:3]
    if not np.isfinite(rms) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("OpenCV returned invalid camera calibration values")
    return float(rms), matrix, distortion


def solve_session(
    session_dir,
    output_path,
    *,
    refine_intrinsics=True,
    max_reprojection_rms_px=1.5,
    minimum_accepted_samples=15,
    max_translation_rms_mm=8.0,
    max_rotation_rms_deg=1.5,
    allow_poor_quality=False,
):
    session_dir = Path(session_dir).expanduser().resolve()
    session_path = session_dir / "samples.json"
    if not session_path.is_file():
        raise FileNotFoundError(f"Calibration session not found: {session_path}")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    samples = session.get("samples", [])
    if len(samples) < minimum_accepted_samples:
        raise RuntimeError(
            f"Session has {len(samples)} samples; at least "
            f"{minimum_accepted_samples} are required"
        )

    board = make_board(session["board"])
    corners, ids = _observation_arrays(samples)
    image_size = tuple(session["image_size_px"])
    factory_matrix = np.asarray(session["factory_camera_matrix"], np.float64)
    distortion = np.zeros((5, 1), dtype=np.float64)
    camera_matrix = factory_matrix.copy()
    camera_rms = None

    if refine_intrinsics:
        camera_rms, camera_matrix, distortion = _refine_intrinsics(
            board, corners, ids, image_size, factory_matrix
        )

    poses, reprojection_errors = _estimate_camera_poses(
        board, corners, ids, camera_matrix, distortion
    )
    accepted = [
        index
        for index, (pose, error) in enumerate(zip(poses, reprojection_errors))
        if pose is not None and error <= max_reprojection_rms_px
    ]

    # A second intrinsic fit prevents rejected blurred/partial views from
    # influencing the final camera model.
    if refine_intrinsics and len(accepted) >= minimum_accepted_samples:
        camera_rms, camera_matrix, distortion = _refine_intrinsics(
            board,
            [corners[index] for index in accepted],
            [ids[index] for index in accepted],
            image_size,
            camera_matrix,
        )
        poses, reprojection_errors = _estimate_camera_poses(
            board, corners, ids, camera_matrix, distortion
        )
        accepted = [
            index
            for index, (pose, error) in enumerate(zip(poses, reprojection_errors))
            if pose is not None and error <= max_reprojection_rms_px
        ]

    rejected = [index for index in range(len(samples)) if index not in accepted]
    if len(accepted) < minimum_accepted_samples:
        report = {
            "solved_at": _utc_now(),
            "status": "insufficient_accepted_samples",
            "accepted_sample_indices": accepted,
            "rejected_sample_indices": rejected,
            "reprojection_rms_px_by_sample": reprojection_errors,
            "maximum_reprojection_rms_px": max_reprojection_rms_px,
        }
        atomic_write_json(session_dir / "calibration_report.json", report)
        raise RuntimeError(
            f"Only {len(accepted)} of {len(samples)} samples passed the "
            f"{max_reprojection_rms_px:.2f} px reprojection threshold; at least "
            f"{minimum_accepted_samples} are required"
        )

    base_gripper = [
        np.asarray(samples[index]["T_base_link7"], np.float64)
        for index in accepted
    ]
    camera_target = [poses[index] for index in accepted]
    best_method, method_results = solve_eye_to_hand(base_gripper, camera_target)
    best = method_results[best_method]
    translation_rms_mm = best["translation_rms_m"] * 1000.0
    translation_max_mm = best["translation_max_m"] * 1000.0
    quality_passed = (
        translation_rms_mm <= max_translation_rms_mm
        and best["rotation_rms_deg"] <= max_rotation_rms_deg
    )

    serializable_methods = {}
    for method, result in method_results.items():
        if "error" in result:
            serializable_methods[method] = result
            continue
        serializable_methods[method] = {
            "T_base_camera": result["T_base_camera"],
            "T_link7_board_mean": result["T_gripper_target_mean"],
            "translation_rms_mm": result["translation_rms_m"] * 1000.0,
            "translation_max_mm": result["translation_max_m"] * 1000.0,
            "rotation_rms_deg": result["rotation_rms_deg"],
            "rotation_max_deg": result["rotation_max_deg"],
            "score": result["score"],
        }

    report = {
        "solved_at": _utc_now(),
        "status": "passed" if quality_passed else "poor_quality",
        "convention": "T_base_camera",
        "selected_method": best_method,
        "T_base_camera": best["T_base_camera"],
        "T_link7_board_mean": best["T_gripper_target_mean"],
        "accepted_sample_indices": accepted,
        "rejected_sample_indices": rejected,
        "reprojection_rms_px_by_sample": reprojection_errors,
        "accepted_reprojection_rms_px": float(
            np.sqrt(np.mean(reprojection_errors[accepted] ** 2))
        ),
        "camera_calibration_rms_px": camera_rms,
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion.reshape(-1),
        "intrinsics_refined": refine_intrinsics,
        "translation_rms_mm": translation_rms_mm,
        "translation_max_mm": translation_max_mm,
        "rotation_rms_deg": best["rotation_rms_deg"],
        "rotation_max_deg": best["rotation_max_deg"],
        "quality_limits": {
            "translation_rms_mm": max_translation_rms_mm,
            "rotation_rms_deg": max_rotation_rms_deg,
            "reprojection_rms_px": max_reprojection_rms_px,
        },
        "methods": serializable_methods,
    }
    report_path = session_dir / "calibration_report.json"
    atomic_write_json(report_path, report)

    if quality_passed or allow_poor_quality:
        calibration = {
            "convention": "T_base_camera",
            "translation_units": "meter",
            "calibrated": True,
            "matrix": best["T_base_camera"],
            "selected_method": best_method,
            "generated_at": _utc_now(),
            "source_session": str(session_dir),
            "quality_passed": quality_passed,
            "translation_rms_mm": translation_rms_mm,
            "rotation_rms_deg": best["rotation_rms_deg"],
            "camera_matrix": camera_matrix,
            "distortion_coefficients": distortion.reshape(-1),
        }
        atomic_write_json(output_path, calibration)

    if not quality_passed and not allow_poor_quality:
        raise RuntimeError(
            "Calibration quality failed: "
            f"translation RMS {translation_rms_mm:.2f} mm "
            f"(limit {max_translation_rms_mm:.2f}), rotation RMS "
            f"{best['rotation_rms_deg']:.3f} deg "
            f"(limit {max_rotation_rms_deg:.3f}). The output calibration was "
            f"not overwritten; see {report_path}"
        )
    return report


def _pose_difference(first, second):
    translation_m = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotation_deg = float(
        np.rad2deg(rotation_angle_rad(first[:3, :3].T @ second[:3, :3]))
    )
    return translation_m, rotation_deg


class RM75StateReader:
    def __init__(self, ip, port):
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except ImportError as exc:
            raise RuntimeError(
                "RM75 SDK is not installed. Install robotic-arm==1.1.6 in the "
                "active kinect environment."
            ) from exc
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.robot.rm_create_robot_arm(ip, port)
        if getattr(self.handle, "id", -1) == -1:
            self.robot.rm_delete_robot_arm()
            self.robot = None
            raise ConnectionError(f"Failed to connect to RM75 at {ip}:{port}")

    def read_joints_deg(self):
        code, state = self.robot.rm_get_current_arm_state()
        if code != 0:
            raise RuntimeError(f"RM75 state read failed with code {code}")
        joints = np.asarray(state.get("joint"), dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"RM75 returned invalid joint state: {joints}")
        return joints

    def close(self):
        if self.robot is not None:
            self.robot.rm_delete_robot_arm()
            self.robot = None


def _detect(detector, image):
    corners, ids, marker_corners, marker_ids = detector.detectBoard(image)
    if ids is None:
        corners = np.empty((0, 1, 2), dtype=np.float32)
        ids = np.empty((0, 1), dtype=np.int32)
    return corners, ids, marker_corners, marker_ids


def _preview_board_pose(board, corners, ids, camera_matrix):
    """Estimate a display-only pose without letting transient detections abort capture."""
    if ids is None or len(ids) < 6:
        return None
    object_points = np.asarray(board.getChessboardCorners())[ids.reshape(-1)]
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            np.asarray(corners).reshape(-1, 2),
            np.asarray(camera_matrix, np.float64),
            np.zeros((5, 1), np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    return (rvec, tvec) if ok else None


def _same_board_definition(first, second):
    exact_keys = ("dictionary", "squares_x", "squares_y", "legacy_pattern")
    if any(first.get(key) != second.get(key) for key in exact_keys):
        return False
    return all(
        np.isclose(float(first.get(key, -1.0)), float(second.get(key, -2.0)))
        for key in ("square_length_m", "marker_length_m")
    )


def capture_session(args):
    stream_client_dir = PROJECT_ROOT / "foundationpose-kinect-0.1.0"
    if str(stream_client_dir) not in sys.path:
        sys.path.insert(0, str(stream_client_dir))
    from stream_client import azure_frames

    definition = board_definition(args)
    board = make_board(definition)
    detector = cv2.aruco.CharucoDetector(board)
    kinematics = WujiFingertipKinematics()
    robot = RM75StateReader(args.robot_ip, args.robot_port)
    frames = None
    try:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        session_dir = (
            Path(args.session_dir).expanduser().resolve()
            if args.session_dir
            else (TELEOP_ROOT / "calibration_runs" / run_name)
        )
        session_path = session_dir / "samples.json"
        image_dir = session_dir / "images"
        if session_path.exists():
            if not args.session_dir:
                raise RuntimeError(
                    f"Generated session path already exists: {session_dir}"
                )
            session = json.loads(session_path.read_text(encoding="utf-8"))
            if not _same_board_definition(session.get("board", {}), definition):
                raise RuntimeError(
                    "Cannot resume because command-line board parameters do not "
                    f"match {session_path}. Stored board: {session.get('board')}; "
                    f"requested board: {definition}"
                )
            image_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Resuming session with {len(session.get('samples', []))} "
                f"captured samples: {session_dir}"
            )
        else:
            image_dir.mkdir(parents=True, exist_ok=False)
            session = {
                "format_version": 1,
                "created_at": _utc_now(),
                "robot": {"model": "RM75", "ip": args.robot_ip, "port": args.robot_port},
                "board_mount": "rigidly_attached_to_link_7",
                "board": definition,
                "factory_camera_matrix": None,
                "image_size_px": None,
                "samples": [],
            }
        if len(session.get("samples", [])) >= args.samples:
            print(
                f"Session already has the requested {len(session['samples'])} samples."
            )
            return session_dir
        frames = azure_frames(fps=args.fps, warmup=args.warmup)
        print(f"Session: {session_dir}")
        print("Move with the teach pendant, stop completely, then press SPACE.")
        print("This program reads feedback only and never sends robot motion commands.")
        for color_bgr, _depth_mm, _depth_scale, camera_matrix in frames:
            height, width = color_bgr.shape[:2]
            if session["factory_camera_matrix"] is None:
                session["factory_camera_matrix"] = np.asarray(
                    camera_matrix, np.float64
                )
                session["image_size_px"] = [width, height]
                atomic_write_json(session_path, session)
            else:
                if session.get("image_size_px") != [width, height]:
                    raise RuntimeError(
                        "Cannot resume with a different Kinect color resolution: "
                        f"stored {session.get('image_size_px')}, current {[width, height]}"
                    )
                if not np.allclose(
                    np.asarray(session["factory_camera_matrix"], np.float64),
                    np.asarray(camera_matrix, np.float64),
                    atol=1.0,
                    rtol=1e-3,
                ):
                    raise RuntimeError(
                        "Cannot resume because the Kinect color intrinsics changed"
                    )

            corners, ids, marker_corners, marker_ids = _detect(
                detector, color_bgr
            )
            display = color_bgr.copy()
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            if len(ids):
                cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
                preview_pose = _preview_board_pose(
                    board, corners, ids, camera_matrix
                )
                if preview_pose is not None:
                    rvec, tvec = preview_pose
                    cv2.drawFrameAxes(
                        display,
                        np.asarray(camera_matrix, np.float64),
                        np.zeros((5, 1), np.float64),
                        rvec,
                        tvec,
                        args.square_length * 2.0,
                    )
            status = (
                f"samples {len(session['samples'])}/{args.samples} | "
                f"corners {len(ids)}/{args.min_corners}"
            )
            color = (40, 210, 40) if len(ids) >= args.min_corners else (30, 30, 230)
            cv2.putText(
                display, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                if len(session["samples"]) < args.minimum_accepted_samples:
                    print(
                        f"Need at least {args.minimum_accepted_samples} captures before solving."
                    )
                    continue
                break
            if key != 32:
                continue
            if len(ids) < args.min_corners:
                print(f"Rejected: only {len(ids)} ChArUco corners are visible.")
                continue

            joints_before = robot.read_joints_deg()
            time.sleep(args.stability_interval)
            joints_after = robot.read_joints_deg()
            motion_deg = float(np.max(np.abs(joints_after - joints_before)))
            if motion_deg > args.max_static_joint_delta_deg:
                print(
                    f"Rejected: robot moved {motion_deg:.3f} deg during the "
                    f"stability check (limit {args.max_static_joint_delta_deg:.3f})."
                )
                continue
            joints = (joints_before + joints_after) * 0.5
            T_base_link7 = kinematics.arm_link_transform(joints, "link_7")
            duplicate = False
            for previous in session["samples"]:
                distance_m, angle_deg = _pose_difference(
                    np.asarray(previous["T_base_link7"], np.float64), T_base_link7
                )
                if (
                    distance_m < args.min_pose_translation_mm / 1000.0
                    and angle_deg < args.min_pose_rotation_deg
                ):
                    duplicate = True
                    print(
                        f"Rejected: near an existing pose ({distance_m * 1000:.1f} mm, "
                        f"{angle_deg:.1f} deg)."
                    )
                    break
            if duplicate:
                continue

            index = len(session["samples"])
            image_name = f"{index:03d}.png"
            if not cv2.imwrite(str(image_dir / image_name), color_bgr):
                raise RuntimeError(f"Failed to save {image_dir / image_name}")
            sample = {
                "index": index,
                "captured_at": _utc_now(),
                "monotonic_time_ns": time.monotonic_ns(),
                "image_file": f"images/{image_name}",
                "joint_angles_deg": joints,
                "static_joint_delta_deg": motion_deg,
                "T_base_link7": T_base_link7,
                "charuco_corners_px": corners.reshape(-1, 2),
                "charuco_ids": ids.reshape(-1),
            }
            session["samples"].append(sample)
            atomic_write_json(session_path, session)
            print(
                f"Captured {index + 1}/{args.samples}: {len(ids)} corners, "
                f"static delta {motion_deg:.3f} deg"
            )
            if len(session["samples"]) >= args.samples:
                break
    finally:
        if frames is not None and hasattr(frames, "close"):
            frames.close()
        robot.close()
        cv2.destroyAllWindows()
    return session_dir


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fixed Azure Kinect to RM75 base_link eye-to-hand calibration"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-board", metavar="PNG")
    mode.add_argument("--solve-session", metavar="DIR")
    parser.add_argument("--robot-ip", default="192.0.2.1")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--fps", type=int, choices=(5, 15, 30), default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--session-dir")
    parser.add_argument(
        "--output", default=str(TELEOP_ROOT / "kinect_hand_eye.json")
    )
    parser.add_argument("--dictionary", choices=tuple(DICTIONARIES), default="DICT_5X5_100")
    parser.add_argument("--squares-x", type=int, default=8)
    parser.add_argument("--squares-y", type=int, default=6)
    parser.add_argument("--square-length", type=float, default=0.040)
    parser.add_argument("--marker-length", type=float, default=0.030)
    parser.add_argument("--pixels-per-square", type=int, default=200)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--minimum-accepted-samples", type=int, default=15)
    parser.add_argument("--max-reprojection-rms-px", type=float, default=1.5)
    parser.add_argument("--no-refine-intrinsics", action="store_true")
    parser.add_argument("--max-translation-rms-mm", type=float, default=8.0)
    parser.add_argument("--max-rotation-rms-deg", type=float, default=1.5)
    parser.add_argument("--allow-poor-quality", action="store_true")
    parser.add_argument("--min-pose-translation-mm", type=float, default=15.0)
    parser.add_argument("--min-pose-rotation-deg", type=float, default=5.0)
    parser.add_argument("--stability-interval", type=float, default=0.10)
    parser.add_argument("--max-static-joint-delta-deg", type=float, default=0.10)
    return parser


def _validate_args(args, parser):
    if args.squares_x < 3 or args.squares_y < 3:
        parser.error("ChArUco board must have at least 3 squares in each direction")
    if not 0.0 < args.marker_length < args.square_length:
        parser.error("marker length must be positive and smaller than square length")
    if args.samples < args.minimum_accepted_samples:
        parser.error("--samples must be >= --minimum-accepted-samples")
    if args.min_corners < 4:
        parser.error("--min-corners must be at least 4")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    definition = board_definition(args)
    if args.generate_board:
        board_path, sidecar, metadata = generate_board(
            args.generate_board, definition, args.pixels_per_square
        )
        print(f"Board image: {board_path}")
        print(f"Board metadata: {sidecar}")
        print(
            f"Physical size: {metadata['printed_width_m'] * 1000:.1f} x "
            f"{metadata['printed_height_m'] * 1000:.1f} mm"
        )
        print("Print at actual size (100%); disable fit-to-page and verify with a ruler.")
        return 0

    if args.solve_session:
        session_dir = Path(args.solve_session)
    else:
        session_dir = capture_session(args)
    report = solve_session(
        session_dir,
        args.output,
        refine_intrinsics=not args.no_refine_intrinsics,
        max_reprojection_rms_px=args.max_reprojection_rms_px,
        minimum_accepted_samples=args.minimum_accepted_samples,
        max_translation_rms_mm=args.max_translation_rms_mm,
        max_rotation_rms_deg=args.max_rotation_rms_deg,
        allow_poor_quality=args.allow_poor_quality,
    )
    print(f"Selected method: {report['selected_method']}")
    print(
        f"Quality: {report['status']}; translation RMS "
        f"{report['translation_rms_mm']:.2f} mm, rotation RMS "
        f"{report['rotation_rms_deg']:.3f} deg"
    )
    print(f"T_base_camera written to: {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConnectionError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
