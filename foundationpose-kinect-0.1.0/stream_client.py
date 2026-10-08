#!/usr/bin/env python3
"""Real-time streaming client for the apple 6DoF pipeline.

Captures live from a RealSense camera (depth aligned to color) OR replays a
recorded RGB-D directory (same layout as tools/capture_realsense.py).
Pushes frames through an SSH direct-tcpip tunnel to pipeline_server.py on the
server, receives the 4x4 pose + overlay JPEG, and displays it live in a window.

Camera AWB: on startup the RealSense auto-white-balance takes ~1s to converge,
so the first `--warmup` frames are discarded before the first (SAM3/reset) frame
is sent -- otherwise the initial segmentation sees a color-shifted image.

Recording: pass --record to write the received overlay frames (the pose result
visualization) to a video file. "auto" names it local_test/run_<timestamp>.mp4.

Keys in the window:
    r   re-run SAM3 + re-register (reset tracking)
    q / ESC   quit

Usage (local, splatsim env, has pyrealsense2 + paramiko).
Prefer the bat launcher over `conda run`: conda run buffers ALL child output until
the process exits (you see no logs while it runs) and adds ~3.5s startup.
    # live camera, show window + record overlay video:
    start_client.bat --record auto
    # live camera, window only, discard 30 frames for AWB convergence:
    start_client.bat --warmup 30
    # replay recorded frames (no camera needed):
    start_client.bat --replay apple_data_v2 --step 2
"""
import argparse
import ctypes.util
import json
import os
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
# NOTE: paramiko is imported lazily inside open_tunnel() to keep startup fast.

HOST = "10.13.21.34"
SSH_PORT = 20303
USER = "root"
PASSWORD = "Guanchao@123"
SERVER_TARGET = ("127.0.0.1", 5555)  # pipeline_server on the server side

WINDOW_TITLE = "apple 6DoF  [r]=re-segment [q]=quit"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def send_frame(sock, header, jpeg, depth_u16):
    header["j"] = len(jpeg)  # server needs jpeg length to split from depth
    hb = json.dumps(header).encode()
    payload = hb + b"\x00" + jpeg + depth_u16.tobytes()
    sock.sendall(struct.pack(">II", len(payload), len(hb)) + payload)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return buf


def recv_frame(sock):
    pre = recv_exact(sock, 8)
    plen, hlen = struct.unpack(">II", pre)
    payload = recv_exact(sock, plen)
    header = json.loads(payload[:hlen])
    jpeg = payload[hlen + 1:]
    return header, jpeg


def open_tunnel():
    import paramiko  # lazy: ~0.7s import only when actually needed
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=SSH_PORT, username=USER, password=PASSWORD, timeout=20)
    chan = ssh.get_transport().open_channel("direct-tcpip", SERVER_TARGET, ("127.0.0.1", 0))
    chan.settimeout(60)
    log(f"SSH tunnel open {HOST}:{SSH_PORT} -> {SERVER_TARGET}")
    return ssh, chan


def replay_frames(data_dir, step=1):
    """Yield (rgb_bgr, depth_u16, d_scale, K) from a recorded dir."""
    d = Path(data_dir)
    intr = json.load(open(d / "intrinsics.json"))
    K = np.array(intr["K"], np.float64)
    rgb_files = sorted((d / "rgb").glob("*.png"))
    log(f"replay: {len(rgb_files)} rgb frames, step={step}, from {data_dir}")
    for f in rgb_files[::step]:
        i = f.stem
        rgb = cv2.imread(str(f))
        depth_m = np.load(d / "depth" / f"{i}.npy")
        depth_u16 = (depth_m / 0.001).astype(np.uint16)
        yield rgb, depth_u16, 0.001, K


def _rs_wait(pipe, label, timeout_ms=8000, retries=3):
    """wait_for_frames with a generous timeout + a few retries.

    The camera can stall briefly under USB-bandwidth pressure or during AWB
    convergence; a short retry recovers from transient stalls. A persistent
    failure is reported with actionable hints instead of a bare RuntimeError.
    """
    last = None
    for i in range(1, retries + 1):
        try:
            return pipe.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError as e:
            last = e
            log(f"[{label}] frame wait timed out ({i}/{retries}, "
                f"{timeout_ms/1000:.0f}s each): {e}")
    log(f"[{label}] giving up after {retries*timeout_ms/1000:.0f}s with no frames. "
        f"Likely causes: USB3 cable/port not actually USB3 (720p color + 848x480 "
        f"depth @{30}fps need the bandwidth), another app holding the camera, or "
        f"the device unplugged. Tip: try a direct USB3 port / --fps 15.")
    raise last


def live_frames(color_w=1280, color_h=720, fps=30, warmup=20):
    """Start the camera immediately and return a generator yielding
    (rgb_bgr, depth_u16, d_scale, K).

    The camera AWB takes ~1s to converge, so `warmup` frames are discarded. To
    hide that latency the discard runs in a background thread WHILE the caller
    opens the SSH tunnel; the generator joins the warmup thread before its first
    frame so `wait_for_frames()` is never called concurrently. If the warmup
    thread fails (camera delivering no frames), that is reported honestly
    rather than a fake "warmup done", and the main loop retries with a timeout.
    """
    import pyrealsense2 as rs  # lazy, only in live mode
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, color_w, color_h, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float64)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    log(f"RealSense started: color {color_w}x{color_h}@{fps}fps, depth_scale={scale:.6f} m/unit")
    log(f"color K = fx={intr.fx:.2f} fy={intr.fy:.2f} ppx={intr.ppx:.2f} ppy={intr.ppy:.2f}")

    warm_err = {"exc": None}

    def _warm():
        try:
            for _ in range(warmup):
                pipe.wait_for_frames(timeout_ms=15000)
        except Exception as e:  # noqa: BLE001 - report, don't kill the app
            warm_err["exc"] = e

    warm_t = None
    if warmup > 0:
        warm_t = threading.Thread(target=_warm, daemon=True)
        warm_t.start()
        log(f"AWB warmup: discarding {warmup} frames in background (parallel with SSH)")

    def gen():
        nonlocal warm_t
        try:
            while True:
                if warm_t is not None:
                    warm_t.join()
                    warm_t = None
                    if warm_err["exc"] is None:
                        log("AWB warmup done, image is color-stable")
                    else:
                        log(f"WARN: warmup got no frames ({warm_err['exc']}); "
                            "continuing -- if this repeats, the camera is not "
                            "delivering frames (USB3 port/cable, or camera busy)")
                frames = _rs_wait(pipe, "stream")
                aligned = align.process(frames)
                color = np.asanyarray(aligned.get_color_frame().get_data())
                depth_u16 = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.uint16)
                yield cv2.cvtColor(color, cv2.COLOR_RGB2BGR), depth_u16, scale, K
        finally:
            pipe.stop()

    return gen()


AZURE_SDK_BIN = r"C:\Program Files\Azure Kinect SDK v1.4.2\sdk\windows-desktop\amd64\release\bin"


def _initialize_azure_sdk(pykinect):
    """Load k4a explicitly on Linux and retain the SDK default on Windows."""
    if os.name == "nt":
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(AZURE_SDK_BIN)
        os.environ["PATH"] = AZURE_SDK_BIN + os.pathsep + os.environ.get("PATH", "")
        pykinect.initialize_libraries()
        return

    candidates = [
        os.environ.get("K4A_LIBRARY_PATH"),
        ctypes.util.find_library("k4a"),
        "/usr/lib/x86_64-linux-gnu/libk4a.so",
        "/usr/lib/x86_64-linux-gnu/libk4a.so.1.4",
        "/usr/local/lib/libk4a.so",
    ]
    library = next((path for path in candidates if path and
                    (not os.path.isabs(path) or os.path.exists(path))), None)
    if library is None:
        raise RuntimeError(
            "Azure Kinect SDK library libk4a.so was not found. Run "
            "'sudo tools/setup_ubuntu22.sh --install' or set K4A_LIBRARY_PATH.")
    try:
        pykinect.initialize_libraries(module_k4a_path=library)
    except TypeError:
        # Older pykinect-azure releases do not expose the keyword, but still
        # resolve the SONAME through ldconfig.
        pykinect.initialize_libraries()


def _get_azure_color_control(device, k4a, command, name):
    """Read one Azure Kinect color-camera control from the device."""
    mode = k4a.k4a_color_control_mode_t()
    value = ctypes.c_int32()
    k4a.VERIFY(
        k4a.k4a_device_get_color_control(
            device.handle(),
            command,
            mode,
            value,
        ),
        f"Read Azure Kinect {name} failed!",
    )
    return int(mode.value), int(value.value)


def _set_azure_manual_color_control(device, k4a, command, value, name):
    """Validate, apply, and read back one manual color-camera control."""
    supports_auto = ctypes.c_bool()
    minimum = ctypes.c_int32()
    maximum = ctypes.c_int32()
    step = ctypes.c_int32()
    default = ctypes.c_int32()
    default_mode = k4a.k4a_color_control_mode_t()
    k4a.VERIFY(
        k4a.k4a_device_get_color_control_capabilities(
            device.handle(),
            command,
            supports_auto,
            minimum,
            maximum,
            step,
            default,
            default_mode,
        ),
        f"Get Azure Kinect {name} capabilities failed!",
    )

    requested = int(value)
    valid_step = max(1, int(step.value))
    if not minimum.value <= requested <= maximum.value:
        raise ValueError(
            f"Azure Kinect {name} must be within "
            f"[{minimum.value}, {maximum.value}] (got {requested})"
        )
    if (requested - minimum.value) % valid_step:
        raise ValueError(
            f"Azure Kinect {name} must use step {valid_step} from "
            f"{minimum.value} (got {requested})"
        )
    k4a.VERIFY(
        k4a.k4a_device_set_color_control(
            device.handle(),
            command,
            k4a.K4A_COLOR_CONTROL_MODE_MANUAL,
            requested,
        ),
        f"Set Azure Kinect {name} failed!",
    )
    actual_mode, actual_value = _get_azure_color_control(
        device,
        k4a,
        command,
        name,
    )
    if actual_mode != int(k4a.K4A_COLOR_CONTROL_MODE_MANUAL):
        raise RuntimeError(
            f"Azure Kinect {name} remained in mode {actual_mode} after "
            "requesting manual mode"
        )
    log(
        f"Azure Kinect manual {name}: requested={requested}, "
        f"actual={actual_value} "
        f"(range {minimum.value}..{maximum.value}, step {valid_step}, "
        f"default {default.value})"
    )
    return actual_value


def azure_frames(
    fps=30,
    warmup=10,
    color_exposure_usec=None,
    color_brightness=None,
):
    """Azure Kinect capture; same generator contract as live_frames():
    yields (rgb_bgr, depth_u16, dscale, K).

    color 720p BGRA32 (1280x720, same as the RealSense default so --scale 0.5
    keeps the same wire traffic), depth NFOV_UNBINNED warped into the color
    space via get_transformed_depth_image() (uint16 mm -> dscale 0.001).
    """
    try:
        from pykinect_azure import pykinect
        from pykinect_azure.k4a import _k4a
    except ImportError as exc:
        raise RuntimeError(
            "pykinect-azure is unavailable; create the kinect conda environment") from exc
    _initialize_azure_sdk(pykinect)

    n = pykinect.Device.device_get_installed_count()
    if n < 1:
        raise RuntimeError(
            "No Azure Kinect device found. Check USB 3, power, udev permissions, "
            "and whether another process owns the camera.")
    fps_values = {
        5: _k4a.K4A_FRAMES_PER_SECOND_5,
        15: _k4a.K4A_FRAMES_PER_SECOND_15,
        30: _k4a.K4A_FRAMES_PER_SECOND_30,
    }
    if fps not in fps_values:
        raise ValueError(f"Azure Kinect fps must be 5, 15, or 30 (got {fps})")
    config = pykinect.default_configuration
    config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
    config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P
    config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_UNBINNED
    config.camera_fps = fps_values[fps]
    try:
        device = pykinect.start_device(config=config)
    except Exception as exc:
        raise RuntimeError(
            "Failed to open Azure Kinect. Check USB 3 bandwidth, permissions, "
            "and that k4aviewer/k4arecorder is closed.") from exc
    actual_exposure_usec = None
    try:
        if color_exposure_usec is not None:
            actual_exposure_usec = _set_azure_manual_color_control(
                device,
                _k4a,
                _k4a.K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                color_exposure_usec,
                "exposure_usec",
            )
        if color_brightness is not None:
            _set_azure_manual_color_control(
                device,
                _k4a,
                _k4a.K4A_COLOR_CONTROL_BRIGHTNESS,
                color_brightness,
                "brightness",
            )
    except Exception:
        device.close()
        raise
    cal = device.get_calibration(_k4a.K4A_DEPTH_MODE_NFOV_UNBINNED,
                                 _k4a.K4A_COLOR_RESOLUTION_720P)
    K = np.array(cal.get_matrix(_k4a.K4A_CALIBRATION_TYPE_COLOR), np.float64)
    dscale = 0.001
    log(f"Azure Kinect started: color 1280x720@{fps}fps, depth NFOV_UNBINNED -> color")
    log(f"color K = fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")

    warm_t = None
    warm_error = []
    if warmup > 0:
        def _warm_azure():
            try:
                for _ in range(warmup):
                    device.update()
            except Exception as exc:  # surfaced by the generator thread join
                warm_error.append(exc)

        warm_t = threading.Thread(
            target=_warm_azure, daemon=True)
        warm_t.start()
        log(f"Azure warmup: discarding {warmup} frames in background (parallel with SSH)")

    def gen():
        nonlocal warm_t
        exposure_logged = False
        try:
            while True:
                if warm_t is not None:
                    warm_t.join()
                    warm_t = None
                    if warm_error:
                        raise RuntimeError(
                            "Azure Kinect stopped while collecting warmup frames") from warm_error[0]
                    log("Azure warmup done")
                try:
                    capture = device.update()
                except Exception as exc:
                    raise RuntimeError(
                        "Azure Kinect frame capture failed; check USB/power and camera ownership") from exc
                color_image = capture.get_color_image_object()
                frame_exposure_usec = (
                    int(_k4a.k4a_image_get_exposure_usec(color_image.handle()))
                    if color_image.is_valid()
                    else None
                )
                ok_c, color_bgra = color_image.to_numpy()
                del color_image
                ok_d, depth_mm = capture.get_transformed_depth_image()
                if not (ok_c and ok_d):
                    continue
                if not exposure_logged:
                    exposure_logged = True
                    log(
                        "Azure Kinect first color frame exposure metadata: "
                        f"{frame_exposure_usec} usec"
                    )
                    if (
                        actual_exposure_usec is not None
                        and abs(frame_exposure_usec - actual_exposure_usec)
                        > max(500, actual_exposure_usec * 0.1)
                    ):
                        log(
                            "WARN: color-control readback and frame exposure "
                            f"differ ({actual_exposure_usec} vs "
                            f"{frame_exposure_usec} usec)"
                        )
                rgb = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
                if depth_mm.shape != rgb.shape[:2]:
                    raise RuntimeError(
                        f"Azure aligned depth shape {depth_mm.shape} does not match color {rgb.shape[:2]}")
                yield rgb, depth_mm.astype(np.uint16), dscale, K
        finally:
            device.close()

    return gen()


class DataRecorder:
    """Record full-resolution data for offline 3D replay:
    recordings/run_<ts>/  meta.json + rgb/*.png + depth/*.npy (m) + poses/*.json.
    The mesh is copied in so replay_cloud.py works standalone."""

    def __init__(self, target, camera, mesh_src, prompt):
        import shutil
        self.dir = Path(target)
        self.mesh_src = mesh_src
        (self.dir / "rgb").mkdir(parents=True, exist_ok=True)
        (self.dir / "depth").mkdir(exist_ok=True)
        (self.dir / "poses").mkdir(exist_ok=True)
        self.meta = {"camera": camera, "prompt": prompt, "n_frames": 0}
        self.n = 0
        shutil.copyfile(mesh_src, self.dir / "mesh.obj")
        log(f"recording data -> {self.dir}")

    def set_geometry(self, K, dscale, color_hw, depth_hw, scale_sent):
        self.meta["K"] = np.asarray(K, np.float64).tolist()
        self.meta["dscale"] = dscale
        self.meta["color_hw"] = list(color_hw)
        self.meta["depth_hw"] = list(depth_hw)
        self.meta["scale_sent"] = scale_sent

    def write_frame(self, rgb_bgr, depth_m, pose4x4):
        i = self.n
        self.n += 1
        cv2.imwrite(str(self.dir / "rgb" / f"{i:06d}.png"), rgb_bgr)
        np.save(self.dir / "depth" / f"{i:06d}.npy", depth_m.astype(np.float32))
        if pose4x4 is not None:
            with open(self.dir / "poses" / f"{i:06d}.json", "w") as f:
                json.dump({"pose": np.asarray(pose4x4, np.float64).tolist()}, f)

    def close(self):
        self.meta["n_frames"] = self.n
        with open(self.dir / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)
        log(f"data recording closed -> {self.dir} ({self.n} frames)")


class Recorder:
    """Lazily-opened VideoWriter for overlay frames."""

    def __init__(self, target, nominal_fps):
        self.target = str(target)
        self.nominal_fps = nominal_fps
        self.writer = None
        self.path = None
        self.frames = 0
        self.first_ts = None
        self.last_ts = None

    def write(self, vis):
        if self.writer is None:
            h, w = vis.shape[:2]
            self.path = self.target
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.path, fourcc, self.nominal_fps, (w, h))
            if not self.writer.isOpened():
                # mp4v not available -> fall back to XVID .avi
                self.path = str(Path(self.target).with_suffix(".avi"))
                self.writer = cv2.VideoWriter(
                    self.path, cv2.VideoWriter_fourcc(*"XVID"), self.nominal_fps, (w, h))
                log(f"mp4v writer failed, fell back to XVID -> {self.path}")
            log(f"recording overlay {w}x{h}@{self.nominal_fps}fps -> {self.path}")
        now = time.time()
        if self.first_ts is None:
            self.first_ts = now
        self.last_ts = now
        self.writer.write(vis)
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            dur = 0.0
            if self.first_ts is not None and self.last_ts is not None and self.frames > 1:
                dur = self.last_ts - self.first_ts
            log(f"recording closed -> {self.path}: {self.frames} frames over {dur:.1f}s "
                f"(~{self.frames/dur:.1f} fps measured)" if dur else
                f"recording closed -> {self.path}: {self.frames} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=None, help="recorded dir; omit for live camera")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--prompt", default="red apple")
    ap.add_argument("--jpg-quality", type=int, default=85)
    ap.add_argument("--fps", type=int, default=30, help="camera capture fps (live only)")
    ap.add_argument("--camera", default="auto", choices=["auto", "realsense", "azure"],
                    help="live camera backend; auto tries RealSense then Azure Kinect")
    ap.add_argument("--cloud", action="store_true",
                    help="also open an Open3D 3D point-cloud window (blue=observed, "
                         "red=CAD@pose)")
    ap.add_argument("--crop", choices=["erode", "surface", "hull", "box", "sphere"],
                    default="erode",
                    help="blue-cloud crop: 'erode' = carve the apple region from "
                         "the depth image around the pose origin and erode its "
                         "edge --erode px inward (default; NO CAD matching, trims "
                         "the silhouette edge where background sneaks in); "
                         "'surface' = within --shrink mm of the CAD surface; "
                         "'hull' = inside the mesh convex hull; 'box' = inside the "
                         "6DoF bounding box; 'sphere' = within --radius of the pose "
                         "centre")
    ap.add_argument("--erode", type=int, default=7,
                    help="nominal pixels to erode the apple-region boundary by "
                         "(crop=erode only) at --erode-ref distance; scaled with "
                         "the apple's apparent size (closer=bigger image=more "
                         "erosion, farther=less), so the physical trim stays "
                         "proportional")
    ap.add_argument("--erode-ref", type=float, default=0.5,
                    help="distance (m) at which --erode is the exact pixel "
                         "count; erosion scales as erode*ref/z (crop=erode only)")
    ap.add_argument("--shrink", type=float, default=5.0,
                    help="trim the outer shell (mm): for crop=surface this is the "
                         "band around the CAD surface to keep; for crop=hull/box "
                         "it offsets the boundary inward by that many mm")
    ap.add_argument("--ref-cloud", action="store_true",
                    help="fixed-reference-cloud mode: accumulate the apple's "
                         "point cloud over the first --ref-frames tracked frames, "
                         "then render it YELLOW moved by the current pose "
                         "(blue/red hidden). Reference keeps points captured "
                         "while unoccluded, so later occlusion still shows the "
                         "full surface.")
    ap.add_argument("--ref-frames", type=int, default=5,
                    help="tracked frames to accumulate the reference cloud over "
                         "(--ref-cloud)")
    ap.add_argument("--voxel", type=float, default=0.005,
                    help="voxel size (m) when merging the reference cloud "
                         "(--ref-cloud)")
    ap.add_argument("--headless", action="store_true", help="no display window (CI/verify)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N sent frames (0=all)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="downscale frames by this factor before sending (K scaled "
                         "accordingly). 0.5 -> 640x360: 4x less depth/overlay traffic, "
                         "much lower latency over the SSH tunnel.")
    ap.add_argument("--warmup", type=int, default=20,
                    help="frames to discard after camera start for AWB convergence (live only)")
    ap.add_argument("--record", default=None,
                    help="record overlay video; 'auto' -> local_test/run_<ts>.mp4")
    ap.add_argument("--record-data", default=None,
                    help="record full-res RGB+D+pose for 3D replay; 'auto' -> "
                         "recordings/run_<ts>/ (works with --cloud and --camera)")
    ap.add_argument("--record-fps", type=float, default=10.0,
                    help="nominal fps written into the recorded video (playback rate; "
                         "the log reports the measured fps)")
    ap.add_argument("--log-every", type=int, default=10,
                    help="print a throughput summary every N frames")
    args = ap.parse_args()

    MESH_SRC = "apple_assets/apple_10cm_hull.obj"

    if args.replay:
        frames = replay_frames(args.replay, step=args.step)
    else:
        if args.camera == "azure":
            frames = azure_frames(fps=args.fps, warmup=args.warmup)
        elif args.camera == "realsense":
            frames = live_frames(fps=args.fps, warmup=args.warmup)
        else:  # auto: RealSense first, fall back to Azure Kinect
            try:
                import pyrealsense2  # noqa: F401  - probe availability
                frames = live_frames(fps=args.fps, warmup=args.warmup)
            except Exception as e:
                log(f"RealSense unavailable ({e}), falling back to Azure Kinect")
                frames = azure_frames(fps=args.fps, warmup=args.warmup)

    ssh, chan = open_tunnel()
    log(f"prompt='{args.prompt}' scale={args.scale} camera={args.camera} "
        f"{'headless' if args.headless else 'window'} "
        f"record={args.record or 'off'} record_data={args.record_data or 'off'} "
        f"cloud={args.cloud} warmup={args.warmup}")
    if args.record_data and not args.cloud:
        log("hint: add --cloud to also open the 3D point-cloud window while recording")

    recorder = None
    if args.record:
        if args.record == "auto":
            Path("local_test").mkdir(exist_ok=True)
            target = Path("local_test") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        else:
            target = Path(args.record)
            target.parent.mkdir(parents=True, exist_ok=True)
        recorder = Recorder(target, args.record_fps)

    data_rec = None
    if args.record_data:
        if args.record_data == "auto":
            target = Path("recordings") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            target = Path(args.record_data)
        data_rec = DataRecorder(target, args.camera, MESH_SRC, args.prompt)

    cloud_view = None
    mesh = None
    ref_cloud = None
    if args.cloud or args.ref_cloud:  # --ref-cloud implies the 3D window
        import pointcloud as pc
        mesh = pc.load_mesh(MESH_SRC)
        log(f"mesh loaded: {len(mesh.vertices)} verts")
        cloud_view = pc.CloudViewer3D()
        if args.ref_cloud:
            ref_cloud = pc.RefCloud(n_frames=args.ref_frames, voxel=args.voxel,
                                    crop=args.crop, shrink_mm=args.shrink,
                                    erode_px=args.erode,
                                    erode_ref_m=args.erode_ref)
            log(f"ref-cloud mode: accumulating {args.ref_frames} tracked frames, "
                f"then YELLOW = ref@pose (blue/red hidden, 3D window implied)")
        else:
            log("3D point-cloud window opened (blue=observed, red=CAD@pose)")

    reset_pending = True
    per_frame = []          # round-trip latency per sent frame
    stat_mark = time.time()
    t_start = time.time()

    # Per-stage timing (ms), reset every log window. "recv" covers the network
    # round-trip + server processing (SAM3/register/track) - the part we can't
    # see from the client alone. Camera capture is timed separately via an
    # explicit next() on the frame generator.
    stage_sum, stage_cnt = {}, {}
    STAGE_ORDER = ("cam", "resize", "jpeg", "send", "recv",
                   "rec_cloud", "dec_show", "ui")

    def mark(name, t0):
        dt = (time.time() - t0) * 1000.0
        stage_sum[name] = stage_sum.get(name, 0.0) + dt
        stage_cnt[name] = stage_cnt.get(name, 0) + 1

    def stage_line():
        return " ".join(f"{k}={stage_sum.get(k, 0.0) / max(1, stage_cnt.get(k, 0)):.1f}"
                        for k in STAGE_ORDER)

    # server_ms comes back in every header (server processing time for this
    # frame); net = client-side recv wait minus server_ms (network + tunnel).
    server_total, net_total = 0.0, 0.0   # full-run accumulators (ms)
    server_base, net_base = 0.0, 0.0     # last-window snapshot for deltas

    try:
        frame_iter = iter(frames)
        fi = -1
        while True:
            t_cam = time.time()
            try:
                rgb, depth_u16, dscale, K = next(frame_iter)
            except StopIteration:
                break
            fi += 1
            mark("cam", t_cam)

            # keep the full-resolution pair for data recording (pre-scale)
            pending = None
            t_stage = time.time()
            if data_rec is not None:
                if not data_rec.meta.get("K"):
                    data_rec.set_geometry(K, dscale,
                                          rgb.shape[:2], depth_u16.shape[:2],
                                          args.scale)
                pending = (rgb, depth_u16)
            if args.scale != 1.0:
                rgb = cv2.resize(rgb, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_AREA)
                depth_u16 = cv2.resize(depth_u16, None, fx=args.scale, fy=args.scale,
                                       interpolation=cv2.INTER_NEAREST)
                K = K.copy()
                K[0, 0] *= args.scale
                K[1, 1] *= args.scale
                K[0, 2] *= args.scale
                K[1, 2] *= args.scale
            mark("resize", t_stage)

            t_stage = time.time()
            ok, jpeg = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
            if not ok:
                log(f"frame {fi}: jpeg encode failed, skipped")
                continue
            mark("jpeg", t_stage)

            t0 = time.time()
            send_frame(chan, {
                "w": rgb.shape[1], "h": rgb.shape[0],
                "d_scale": dscale, "prompt": args.prompt,
                "reset": reset_pending, "K": K.tolist(),
            }, jpeg.tobytes(), depth_u16)
            mark("send", t0)

            t_stage = time.time()
            header, vis_jpeg = recv_frame(chan)
            t2 = time.time()
            per_frame.append(t2 - t0)
            mark("recv", t_stage)
            server_ms = float(header.get("server_ms") or 0.0)
            net_ms = (t2 - t_stage) * 1000.0 - server_ms
            if net_ms < 0.0:
                net_ms = 0.0
            server_total += server_ms
            net_total += net_ms

            if reset_pending:
                log(f"frame {fi}: sent reset=1 -> SAM3 + re-register")
                reset_pending = False

            status = header.get("status", "?")
            pose_arr = None
            if "pose" in header:
                pose_arr = np.array(header["pose"], np.float64)
                t = pose_arr[:3, 3]
                print(f"[{status}] frame {header.get('frame')}: "
                      f"pos=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.3f}) "
                      f"latency={1000*(t2-t0):.0f}ms", flush=True)
            else:
                print(f"[{status}] frame {header.get('frame')}: no pose", flush=True)

            # data recording + 3D point-cloud update use the FULL-resolution
            # (recorded) depth / the SENT (downscaled) depth, respectively.
            t_stage = time.time()
            if data_rec is not None and pending is not None:
                depth_m = pending[1].astype(np.float32) * dscale
                data_rec.write_frame(pending[0], depth_m, pose_arr)
            if cloud_view is not None and pose_arr is not None:
                depth_m = depth_u16.astype(np.float32) * dscale
                if ref_cloud is not None:
                    # fixed-reference mode: feed the accumulation window, then
                    # show yellow = ref @ current pose (blue/red hidden).
                    if not ref_cloud.ready:
                        ref_cloud.add(depth_m, K, pose_arr, mesh)
                        if ref_cloud.ready:
                            log(f"ref-cloud ready: {ref_cloud.count} reference points")
                    yellow = ref_cloud.at_pose(pose_arr)
                    cloud_view.update(np.zeros((0, 3)), np.zeros((0, 3)), yellow)
                else:
                    blue, red = pc.make_clouds(depth_m, K, pose_arr, mesh,
                                               crop=args.crop, shrink_mm=args.shrink,
                                               erode_px=args.erode,
                                               erode_ref_m=args.erode_ref)
                    cloud_view.update(blue, red)
            mark("rec_cloud", t_stage)

            t_stage = time.time()
            if vis_jpeg:
                vis = cv2.imdecode(np.frombuffer(vis_jpeg, np.uint8), cv2.IMREAD_COLOR)
                if vis is not None:
                    if recorder is not None:
                        recorder.write(vis)
                    if not args.headless:
                        cv2.imshow(WINDOW_TITLE, vis)
                    elif header.get("frame", 0) % 20 == 0:
                        out_png = f"local_test/live_overlay_{header.get('frame'):06d}.png"
                        cv2.imwrite(out_png, vis)
                        log(f"saved {out_png}")
            mark("dec_show", t_stage)

            if args.log_every and (fi + 1) % args.log_every == 0:
                win = per_frame[-args.log_every:]
                now = time.time()
                win_fps = args.log_every / (now - stat_mark)
                stat_mark = now
                print(f"[stats] sent={fi+1} win_fps={win_fps:.1f} "
                      f"latency_avg={np.mean(win)*1000:.0f}ms "
                      f"latency_last={win[-1]*1000:.0f}ms", flush=True)
                win_server = server_total - server_base
                win_net = net_total - net_base
                server_base, net_base = server_total, net_total
                print(f"[stage] {stage_line()} ms   "
                      f"server={win_server/args.log_every:.1f} net={win_net/args.log_every:.1f} ms "
                      f"(cam=采集 resize=降采样 jpeg=编码 send=发送 recv=网络+服务器 "
                      f"server=服务器推理 net=通信 rec_cloud=录制+点云 "
                      f"dec_show=解码+显示 ui=waitKey)", flush=True)
                stage_sum, stage_cnt = {}, {}

            if args.max_frames and fi + 1 >= args.max_frames:
                log(f"reached --max-frames {args.max_frames}")
                break

            t_ui = time.time()
            key = 0 if args.headless else cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                reset_pending = True
                log(">> requesting SAM3 re-segment + re-register")
            elif key in (ord("q"), 27):
                log("quit requested")
                break
            mark("ui", t_ui)
    except (ConnectionError, OSError) as e:
        log(f"connection error: {e}")
    finally:
        if recorder is not None:
            recorder.close()
        if data_rec is not None:
            data_rec.close()
        if cloud_view is not None:
            cloud_view.close()
        ssh.close()
        cv2.destroyAllWindows()
        el = time.time() - t_start
        if per_frame:
            log(f"done: {len(per_frame)} frames in {el:.1f}s "
                f"({len(per_frame)/el:.2f} fps), mean latency {np.mean(per_frame)*1000:.0f}ms")
            line = " ".join(f"{k}={stage_sum.get(k, 0.0)/len(per_frame):.1f}"
                            for k in STAGE_ORDER)
            log(f"[stage] per-frame avg: {line} ms  "
                f"server={server_total/len(per_frame):.1f} net={net_total/len(per_frame):.1f} ms  "
                f"(recv = server 推理 + net 通信)")
        log("client done")


if __name__ == "__main__":
    main()
