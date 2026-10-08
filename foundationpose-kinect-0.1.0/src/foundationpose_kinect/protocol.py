"""FoundationPose live TCP protocol implementation."""
import socket
import struct

import numpy as np

from .types import RGBDFrame


FRAME_HEADER = struct.Struct("<4sIQIII9fQQQ")
POSE_RESPONSE = struct.Struct("<4sIIQf16f")
FLAG_INITIALIZE = 1
FLAG_STOP = 2
STATUS_MESSAGES = {
    1: "FoundationPose registration failed",
    2: "FoundationPose tracking failed",
    3: "FoundationPose rejected the frame",
}


def receive_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("FoundationPose server closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def send_frame(connection: socket.socket, frame_id: int, frame: RGBDFrame,
               mask=None, initialize=False):
    frame.validated()
    color = np.ascontiguousarray(frame.color_bgr, dtype=np.uint8)
    depth = np.ascontiguousarray(frame.depth_mm, dtype=np.uint16)
    intrinsics = np.asarray(frame.intrinsics, dtype=np.float32)
    mask_bytes = b""
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != depth.shape:
            raise ValueError("mask must match the RGB-D frame dimensions")
        if initialize and not np.any(mask):
            raise ValueError("registration mask contains no target pixels")
        mask_bytes = np.ascontiguousarray(mask > 0, dtype=np.uint8).tobytes()
    if initialize and not mask_bytes:
        raise ValueError("registration requires a non-empty mask payload")
    color_bytes = color.tobytes()
    depth_bytes = depth.tobytes()
    height, width = depth.shape
    header = FRAME_HEADER.pack(
        b"FPLV", 1, frame_id, width, height, FLAG_INITIALIZE if initialize else 0,
        *intrinsics.reshape(-1), len(color_bytes), len(depth_bytes), len(mask_bytes))
    connection.sendall(header)
    connection.sendall(color_bytes)
    connection.sendall(depth_bytes)
    if mask_bytes:
        connection.sendall(mask_bytes)

    values = POSE_RESPONSE.unpack(receive_exact(connection, POSE_RESPONSE.size))
    magic, version, status, response_id, inference_ms = values[:5]
    if magic != b"FPOS" or version != 1 or response_id != frame_id:
        raise RuntimeError(
            f"invalid FoundationPose response: magic={magic!r}, version={version}, "
            f"frame={response_id}, expected={frame_id}")
    if status:
        raise RuntimeError(STATUS_MESSAGES.get(status, f"FoundationPose status {status}"))
    pose = np.asarray(values[5:], dtype=np.float32).reshape(4, 4)
    return pose, float(inference_ms)


def send_stop(connection: socket.socket, frame_id: int):
    header = FRAME_HEADER.pack(
        b"FPLV", 1, frame_id, 0, 0, FLAG_STOP, *([0.0] * 9), 0, 0, 0)
    connection.sendall(header)
