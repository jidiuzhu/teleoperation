"""Network Tracker input with round-trip freshness (no shared clock required)."""
import json
import secrets
import socket
import time
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from utils.vive_tracker import make_transform, matrix34_to_transform, average_transforms


class NetworkWristTracker:
    def __init__(self, serial, *, bind='0.0.0.0', port=6210, source_ip=None,
                 timeout_s=0.25, tracker_to_wrist_xyz_m=(0,0,0),
                 tracker_to_wrist_rpy_deg=(0,0,0)):
        if source_ip is None:
            raise ValueError('UDP input requires --tracker-source-ip (Windows IPv4)')
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Invalid tracker timeout')
        self.serial, self.source_ip, self.timeout = serial, source_ip, timeout_s
        self.tracker_to_wrist = make_transform(tracker_to_wrist_xyz_m,tracker_to_wrist_rpy_deg)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind, port)); self.sock.setblocking(False)
        self.probes = {}
        self.peer = self.session = None
        self.retired = set()
        self.seq = -1
        self.last_probe = 0
        self.last_valid_time = None
        self.valid_until = 0
        self.pose = None
        self.ever_connected = self.ever_valid = False
        self.initial_world_wrist = None
        self.closed = False

    def _probe(self, now):
        self.probes = {k:v for k,v in self.probes.items() if now-v < self.timeout}
        if self.peer and now-self.last_probe >= self.timeout/3:
            nonce = secrets.token_hex(12)
            self.probes[nonce] = now
            self.sock.sendto(json.dumps({'type':'tracker_probe','nonce':nonce}).encode(), self.peer)
            self.last_probe = now

    def read_world_wrist(self):
        now = time.monotonic()
        interrupted = self.pose is not None and now >= self.valid_until
        self._probe(now)
        # Bounded drain avoids being trapped by a flooded socket.
        for _ in range(256):
            try:
                data, addr = self.sock.recvfrom(8192)
            except BlockingIOError:
                break
            if addr[0] != self.source_ip:
                continue
            try:
                p = json.loads(data)
                if p.get('version') != 2 or p.get('serial') != self.serial or p.get('frame') != 'steamvr_standing':
                    continue
                session = p.get('session')
                if not isinstance(session,str) or not session or len(session)>100:
                    continue
                if self.session is not None and (addr != self.peer or session != self.session):
                    if now < self.valid_until or session in self.retired:
                        continue
                    self.retired.add(self.session)
                    self.probes.clear(); self.pose = None; self.seq = -1
                    interrupted = True
                if session != self.session or addr != self.peer:
                    self.session, self.peer = session, addr
                    self.last_probe = 0
                    self._probe(now)
                self.ever_connected = True
                seq = p.get('seq')
                if type(seq) is not int or seq <= self.seq:
                    continue
                stamp = self.probes.get(p.get('echo'))
                if stamp is None or now-stamp >= self.timeout:
                    continue
                self.seq = seq
                if p.get('connected') is not True or p.get('valid') is not True or p.get('tracking_result') != 200:
                    self.pose = None; self.valid_until = 0; interrupted = True
                    continue
                transform = matrix34_to_transform(p['matrix34'])
                self.pose = transform @ self.tracker_to_wrist
                # Deadline derives from a locally-issued challenge, not receipt time.
                self.valid_until = stamp + self.timeout
                self.last_valid_time = now
                self.ever_valid = True
            except (ValueError, TypeError, KeyError, AttributeError, IndexError):
                self.pose = None; self.valid_until = 0; interrupted = True
        if interrupted or self.pose is None or now >= self.valid_until:
            return None
        return self.pose.copy()

    def seconds_since_valid(self, now=None):
        now = time.monotonic() if now is None else now
        return float('inf') if now >= self.valid_until else now-self.last_valid_time

    def wait_and_zero(self, required_frames=20, timeout_seconds=15,
                      max_position_spread_m=0.02, max_rotation_spread_deg=5):
        samples = deque(maxlen=required_frames)
        previous = -1
        deadline = time.monotonic()+timeout_seconds
        while time.monotonic() < deadline:
            pose = self.read_world_wrist()
            if pose is None:
                samples.clear()
            elif self.seq != previous:
                previous = self.seq; samples.append(pose)
                if len(samples) == required_frames:
                    mean = average_transforms(list(samples))
                    delta = max(np.linalg.norm(p[:3,3]-mean[:3,3]) for p in samples)
                    angle = max((Rotation.from_matrix(mean[:3,:3]).inv()*Rotation.from_matrix(p[:3,:3])).magnitude() for p in samples)
                    if delta <= max_position_spread_m and angle <= np.deg2rad(max_rotation_spread_deg):
                        self.initial_world_wrist = mean
                        return
                    samples.clear()
            time.sleep(0.005)
        raise TimeoutError('No stable UDP tracker frames; use updated version-2 Windows sender and check UDP both ways')

    def close(self):
        if not self.closed:
            self.closed = True; self.sock.close()
