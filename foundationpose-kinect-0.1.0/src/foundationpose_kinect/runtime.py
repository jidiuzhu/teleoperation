"""Managed subprocess wrapper around the existing end-to-end launcher."""
import collections
import os
import signal
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .artifacts import RunArtifacts
from .config import ProjectConfig


class FoundationPoseService:
    """Own one Dockerized C++ service for direct ``FoundationPoseSession`` use."""

    def __init__(self, project_root, *, config=None, object_name=None, port=None,
                 image=None, startup_timeout=300.0, build_image=True):
        self.project_root = Path(project_root).resolve()
        self.cpp_root = self.project_root / "foundationpose_cpp"
        config_path = (Path(config).resolve() if config else
                       self.project_root / "configs/kinect_foundationpose.json")
        self.config = ProjectConfig(config_path, self.project_root)
        self.object_spec = self.config.object(object_name)
        self.port = int(port or self.config.port)
        self.image = image or self.config.data.get("runtime", {}).get(
            "docker_image", "foundationpose-cpp:trt10-ubuntu22.04")
        self.startup_timeout = float(startup_timeout)
        self.build_image = bool(build_image)
        self.container_name = f"foundationpose-module-{uuid.uuid4().hex[:10]}"
        self._running = False

    @property
    def running(self):
        if not self._running:
            return False
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
            capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def start(self):
        if self.running:
            return self
        if shutil.which("docker") is None:
            raise RuntimeError("docker is not installed")
        if subprocess.run(["docker", "info"], capture_output=True).returncode:
            raise RuntimeError("docker daemon is unavailable to the current user")
        image_exists = subprocess.run(
            ["docker", "image", "inspect", self.image], capture_output=True).returncode == 0
        if not image_exists:
            if not self.build_image:
                raise RuntimeError(f"Docker image is missing: {self.image}")
            environment = dict(os.environ, FP_IMAGE=self.image)
            subprocess.run(
                [str(self.cpp_root / "tools/build_ubuntu_image.sh")],
                cwd=self.project_root, env=environment, check=True)

        (self.cpp_root / "test_data/live_object").mkdir(parents=True, exist_ok=True)
        command = [
            "docker", "run", "-d", "--name", self.container_name, "--gpus", "all",
            "-p", f"127.0.0.1:{self.port}:{self.port}",
            "-v", f"{self.cpp_root}:/workspace",
            "-v", f"{self.object_spec.mesh_path}:/workspace/test_data/live_object/mesh.obj:ro",
            "-w", "/workspace", self.image,
            "bash", "/workspace/tools/run_live_fp.sh", str(self.port),
            self.object_spec.name, "", "0",
        ]
        subprocess.run(command, cwd=self.project_root, check=True, capture_output=True)
        self._running = True
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "80", self.container_name],
                capture_output=True, text=True)
            output = logs.stdout + logs.stderr
            if f"READY port={self.port}" in output:
                return self
            if not self.running:
                self.stop()
                raise RuntimeError(f"FoundationPose container exited during startup:\n{output}")
            time.sleep(0.5)
        logs = self.logs()
        self.stop()
        raise TimeoutError(f"FoundationPose service did not become ready:\n{logs}")

    def logs(self):
        result = subprocess.run(
            ["docker", "logs", self.container_name], capture_output=True, text=True)
        return result.stdout + result.stderr

    def stop(self):
        if self._running:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name], capture_output=True)
        self._running = False

    def __enter__(self):
        return self.start()

    def __exit__(self, _type, _value, _traceback):
        self.stop()


class PipelineProcess:
    """Start/stop the Kinect + SAM3 + Docker pipeline without a shell wrapper."""

    def __init__(self, project_root, *, config=None, output_limit=2000):
        self.project_root = Path(project_root).resolve()
        self.launcher = self.project_root / "run_kinect_foundationpose.sh"
        if not self.launcher.is_file():
            raise FileNotFoundError(f"launcher does not exist: {self.launcher}")
        self.config = (Path(config).resolve() if config else
                       self.project_root / "configs/kinect_foundationpose.json")
        self.output_root = self.project_root / "foundationpose_cpp/test_output/live"
        self._lines = collections.deque(maxlen=output_limit)
        self._process = None
        self._reader = None
        self._known_runs = set()

    @property
    def running(self):
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    @property
    def returncode(self):
        return self._process.poll() if self._process is not None else None

    def start(self, *launcher_args):
        if self.running:
            raise RuntimeError("pipeline is already running")
        self._known_runs = set(self.output_root.glob("run_*"))
        command = [str(self.launcher), "--config", str(self.config), *map(str, launcher_args)]
        self._process = subprocess.Popen(
            command, cwd=self.project_root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        return self

    def _read_output(self):
        for line in self._process.stdout:
            self._lines.append(line.rstrip("\n"))

    def output(self):
        return tuple(self._lines)

    def wait(self, timeout=None):
        status = self._process.wait(timeout=timeout)
        if self._reader is not None:
            self._reader.join(timeout=2)
        return status

    def stop(self, timeout=15):
        if not self.running:
            return
        os.killpg(self._process.pid, signal.SIGINT)
        try:
            self.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self._process.pid, signal.SIGTERM)
            self.wait(5)

    def artifacts(self):
        new_runs = sorted(set(self.output_root.glob("run_*")) - self._known_runs)
        if new_runs:
            return RunArtifacts(new_runs[-1])
        return RunArtifacts.latest(self.output_root)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.stop()
