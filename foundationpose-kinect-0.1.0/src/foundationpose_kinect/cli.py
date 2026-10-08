"""Small CLI for package verification and managed execution."""
import argparse
import json
import time
from pathlib import Path

from .artifacts import RunArtifacts
from .config import ProjectConfig
from .runtime import PipelineProcess


def main():
    parser = argparse.ArgumentParser(prog="fp-kinect")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-config")
    inspect_parser.add_argument("--config", default="configs/kinect_foundationpose.json")
    inspect_parser.add_argument("--root", default=".")
    inspect_parser.add_argument("--object")
    artifacts_parser = subparsers.add_parser("artifacts")
    artifacts_parser.add_argument("--run", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", default=".")
    run_parser.add_argument("--config")
    run_parser.add_argument("launcher_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command == "inspect-config":
        root = Path(args.root).resolve()
        config = ProjectConfig(root / args.config, root)
        obj = config.object(args.object)
        print(json.dumps({"object": obj.name, "mesh": str(obj.mesh_path),
                          "prompts": obj.prompts, "port": config.port}, indent=2))
    elif args.command == "artifacts":
        artifacts = RunArtifacts(args.run)
        pose = artifacts.latest_pose()
        print(json.dumps({"run": str(artifacts.path),
                          "translation_m": pose[:3, 3].tolist(),
                          "metrics": artifacts.metrics()}, indent=2))
    else:
        process = PipelineProcess(args.root, config=args.config).start(*args.launcher_args)
        shown = 0
        try:
            while process.running:
                lines = process.output()
                for line in lines[shown:]:
                    print(line, flush=True)
                shown = len(lines)
                time.sleep(0.1)
        except KeyboardInterrupt:
            process.stop()
        raise SystemExit(process.wait() if process.running else process.returncode)


if __name__ == "__main__":
    main()
