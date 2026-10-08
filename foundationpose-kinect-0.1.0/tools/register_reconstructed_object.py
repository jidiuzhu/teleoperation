#!/usr/bin/env python3
"""Atomically add a reconstructed mesh to the pipeline object configuration."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-select", action="store_true")
    args = parser.parse_args()

    if not args.name.strip() or any(char in args.name for char in "/\\\n\r\t"):
        parser.error("name must be non-empty and cannot contain path separators")
    config_path = Path(args.config).resolve()
    root = Path(args.root).resolve()
    mesh = Path(args.mesh).resolve()
    if not mesh.is_file() or mesh.stat().st_size == 0:
        parser.error(f"reconstructed mesh is missing or empty: {mesh}")
    prompts = [value.strip() for value in args.prompts.split(",") if value.strip()]
    if not prompts:
        parser.error("at least one prompt is required")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    objects = config.setdefault("objects", {})
    if args.name in objects and not args.replace:
        parser.error(f"object '{args.name}' already exists; use --replace to update it")
    try:
        mesh_value = str(mesh.relative_to(root))
    except ValueError:
        mesh_value = str(mesh)
    objects[args.name] = {"mesh": mesh_value, "prompts": prompts, "source": "model-free-rgbd"}
    if not args.no_select:
        config["selected_object"] = args.name

    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, config_path)
    print(f"Registered object '{args.name}' -> {mesh_value}")


if __name__ == "__main__":
    main()
