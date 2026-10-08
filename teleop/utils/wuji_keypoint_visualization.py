"""Generate an interactive HTML snapshot of RM75/Wuji keypoints."""

import base64
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from utils.wuji_fingertip_kinematics import (
    DEFAULT_ARM_URDF,
    DEFAULT_HAND_URDF,
    WujiFingertipKinematics,
)


_ARM_MESH_DIR = DEFAULT_ARM_URDF.parents[1] / "meshes"
_HAND_MESH_DIR = DEFAULT_HAND_URDF.parents[1] / "meshes" / "right"


# Add future points here. ``offset_m`` is expressed in the named link frame.
DEFAULT_KEYPOINT_SPECS = (
    {"name": "thumb_tip", "link_name": "finger1_tip_link", "offset_m": (0.0, 0.0, 0.0), "color": "#d1495b",},
    {"name": "thumb_3", "link_name": "finger1_link3", "offset_m": (0.0, 0.0, 0.015), "color": "#d1495b",},
    {"name": "thumb_2", "link_name": "finger1_link2", "offset_m": (0.0, 0.0, 0.02), "color": "#d1495b",},

    {"name": "index_tip", "link_name": "finger2_tip_link", "offset_m": (0.0, 0.0, -0.005), "color": "#00798c",},
    {"name": "index_3", "link_name": "finger2_link3", "offset_m": (0.0, 0.0, 0.015), "color": "#00798c",},
    {"name": "index_2", "link_name": "finger2_link2", "offset_m": (-0.005, 0.0, 0.02), "color": "#00798c",},

    {"name": "middle_tip", "link_name": "finger3_tip_link", "offset_m": (0.0, 0.0, -0.005), "color": "#edae49",},
    {"name": "middle_3", "link_name": "finger3_link3", "offset_m": (0.0, 0.0, 0.015), "color": "#edae49", },
    {"name": "middle_2", "link_name": "finger3_link2", "offset_m": (-0.005, 0.0, 0.02), "color": "#edae49", },

    {"name": "ring_tip", "link_name": "finger4_tip_link", "offset_m": (0.0, 0.0, -0.005), "color": "#30638e",},
    {"name": "ring_3", "link_name": "finger4_link3", "offset_m": (0.0, 0.0, 0.015), "color": "#30638e",},
    {"name": "ring_2", "link_name": "finger4_link2", "offset_m": (-0.005, 0.0, 0.02), "color": "#30638e",},

    {"name": "little_tip", "link_name": "finger5_tip_link", "offset_m": (0.0, 0.0, -0.005), "color": "#7a5195",},
    {"name": "little_3", "link_name": "finger5_link3", "offset_m": (0.0, 0.0, 0.015), "color": "#7a5195", },
    {"name": "little_2", "link_name": "finger5_link2", "offset_m": (-0.005, 0.0, 0.02), "color": "#7a5195", },
)


def _serializable_transform(transform):
    return np.asarray(transform, dtype=np.float64).tolist()


def _embedded_stl_meshes(link_names, mesh_directory):
    meshes = []
    for link_name in link_names:
        mesh_path = mesh_directory / f"{link_name}.STL"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing STL mesh for {link_name!r}: {mesh_path}")
        meshes.append(
            {
                "link_name": link_name,
                "stl_base64": base64.b64encode(mesh_path.read_bytes()).decode("ascii"),
            }
        )
    return meshes


def _validate_keypoint_specs(specs, hand_transforms):
    result = []
    names = set()
    for raw in specs:
        name = str(raw["name"])
        link_name = str(raw["link_name"])
        color = str(raw.get("color", "#d1495b"))
        offset = np.asarray(raw.get("offset_m", (0, 0, 0)), dtype=np.float64)
        if not name or name in names:
            raise ValueError(f"Keypoint names must be non-empty and unique: {name!r}")
        if link_name not in hand_transforms:
            raise ValueError(f"Unknown Wuji keypoint link: {link_name!r}")
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError(f"Keypoint {name!r} offset_m must be a finite 3-vector")
        if len(color) != 7 or not color.startswith("#"):
            raise ValueError(f"Keypoint {name!r} color must use #RRGGBB")
        try:
            int(color[1:], 16)
        except ValueError as exc:
            raise ValueError(f"Keypoint {name!r} color must use #RRGGBB") from exc
        names.add(name)
        result.append(
            {
                "name": name,
                "link_name": link_name,
                "offset_m": offset,
                "color": color,
            }
        )
    return result


def generate_wuji_keypoint_html(
    output_path,
    arm_joint_angles,
    hand_joint_angles,
    flange_thickness_m,
    *,
    arm_angle_unit="degree",
    mount_to_palm_transform=None,
    keypoint_specs=DEFAULT_KEYPOINT_SPECS,
    kinematics=None,
):
    """Write a Three.js HTML snapshot and return its absolute path."""
    if kinematics is None:
        kinematics = WujiFingertipKinematics()
    transforms = kinematics.model_transforms(
        arm_joint_angles,
        hand_joint_angles,
        flange_thickness_m,
        arm_angle_unit=arm_angle_unit,
        mount_to_palm_transform=mount_to_palm_transform,
    )
    specs = _validate_keypoint_specs(keypoint_specs, transforms["hand"])

    positions = kinematics.estimate_keypoints(
        arm_joint_angles,
        hand_joint_angles,
        flange_thickness_m,
        keypoint_specs=specs,
        arm_angle_unit=arm_angle_unit,
        mount_to_palm_transform=mount_to_palm_transform,
    )
    keypoints = []
    for spec in specs:
        keypoints.append(
            {
                "name": spec["name"],
                "link_name": spec["link_name"],
                "offset_m": spec["offset_m"].tolist(),
                "position_m": positions[spec["name"]].tolist(),
                "color": spec["color"],
            }
        )

    arm_links = {
        name: _serializable_transform(transform)
        for name, transform in transforms["arm"].items()
    }
    hand_links = {
        name: _serializable_transform(transform)
        for name, transform in transforms["hand"].items()
    }
    scene_data = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frame": "RM75 base_link",
        "handedness": "right",
        "hand_model": "wujihand-right-v1.0.1",
        "units": "meter",
        "flange_thickness_m": float(flange_thickness_m),
        "arm_joint_angles_deg": (
            np.asarray(arm_joint_angles, dtype=np.float64)
            if arm_angle_unit == "degree"
            else np.rad2deg(np.asarray(arm_joint_angles, dtype=np.float64))
        ).reshape(7).tolist(),
        "hand_joint_angles_rad": np.asarray(
            hand_joint_angles, dtype=np.float64
        ).reshape(5, 4).tolist(),
        "arm_links": arm_links,
        "hand_links": hand_links,
        "arm_edges": [list(edge) for edge in transforms["arm_edges"]],
        "hand_edges": [list(edge) for edge in transforms["hand_edges"]],
        "arm_meshes": _embedded_stl_meshes(arm_links, _ARM_MESH_DIR),
        "hand_meshes": _embedded_stl_meshes(hand_links, _HAND_MESH_DIR),
        "keypoints": keypoints,
    }
    encoded_data = json.dumps(scene_data, ensure_ascii=True, separators=(",", ":"))
    encoded_data = encoded_data.replace("</", "<\\/")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _HTML_TEMPLATE.replace("__SCENE_DATA__", encoded_data), encoding="utf-8"
    )
    return destination


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RM75 / Wuji keypoints</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #182026;
      background: #eef1f3;
    }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
    body { display: grid; grid-template-columns: minmax(0, 1fr) 360px; }
    #viewport { position: relative; min-width: 0; background: #eef1f3; }
    #viewport canvas { display: block; width: 100%; height: 100%; }
    #labels { position: absolute; inset: 0; pointer-events: none; }
    .label {
      padding: 3px 6px;
      border: 1px solid rgba(24, 32, 38, 0.2);
      border-radius: 3px;
      background: rgba(255, 255, 255, 0.92);
      color: #182026;
      font-size: 11px;
      font-weight: 650;
      line-height: 1.2;
      white-space: nowrap;
      box-shadow: 0 1px 3px rgba(24, 32, 38, 0.12);
    }
    aside {
      min-width: 0;
      overflow: auto;
      border-left: 1px solid #cbd2d6;
      background: #ffffff;
    }
    header { padding: 18px 18px 14px; border-bottom: 1px solid #d9dfe2; }
    h1 { margin: 0; font-size: 18px; line-height: 1.25; letter-spacing: 0; }
    .meta { margin-top: 8px; color: #59666d; font-size: 12px; line-height: 1.55; }
    .hand-side {
      display: inline-flex;
      margin-top: 9px;
      padding: 4px 7px;
      border-left: 3px solid #00798c;
      background: #eef6f7;
      color: #005f6b;
      font-size: 11px;
      font-weight: 750;
    }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .mode-control { display: inline-flex; }
    .mode-control .mode-button {
      position: relative;
      border-radius: 0;
    }
    .mode-control .mode-button + .mode-button { margin-left: -1px; }
    .mode-control .mode-button:first-child { border-radius: 4px 0 0 4px; }
    .mode-control .mode-button:last-child { border-radius: 0 4px 4px 0; }
    .mode-control .mode-button.active {
      z-index: 1;
      border-color: #182026;
      background: #182026;
      color: #ffffff;
    }
    button {
      min-height: 34px;
      padding: 0 11px;
      border: 1px solid #aeb8bd;
      border-radius: 4px;
      background: #ffffff;
      color: #182026;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }
    button:hover { background: #f2f5f6; border-color: #7b898f; }
    button:focus-visible { outline: 2px solid #00798c; outline-offset: 2px; }
    .section-title {
      margin: 0;
      padding: 14px 18px 8px;
      color: #59666d;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }
    #point-list { padding: 0 10px 12px; }
    .point-row {
      width: 100%;
      display: grid;
      grid-template-columns: 12px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      padding: 10px 8px;
      border: 1px solid transparent;
      border-bottom-color: #e4e8ea;
      background: transparent;
      text-align: left;
    }
    .point-row:hover, .point-row.active { background: #f3f6f7; }
    .point-row.active { border-color: #aeb8bd; }
    .swatch { width: 10px; height: 10px; margin-top: 3px; border-radius: 50%; }
    .point-name { display: block; font-size: 13px; font-weight: 700; }
    .point-link { display: block; margin-top: 2px; color: #6b777d; font-size: 11px; }
    .point-offset { display: block; margin-top: 2px; color: #7b898f; font-size: 10px; }
    .coords {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 5px;
      margin-top: 7px;
      font: 11px/1.3 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #344047;
    }
    .coord { padding: 4px; background: #eef1f3; border-radius: 3px; }
    .legend { display: flex; gap: 14px; padding: 4px 18px 16px; color: #59666d; font-size: 11px; }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .legend-line { width: 18px; height: 3px; }
    #load-error {
      display: none;
      position: absolute;
      inset: 20px;
      place-items: center;
      padding: 24px;
      border: 1px solid #d1495b;
      background: #fff;
      color: #8f2435;
      text-align: center;
      z-index: 5;
    }
    @media (max-width: 760px) {
      body { grid-template-columns: 1fr; grid-template-rows: minmax(300px, 56vh) minmax(0, 1fr); }
      aside { border-left: 0; border-top: 1px solid #cbd2d6; }
      header { padding: 12px 14px 10px; }
      .section-title { padding-left: 14px; }
    }
  </style>
  <script type="importmap">
    {"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
  </script>
</head>
<body>
  <main id="viewport" aria-label="3D robot keypoint view">
    <div id="labels"></div>
    <div id="load-error">Three.js could not be loaded. Check this computer's network access and reopen the HTML file.</div>
  </main>
  <aside>
    <header>
      <h1>RM75 / Wuji keypoints</h1>
      <div class="hand-side">RIGHT HAND / right.urdf</div>
      <div class="meta" id="meta"></div>
      <div class="toolbar">
        <div class="mode-control" role="group" aria-label="Pointer mode">
          <button id="rotate-mode" class="mode-button active" type="button" aria-pressed="true">Rotate</button>
          <button id="pan-mode" class="mode-button" type="button" aria-pressed="false">Pan</button>
        </div>
        <button id="reset-view" type="button">Reset view</button>
        <button id="palm-view" type="button">Palm view</button>
        <button id="back-view" type="button">Back view</button>
        <button id="toggle-labels" type="button">Hide labels</button>
      </div>
    </header>
    <h2 class="section-title">Estimated points</h2>
    <div id="point-list"></div>
    <div class="legend">
      <span class="legend-item"><span class="legend-line" style="background:#87a4b2"></span>RM75 mesh</span>
      <span class="legend-item"><span class="legend-line" style="background:#53656e"></span>Right Wuji mesh</span>
    </div>
  </aside>
  <script id="scene-data" type="application/json">__SCENE_DATA__</script>
  <script type="module">
    const data = JSON.parse(document.getElementById('scene-data').textContent);
    const errorBox = document.getElementById('load-error');

    function renderCanvasFallback(loadError) {
      console.warn('Using offline Canvas renderer:', loadError);
      const viewport = document.getElementById('viewport');
      const canvas = document.createElement('canvas');
      canvas.setAttribute('aria-label', 'Offline 3D robot keypoint view');
      viewport.prepend(canvas);
      const context = canvas.getContext('2d');
      const list = document.getElementById('point-list');
      let yaw = -0.75;
      let pitch = 0.55;
      let zoom = 1.0;
      let dragging = false;
      let previousX = 0;
      let previousY = 0;
      let selectedName = null;
      let interactionMode = 'rotate';
      let panX = 0;
      let panY = 0;

      const linkOrigin = (matrix) => [matrix[0][3], matrix[1][3], matrix[2][3]];
      const positions = [
        ...Object.values(data.arm_links).map(linkOrigin),
        ...Object.values(data.hand_links).map(linkOrigin),
        ...data.keypoints.map((point) => point.position_m),
      ];
      const minimum = [Infinity, Infinity, Infinity];
      const maximum = [-Infinity, -Infinity, -Infinity];
      for (const point of positions) {
        for (let axis = 0; axis < 3; axis += 1) {
          minimum[axis] = Math.min(minimum[axis], point[axis]);
          maximum[axis] = Math.max(maximum[axis], point[axis]);
        }
      }
      const center = minimum.map((value, axis) => (value + maximum[axis]) * 0.5);
      const extent = Math.max(...maximum.map((value, axis) => value - minimum[axis]), 0.25);

      function project(point) {
        const x = point[0] - center[0];
        const y = point[1] - center[1];
        const z = point[2] - center[2];
        const cy = Math.cos(yaw);
        const sy = Math.sin(yaw);
        const cp = Math.cos(pitch);
        const sp = Math.sin(pitch);
        const rotatedX = cy * x - sy * y;
        const rotatedY = sy * x + cy * y;
        const rotatedZ = cp * z - sp * rotatedY;
        const depth = sp * z + cp * rotatedY;
        const scale = Math.min(canvas.clientWidth, canvas.clientHeight) / extent * 0.68 * zoom;
        return {
          x: canvas.clientWidth * 0.5 + panX + rotatedX * scale,
          y: canvas.clientHeight * 0.5 + panY - rotatedZ * scale,
          depth,
        };
      }

      function drawLine(a, b, color, width, dashed = false) {
        const start = project(a);
        const end = project(b);
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.strokeStyle = color;
        context.lineWidth = width;
        context.setLineDash(dashed ? [7, 5] : []);
        context.stroke();
        context.setLineDash([]);
      }

      function transformPoint(matrix, point) {
        return [
          matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3],
          matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3],
          matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + matrix[2][3],
        ];
      }

      function decodePreviewTriangles(meshRecords, links, fill, stroke) {
        const triangles = [];
        for (const record of meshRecords) {
          const binary = atob(record.stl_base64);
          const bytes = new Uint8Array(binary.length);
          for (let index = 0; index < binary.length; index += 1) {
            bytes[index] = binary.charCodeAt(index);
          }
          const view = new DataView(bytes.buffer);
          const triangleCount = view.getUint32(80, true);
          const stride = Math.max(1, Math.ceil(triangleCount / 350));
          const matrix = links[record.link_name];
          for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex += stride) {
            const base = 84 + triangleIndex * 50 + 12;
            const vertices = [];
            for (let vertexIndex = 0; vertexIndex < 3; vertexIndex += 1) {
              const offset = base + vertexIndex * 12;
              vertices.push(transformPoint(matrix, [
                view.getFloat32(offset, true),
                view.getFloat32(offset + 4, true),
                view.getFloat32(offset + 8, true),
              ]));
            }
            triangles.push({ vertices, fill, stroke });
          }
        }
        return triangles;
      }

      const previewTriangles = [
        ...decodePreviewTriangles(data.arm_meshes, data.arm_links, 'rgba(135, 164, 178, 0.23)', 'rgba(78, 110, 129, 0.14)'),
        ...decodePreviewTriangles(data.hand_meshes, data.hand_links, 'rgba(83, 101, 110, 0.31)', 'rgba(54, 69, 79, 0.17)'),
      ];

      function drawMeshes() {
        const projected = previewTriangles.map((triangle) => {
          const vertices = triangle.vertices.map(project);
          return {
            ...triangle,
            projected: vertices,
            depth: vertices.reduce((sum, vertex) => sum + vertex.depth, 0) / 3,
          };
        }).sort((a, b) => a.depth - b.depth);
        for (const triangle of projected) {
          const [first, second, third] = triangle.projected;
          context.beginPath();
          context.moveTo(first.x, first.y);
          context.lineTo(second.x, second.y);
          context.lineTo(third.x, third.y);
          context.closePath();
          context.fillStyle = triangle.fill;
          context.fill();
          context.strokeStyle = triangle.stroke;
          context.lineWidth = 0.5;
          context.stroke();
        }
      }

      function render() {
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.max(viewport.clientWidth, 1);
        const height = Math.max(viewport.clientHeight, 1);
        if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
          canvas.width = Math.round(width * ratio);
          canvas.height = Math.round(height * ratio);
          canvas.style.width = `${width}px`;
          canvas.style.height = `${height}px`;
        }
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);
        context.fillStyle = '#eef1f3';
        context.fillRect(0, 0, width, height);

        const gridRadius = Math.max(extent, 0.7);
        for (let index = -7; index <= 7; index += 1) {
          const value = index * gridRadius / 7;
          drawLine([-gridRadius, value, 0], [gridRadius, value, 0], '#d9dfe2', 1);
          drawLine([value, -gridRadius, 0], [value, gridRadius, 0], '#d9dfe2', 1);
        }
        drawLine([0, 0, 0], [0.18, 0, 0], '#c33c54', 2.5);
        drawLine([0, 0, 0], [0, 0.18, 0], '#2f855a', 2.5);
        drawLine([0, 0, 0], [0, 0, 0.18], '#30638e', 2.5);
        drawMeshes();

        const orderedPoints = [...data.keypoints].sort(
          (a, b) => project(a.position_m).depth - project(b.position_m).depth
        );
        context.font = '600 12px system-ui, sans-serif';
        for (const point of orderedPoints) {
          const screen = project(point.position_m);
          const radius = point.name === selectedName ? 10 : 7;
          context.beginPath();
          context.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
          context.fillStyle = point.color;
          context.fill();
          context.lineWidth = 2;
          context.strokeStyle = '#ffffff';
          context.stroke();
          context.fillStyle = '#182026';
          context.fillText(point.name, screen.x + 11, screen.y - 10);
        }
      }

      function selectPoint(name, focus = false) {
        selectedName = name;
        for (const row of list.querySelectorAll('.point-row')) {
          row.classList.toggle('active', row.dataset.name === name);
        }
        if (focus) {
          const point = data.keypoints.find((candidate) => candidate.name === name);
          if (point) {
            const screen = project(point.position_m);
            panX += canvas.clientWidth * 0.5 - screen.x;
            panY += canvas.clientHeight * 0.5 - screen.y;
          }
        }
        render();
      }

      function setInteractionMode(mode) {
        interactionMode = mode;
        for (const button of document.querySelectorAll('.mode-button')) {
          const active = button.id === mode + '-mode';
          button.classList.toggle('active', active);
          button.setAttribute('aria-pressed', String(active));
        }
        canvas.style.cursor = mode === 'pan' ? 'move' : 'grab';
      }

      if (!list.children.length) {
        for (const point of data.keypoints) {
          const row = document.createElement('button');
          row.type = 'button';
          row.className = 'point-row';
          row.dataset.name = point.name;
          const millimeters = point.position_m.map((value) => value * 1000);
          const offsetMillimeters = point.offset_m.map((value) => value * 1000);
          row.innerHTML = `
            <span class="swatch" style="background:${point.color}"></span>
            <span>
              <span class="point-name">${point.name}</span>
              <span class="point-link">${point.link_name}</span>
              <span class="point-offset">offset [${offsetMillimeters.map((value) => value.toFixed(2)).join(', ')}] mm</span>
              <span class="coords">
                <span class="coord">X ${millimeters[0].toFixed(2)}</span>
                <span class="coord">Y ${millimeters[1].toFixed(2)}</span>
                <span class="coord">Z ${millimeters[2].toFixed(2)}</span>
              </span>
            </span>`;
          row.addEventListener('click', () => selectPoint(point.name, true));
          list.appendChild(row);
        }
      }
      document.getElementById('meta').innerHTML =
        `${data.frame}<br>${data.hand_model} / ${data.handedness} hand<br>${data.generated_at}<br>Flange offset ${Number(data.flange_thickness_m).toFixed(6)} m`;
      document.getElementById('toggle-labels').style.display = 'none';
      document.getElementById('reset-view').addEventListener('click', () => {
        yaw = -0.75;
        pitch = 0.55;
        zoom = 1.0;
        panX = 0;
        panY = 0;
        render();
      });
      const palmMatrix = data.hand_links.palm_link;
      const palmNormalYaw = Math.atan2(palmMatrix[1][0], palmMatrix[0][0]);
      yaw = -palmNormalYaw;
      pitch = 0.0;
      zoom = 1.0;
      document.getElementById('palm-view').addEventListener('click', () => {
        yaw = -palmNormalYaw;
        pitch = 0.0;
        zoom = 1.0;
        panX = 0;
        panY = 0;
        render();
      });
      document.getElementById('back-view').addEventListener('click', () => {
        yaw = Math.PI - palmNormalYaw;
        pitch = 0.0;
        zoom = 1.0;
        panX = 0;
        panY = 0;
        render();
      });
      document.getElementById('rotate-mode').addEventListener('click', () => setInteractionMode('rotate'));
      document.getElementById('pan-mode').addEventListener('click', () => setInteractionMode('pan'));
      canvas.addEventListener('pointerdown', (event) => {
        dragging = true;
        previousX = event.clientX;
        previousY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener('pointermove', (event) => {
        if (!dragging) return;
        const deltaX = event.clientX - previousX;
        const deltaY = event.clientY - previousY;
        if (interactionMode === 'pan') {
          panX += deltaX;
          panY += deltaY;
        } else {
          yaw += deltaX * 0.008;
          pitch = Math.max(-1.45, Math.min(1.45, pitch + deltaY * 0.008));
        }
        previousX = event.clientX;
        previousY = event.clientY;
        render();
      });
      canvas.addEventListener('pointerup', () => { dragging = false; });
      canvas.addEventListener('pointercancel', () => { dragging = false; });
      canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        zoom = Math.max(0.35, Math.min(4.0, zoom * Math.exp(-event.deltaY * 0.001)));
        render();
      }, { passive: false });
      window.addEventListener('resize', render);
      setInteractionMode('rotate');
      render();
      window.__wujiVisualizationReady = true;
      window.__wujiFallbackRenderer = true;
    }

    try {
      const modules = await Promise.race([
        Promise.all([
          import('three'),
          import('three/addons/controls/OrbitControls.js'),
          import('three/addons/renderers/CSS2DRenderer.js'),
          import('three/addons/loaders/STLLoader.js'),
        ]),
        new Promise((_, reject) => setTimeout(
          () => reject(new Error('Three.js load timeout; switching to offline renderer')),
          1500,
        )),
      ]);
      const [THREE, controlsModule, labelModule, stlModule] = modules;
      const { OrbitControls } = controlsModule;
      const { CSS2DRenderer, CSS2DObject } = labelModule;
      const { STLLoader } = stlModule;

      THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
      const viewport = document.getElementById('viewport');
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xeef1f3);

      const camera = new THREE.PerspectiveCamera(42, 1, 0.001, 20);
      camera.up.set(0, 0, 1);
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      viewport.prepend(renderer.domElement);

      const labelRenderer = new CSS2DRenderer();
      labelRenderer.domElement.id = 'labels';
      viewport.appendChild(labelRenderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.screenSpacePanning = true;

      function setInteractionMode(mode) {
        const panMode = mode === 'pan';
        controls.mouseButtons.LEFT = panMode ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
        controls.mouseButtons.RIGHT = panMode ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
        controls.touches.ONE = panMode ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE;
        controls.touches.TWO = THREE.TOUCH.DOLLY_PAN;
        for (const button of document.querySelectorAll('.mode-button')) {
          const active = button.id === mode + '-mode';
          button.classList.toggle('active', active);
          button.setAttribute('aria-pressed', String(active));
        }
        renderer.domElement.style.cursor = panMode ? 'move' : 'grab';
      }

      const ambient = new THREE.HemisphereLight(0xffffff, 0x7b898f, 2.2);
      scene.add(ambient);
      const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
      keyLight.position.set(1.5, -1.2, 2.5);
      scene.add(keyLight);

      const root = new THREE.Group();
      scene.add(root);
      const axes = new THREE.AxesHelper(0.18);
      root.add(axes);
      const grid = new THREE.GridHelper(1.4, 28, 0xaeb8bd, 0xd9dfe2);
      grid.rotation.x = Math.PI / 2;
      root.add(grid);

      const vector = (values) => new THREE.Vector3(values[0], values[1], values[2]);
      const origin = (matrix) => vector([matrix[0][3], matrix[1][3], matrix[2][3]]);
      const allPositions = [
        ...Object.values(data.arm_links).map(origin),
        ...Object.values(data.hand_links).map(origin),
      ];

      function decodeBase64(value) {
        const binary = atob(value);
        const bytes = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index += 1) {
          bytes[index] = binary.charCodeAt(index);
        }
        return bytes.buffer;
      }

      function threeMatrix(rows) {
        return new THREE.Matrix4().set(
          rows[0][0], rows[0][1], rows[0][2], rows[0][3],
          rows[1][0], rows[1][1], rows[1][2], rows[1][3],
          rows[2][0], rows[2][1], rows[2][2], rows[2][3],
          rows[3][0], rows[3][1], rows[3][2], rows[3][3],
        );
      }

      const stlLoader = new STLLoader();
      function addMeshes(records, links, material) {
        for (const record of records) {
          const geometry = stlLoader.parse(decodeBase64(record.stl_base64));
          geometry.computeVertexNormals();
          const mesh = new THREE.Mesh(geometry, material);
          mesh.matrixAutoUpdate = false;
          mesh.matrix.copy(threeMatrix(links[record.link_name]));
          mesh.renderOrder = 1;
          root.add(mesh);
        }
      }

      const armMaterial = new THREE.MeshStandardMaterial({
        color: 0x87a4b2,
        transparent: true,
        opacity: 0.24,
        roughness: 0.7,
        metalness: 0.05,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const handMaterial = new THREE.MeshStandardMaterial({
        color: 0x53656e,
        transparent: true,
        opacity: 0.30,
        roughness: 0.65,
        metalness: 0.04,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      addMeshes(data.arm_meshes, data.arm_links, armMaterial);
      addMeshes(data.hand_meshes, data.hand_links, handMaterial);

      const pointMeshes = new Map();
      const list = document.getElementById('point-list');
      let labelsVisible = true;

      function selectPoint(name) {
        for (const row of list.querySelectorAll('.point-row')) {
          row.classList.toggle('active', row.dataset.name === name);
        }
        for (const [pointName, mesh] of pointMeshes) {
          mesh.scale.setScalar(pointName === name ? 1.45 : 1.0);
        }
      }

      function focusPoint(position) {
        const cameraOffset = camera.position.clone().sub(controls.target);
        controls.target.copy(position);
        camera.position.copy(position).add(cameraOffset);
        controls.update();
      }

      for (const point of data.keypoints) {
        const position = vector(point.position_m);
        allPositions.push(position);
        const geometry = new THREE.SphereGeometry(0.011, 24, 16);
        const material = new THREE.MeshStandardMaterial({
          color: point.color,
          roughness: 0.45,
          metalness: 0.05,
          depthTest: false,
          depthWrite: false,
        });
        const marker = new THREE.Mesh(geometry, material);
        marker.position.copy(position);
        marker.userData.keypoint = point;
        marker.renderOrder = 10;
        root.add(marker);
        pointMeshes.set(point.name, marker);

        const labelElement = document.createElement('div');
        labelElement.className = 'label';
        labelElement.textContent = point.name;
        const label = new CSS2DObject(labelElement);
        label.position.set(0, 0, 0.018);
        marker.add(label);

        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'point-row';
        row.dataset.name = point.name;
        const millimeters = point.position_m.map((value) => value * 1000);
        const offsetMillimeters = point.offset_m.map((value) => value * 1000);
        row.innerHTML = `
          <span class="swatch" style="background:${point.color}"></span>
          <span>
            <span class="point-name">${point.name}</span>
            <span class="point-link">${point.link_name}</span>
            <span class="point-offset">offset [${offsetMillimeters.map((value) => value.toFixed(2)).join(', ')}] mm</span>
            <span class="coords">
              <span class="coord">X ${millimeters[0].toFixed(2)}</span>
              <span class="coord">Y ${millimeters[1].toFixed(2)}</span>
              <span class="coord">Z ${millimeters[2].toFixed(2)}</span>
            </span>
          </span>`;
        row.addEventListener('click', () => {
          selectPoint(point.name);
          focusPoint(position);
        });
        list.appendChild(row);
      }

      document.getElementById('meta').innerHTML =
        `${data.frame}<br>${data.hand_model} / ${data.handedness} hand<br>${data.generated_at}<br>Flange offset ${Number(data.flange_thickness_m).toFixed(6)} m`;

      const bounds = new THREE.Box3().setFromPoints(allPositions);
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      const radius = Math.max(size.length() * 0.65, 0.3);
      const initialCamera = center.clone().add(new THREE.Vector3(radius, -radius * 1.2, radius * 0.8));
      const handPositions = [
        ...Object.values(data.hand_links).map(origin),
        ...data.keypoints.map((point) => vector(point.position_m)),
      ];
      const handBounds = new THREE.Box3().setFromPoints(handPositions);
      const handCenter = handBounds.getCenter(new THREE.Vector3());
      const handRadius = Math.max(handBounds.getSize(new THREE.Vector3()).length() * 0.9, 0.18);
      const palmFrame = threeMatrix(data.hand_links.palm_link);
      const palmNormal = new THREE.Vector3(1, 0, 0).transformDirection(palmFrame);
      const palmUp = new THREE.Vector3(0, 0, 1).transformDirection(palmFrame);

      function resetView() {
        camera.up.set(0, 0, 1);
        controls.target.copy(center);
        camera.position.copy(initialCamera);
        camera.near = Math.max(radius / 1000, 0.001);
        camera.far = Math.max(radius * 20, 5);
        camera.updateProjectionMatrix();
        controls.update();
      }
      function setHandView(side) {
        camera.up.copy(palmUp);
        controls.target.copy(handCenter);
        camera.position.copy(handCenter).addScaledVector(palmNormal, side * handRadius * 2.8);
        camera.near = Math.max(handRadius / 1000, 0.001);
        camera.far = Math.max(handRadius * 30, 5);
        camera.updateProjectionMatrix();
        controls.update();
      }
      setHandView(1);
      document.getElementById('reset-view').addEventListener('click', resetView);
      document.getElementById('palm-view').addEventListener('click', () => setHandView(1));
      document.getElementById('back-view').addEventListener('click', () => setHandView(-1));
      document.getElementById('rotate-mode').addEventListener('click', () => setInteractionMode('rotate'));
      document.getElementById('pan-mode').addEventListener('click', () => setInteractionMode('pan'));
      document.getElementById('toggle-labels').addEventListener('click', (event) => {
        labelsVisible = !labelsVisible;
        labelRenderer.domElement.style.display = labelsVisible ? 'block' : 'none';
        event.currentTarget.textContent = labelsVisible ? 'Hide labels' : 'Show labels';
      });

      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      const pointerDown = new THREE.Vector2();
      renderer.domElement.addEventListener('pointerdown', (event) => {
        pointerDown.set(event.clientX, event.clientY);
      });
      renderer.domElement.addEventListener('pointerup', (event) => {
        if (pointerDown.distanceTo(new THREE.Vector2(event.clientX, event.clientY)) > 4) return;
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObjects([...pointMeshes.values()], false)[0];
        if (hit) {
          selectPoint(hit.object.userData.keypoint.name);
          focusPoint(hit.object.position);
        }
      });

      function resize() {
        const width = viewport.clientWidth;
        const height = viewport.clientHeight;
        camera.aspect = width / Math.max(height, 1);
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
        labelRenderer.setSize(width, height);
      }
      window.addEventListener('resize', resize);
      resize();
      setInteractionMode('rotate');

      function animate() {
        controls.update();
        renderer.render(scene, camera);
        labelRenderer.render(scene, camera);
        requestAnimationFrame(animate);
      }
      animate();
      window.__wujiVisualizationReady = true;
    } catch (error) {
      renderCanvasFallback(error);
    }
  </script>
</body>
</html>
'''


__all__ = ["DEFAULT_KEYPOINT_SPECS", "generate_wuji_keypoint_html"]
