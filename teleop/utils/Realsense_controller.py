import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import threading

class MultiCameraReader:
    def __init__(self):
        # Define device serial numbers
        self.device_config = {

            "D405": "218622273575" # Serial number needs to be modified.
        }

        self.pipelines = {}
        self.aligners = {}
        self.intrinsics = {}  # Store intrinsics for each camera

        ctx = rs.context()

        for name, serial in self.device_config.items():
            print(f"[Init] Configuring {name} ({serial})...")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)

            # Configure resolution and frame rate
            if name == "D405":
                # D405 often supports higher resolution at close range
                config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            else:
                # Standard D435i configuration
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            try:
                # Start the pipeline and get the active profile
                profile = pipeline.start(config)
                self.pipelines[name] = pipeline

                # Create align object (align depth to color)
                self.aligners[name] = rs.align(rs.stream.color)

                # --- NEW: Retrieve and store Intrinsics ---
                # We need the intrinsics of the COLOR stream because we align depth to color.
                color_stream = profile.get_stream(rs.stream.color)
                intr = color_stream.as_video_stream_profile().get_intrinsics()

                self.intrinsics[name] = {
                    "width": intr.width,
                    "height": intr.height,
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "ppx": intr.ppx,
                    "ppy": intr.ppy,
                    "coeffs": intr.coeffs
                }

                print(
                    f"[Success] {name} started. Intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, ppx={intr.ppx:.2f}, ppy={intr.ppy:.2f}")

            except Exception as e:
                print(f"[Error] Failed to start {name}: {e}")

    def get_frames(self):
        """
        Capture aligned frames from all connected cameras.
        Returns:
            dict: { "D435i": {"color": ..., "depth": ..., "timestamp": ...}, ... }
        """
        frames_data = {}

        for name, pipeline in self.pipelines.items():
            try:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames(timeout_ms=5000)

                # Align the depth frame to color frame
                aligned_frames = self.aligners[name].process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                frames_data[name] = {
                    "color": color_image,
                    "depth": depth_image,
                    "timestamp": frames.get_timestamp()  # Hardware timestamp
                }
            except RuntimeError as e:
                print(f"[Warn] Frame drop or timeout on {name}: {e}")

        return frames_data

    def get_intrinsics(self):
        """
        Return the dictionary of intrinsics for all cameras.
        Useful for saving metadata or point cloud projection.
        """
        return self.intrinsics

    def stop(self):
        print("[System] Stopping all pipelines...")
        for name, pipeline in self.pipelines.items():
            pipeline.stop()


class AsyncCameraReader:
    def __init__(self, camera_reader):
        self.reader = camera_reader
        self.latest_frames = None
        self.running = True
        self.lock = threading.Lock()

        print("[AsyncCam] Pre-fetching first frame...")
        self.latest_frames = self.reader.get_frames()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("[AsyncCam] Background thread started.")

    def _worker(self):
        while self.running:
            try:
                frames = self.reader.get_frames()
                if frames is not None:
                    with self.lock:
                        self.latest_frames = frames
            except Exception as e:
                print(f"[AsyncCam Error] {e}")
                time.sleep(0.1)

    def get_frames(self):
        with self.lock:
            return self.latest_frames

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.reader.stop()


if __name__ == "__main__":
    # Test logic
    recorder = MultiCameraReader()

    # Create directory for saving test snapshots
    save_dir = "data_records"
    os.makedirs(save_dir, exist_ok=True)

    print("\n>>> Press 'q' to stop recording, 's' to save a snapshot.")

    # Print intrinsics info for verification
    all_intrinsics = recorder.get_intrinsics()
    print(f">>> Loaded Intrinsics: {all_intrinsics}")

    try:
        while True:
            data = recorder.get_frames()
            if not data:
                continue

            # Simple visualization
            display_imgs = []
            for name, frames in data.items():
                # Colorize depth map for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(frames['depth'], alpha=0.03), cv2.COLORMAP_JET)

                # Stack RGB and Depth side by side
                combined = np.hstack((frames['color'], depth_colormap))

                # Resize for display (half width)
                combined = cv2.resize(combined, (640, 240))

                # Add label
                cv2.putText(combined, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                display_imgs.append(combined)

            if display_imgs:
                # Vertically stack views from different cameras
                final_view = np.vstack(display_imgs)
                cv2.imshow('Dual RealSense Recorder', final_view)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                print(f"[Info] Saving snapshot_{ts} to {save_dir}...")

                # Example: Save images using cv2
                for name, frames in data.items():
                    cv2.imwrite(f"{save_dir}/{name}_color_{ts}.jpg", frames['color'])
                    # Save depth as raw 16-bit PNG (lossless)
                    cv2.imwrite(f"{save_dir}/{name}_depth_{ts}.png", frames['depth'])

                print("[Info] Snapshot saved.")

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
    except Exception as e:
        print(f"[Error] Unexpected error: {e}")
        
    finally:
        recorder.stop()
        cv2.destroyAllWindows()
