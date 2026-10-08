import pyzed.sl as sl
import numpy as np
import cv2
import time
import os


class ZEDCameraReader:
    def __init__(self, fps=30):
        print("[Init] Configuring ZED M Camera...")
        self.cam = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = fps

        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.depth_minimum_distance = 150

        err = self.cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[Fatal] ZED Camera failed to open: {err}")
            exit(1)

        self.runtime_params = sl.RuntimeParameters()

        self.mat_left = sl.Mat()
        self.mat_right = sl.Mat()
        self.mat_depth = sl.Mat()

        calib = self.cam.get_camera_information().camera_configuration.calibration_parameters
        left_cam = calib.left_cam
        self.intrinsics = {
            "ZED_M": {
                "width": left_cam.image_size.width,
                "height": left_cam.image_size.height,
                "fx": left_cam.fx,
                "fy": left_cam.fy,
                "ppx": left_cam.cx,
                "ppy": left_cam.cy
            }
        }
        print(f"[Success] ZED M started. Intrinsics: {self.intrinsics['ZED_M']}")

    def get_frames(self):
        frames_data = {}

        if self.cam.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.cam.retrieve_image(self.mat_left, sl.VIEW.LEFT)
            self.cam.retrieve_image(self.mat_right, sl.VIEW.RIGHT)

            self.cam.retrieve_measure(self.mat_depth, sl.MEASURE.DEPTH)

            left_bgr = cv2.cvtColor(self.mat_left.get_data(), cv2.COLOR_BGRA2BGR)
            right_bgr = cv2.cvtColor(self.mat_right.get_data(), cv2.COLOR_BGRA2BGR)

            depth_float = self.mat_depth.get_data()
            depth_uint16 = np.nan_to_num(depth_float, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16)

            timestamp = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

            frames_data["ZED_M"] = {
                "color": left_bgr,
                "color_right": right_bgr,
                "depth": depth_uint16,
                "timestamp": timestamp
            }

        return frames_data

    def get_intrinsics(self):
        return self.intrinsics

    def stop(self):
        print("[System] Closing ZED Camera...")
        self.cam.close()


if __name__ == "__main__":
    reader = ZEDCameraReader()
    print("\n>>> Press 'q' to quit.")
    try:
        while True:
            frames = reader.get_frames()
            if "ZED_M" in frames:
                left = frames["ZED_M"]["color"]
                right = frames["ZED_M"]["color_right"]
                depth = frames["ZED_M"]["depth"]

                stereo_view = np.hstack((left, right))
                cv2.imshow("ZED M Stereo Vision", cv2.resize(stereo_view, (1280, 360)))

                depth_clipped = np.clip(depth, 200, 1500)

                depth_norm = cv2.normalize(depth_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_8U)

                depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                cv2.imshow("ZED M Neural Depth", cv2.resize(depth_colormap, (640, 360)))
                # ------------------------------

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        reader.stop()
        cv2.destroyAllWindows()