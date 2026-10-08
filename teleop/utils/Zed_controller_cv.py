import cv2
import time
import numpy as np


class ZEDCameraReader:
    def __init__(self, fps=30, camera_index=6):
        print("[Init] Configuring ZED M Camera in UVC (No-SDK) Mode...")
        self.cam = cv2.VideoCapture(camera_index)

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cam.set(cv2.CAP_PROP_FPS, fps)

        if not self.cam.isOpened():
            print(f"[Fatal] ZED Camera (Index {camera_index}) failed to open!")
            exit(1)

        self.intrinsics = {
            "ZED_M": {
                "width": 1280,
                "height": 720,
                "fx": 700.0,
                "fy": 700.0,
                "ppx": 640.0,
                "ppy": 360.0
            }
        }
        print(f"[Success] ZED M (UVC) started.")

    def get_frames(self):
        frames_data = {}
        ret, frame = self.cam.read()

        if ret:
            h, w, _ = frame.shape
            half_width = w // 2

            # 1. 切割图像
            left_bgr = frame[:, :half_width]
            right_bgr = frame[:, half_width:]

            # 2. 强制将 BGR 转换为 RGB (模型需要的格式)
            left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

            depth_uint16 = np.zeros((h, half_width), dtype=np.uint16)
            timestamp = int(time.time() * 1000)

            # 3. 将 RGB 数据存入字典
            frames_data["ZED_M"] = {
                "color": left_rgb,  # 喂给模型的左眼 RGB 画面
                "color_right": right_rgb,  # 喂给模型的右眼 RGB 画面
                "depth": depth_uint16,
                "timestamp": timestamp
            }

        return frames_data

    def get_intrinsics(self):
        return self.intrinsics

    def stop(self):
        print("[System] Closing ZED Camera (UVC)...")
        self.cam.release()


if __name__ == "__main__":
    reader = ZEDCameraReader()
    print("\n>>> Press 'q' to quit.")
    try:
        while True:
            frames = reader.get_frames()
            if "ZED_M" in frames:
                left = frames["ZED_M"]["color"]
                right = frames["ZED_M"]["color_right"]

                stereo_view = np.hstack((left, right))
                cv2.imshow("ZED M Stereo Vision (UVC)", cv2.resize(stereo_view, (1280, 360)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        reader.stop()
        cv2.destroyAllWindows()