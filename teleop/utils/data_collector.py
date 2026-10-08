import os
import time
import h5py
import numpy as np
from datetime import datetime


class DataCollector:
    """
    Handles data buffering and HDF5 file saving for robotic data collection.
    """

    def __init__(self, save_dir: str = "data_records"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.recording = False
        self.buffer = []
        self.start_time = 0.0
        self.task_name = "default"

        # --- Profiling Variables ---
        self.frame_count = 0

    def start_recording(self, task_name: str):
        """Start a new recording session."""
        self.task_name = task_name if task_name else "default"
        self.buffer = []
        self.recording = True
        self.start_time = time.time()
        self.frame_count = 0
        print(f"[DataCollector] Recording STARTED. Task: {self.task_name}")

    def stop_recording(self):
        """Stop recording, print stats, and flush to disk."""
        if not self.recording:
            return

        self.recording = False
        duration = time.time() - self.start_time
        frames = len(self.buffer)

        # --- Performance Report ---
        avg_fps = frames / duration if duration > 0 else 0.0
        print("-" * 40)
        print(f"[DataCollector] Report for '{self.task_name}':")
        print(f"  Total Duration : {duration:.2f} s")
        print(f"  Total Frames   : {frames}")
        print(f"  Average FPS    : {avg_fps:.2f} Hz (Target: 15.0 Hz)")

        if avg_fps < 29.0:
            print("  [WARN] Data collection frequency is unstable!")
        else:
            print("  [PASS] Frequency is stable.")
        print("-" * 40)

        if frames > 0:
            self._flush_to_disk()
        else:
            print("[DataCollector] Buffer empty, nothing to save.")

    def add_frame(self, frame_data: dict):
        """Add a single timestep of data to the buffer."""
        if not self.recording:
            return
        self.buffer.append(frame_data)
        self.frame_count += 1

    def _flush_to_disk(self):
        """Internal method to write the RAM buffer to an HDF5 file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.task_name}_{timestamp}.h5"
        filepath = os.path.join(self.save_dir, filename)

        print(f"[DataCollector] Saving to {filepath} ... (Do not kill process)")
        t0 = time.time()

        try:
            data_dict = self._stack_buffer(self.buffer)
            with h5py.File(filepath, 'w') as f:
                f.attrs['task_name'] = self.task_name
                f.attrs['total_frames'] = len(self.buffer)
                f.attrs['created_at'] = timestamp
                f.attrs['avg_fps'] = self.frame_count / (time.time() - self.start_time)  # Save FPS to file
                self._save_recursive(f, data_dict)

            print(f"[DataCollector] Save COMPLETE. Time taken: {time.time() - t0:.2f}s")

        except Exception as e:
            print(f"[DataCollector] Save FAILED: {e}")
            import traceback
            traceback.print_exc()

    def _stack_buffer(self, buffer_list):
        if not buffer_list: return {}
        first_frame = buffer_list[0]
        stacked = {}
        for key, value in first_frame.items():
            if isinstance(value, dict):
                nested_list = [frame[key] for frame in buffer_list]
                stacked[key] = self._stack_buffer(nested_list)
            else:
                try:
                    stacked[key] = np.array([frame[key] for frame in buffer_list])
                except Exception as e:
                    print(f"[Warn] Failed to stack key '{key}': {e}")
        return stacked

    def _save_recursive(self, group, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                sub_group = group.create_group(key)
                self._save_recursive(sub_group, value)
            else:
                group.create_dataset(key, data=value)
