"""Wuji-specific OpenTeleVision lifecycle and tracking freshness support."""

from multiprocessing import Value
import time

import numpy as np

from teleop.TeleVision import OpenTeleVision


class OpenTeleVisionWuji(OpenTeleVision):
    """Expose right-hand update age and deterministic child-process cleanup."""

    def __init__(self, *args, **kwargs):
        self._right_hand_update_time = Value("d", 0.0, lock=True)
        super().__init__(*args, **kwargs)

    async def on_hand_move(self, event, session, fps=60):
        valid_right_hand = False
        try:
            right_hand = np.asarray(event.value["rightHand"], dtype=np.float64)
            right_landmarks = np.asarray(
                event.value["rightLandmarks"], dtype=np.float64
            )
            valid_right_hand = (
                right_hand.size == 16
                and right_landmarks.size == 75
                and np.all(np.isfinite(right_hand))
                and np.all(np.isfinite(right_landmarks))
            )
        except (KeyError, TypeError, ValueError):
            pass

        await super().on_hand_move(event, session, fps=fps)
        if valid_right_hand:
            with self._right_hand_update_time.get_lock():
                self._right_hand_update_time.value = time.monotonic()

    @property
    def right_hand_update_age(self):
        with self._right_hand_update_time.get_lock():
            update_time = self._right_hand_update_time.value
        if update_time <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - update_time)

    def close(self):
        for name in ("webrtc_process", "process"):
            process = getattr(self, name, None)
            if process is None:
                continue
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)


__all__ = ["OpenTeleVisionWuji"]
