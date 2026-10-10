from pathlib import Path
from types import SimpleNamespace
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

TELEOP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TELEOP_ROOT))

from scripts.teleop_real_client_wuji import parse_args
from utils.glove_tracker_input import GloveTrackerInput, OPENVR_TO_ROBOT_TRACKING
from utils.vive_tracker import make_transform, matrix34_to_transform
from utils.wuji_hand_spec import JOINT_LIMITS


class FakeSubscription:
    def __init__(self):
        self.enabled = True
        self.pending = True
        self.closed = False

    def recv(self):
        if not self.enabled or not self.pending:
            self.pending = True
            return None
        self.pending = False
        joints = [SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=0, y=0, z=0))) for _ in range(21)]
        return SimpleNamespace(joints=joints)

    def close(self):
        self.closed = True


class FakeTracker:
    def __init__(self, serial, **_kwargs):
        self.serial = serial
        self.pose = np.eye(4)
        self.closed = False

    def wait_and_zero(self, **_kwargs):
        pass

    def read_world_wrist(self):
        return self.pose

    def seconds_since_valid(self, _now):
        return 0.0

    def close(self):
        self.closed = True


class FakeManager:
    def __init__(self, subscription):
        self.subscription = subscription
        self.disconnected = False

    def scan(self):
        return [SimpleNamespace(device_type="glove", sn="glove-1")]

    def connect(self, **_kwargs):
        subscription = self.subscription
        return SimpleNamespace(
            hand_side=lambda: SimpleNamespace(get=lambda: "Right"),
            hand_skeleton=lambda: SimpleNamespace(subscribe=lambda: subscription),
        )

    def disconnect_all(self):
        self.disconnected = True


class GloveInputTest(unittest.TestCase):
    def test_tracker_reflection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "right-handed"):
            matrix34_to_transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    def test_tracker_mount_is_applied_before_axis_conversion(self):
        tracker = matrix34_to_transform([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
        ])
        mount = make_transform((0.1, 0.2, 0.3), (0, 0, 0))
        wrist = OPENVR_TO_ROBOT_TRACKING @ tracker @ mount
        np.testing.assert_allclose(wrist[:3, 3], [-1.1, 3.3, 2.2])

    def test_parser_accepts_requested_devise_and_rejects_missing_serials(self):
        with patch.object(sys, "argv", ["client", "--devise", "glove", "--glove-sn", "a", "--tracker-serial", "b"]):
            self.assertEqual(parse_args().input_device, "glove")
        with patch.object(sys, "argv", ["client", "--devise", "glove"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_glove_and_tracker_share_finite_command_contract(self):
        subscription = FakeSubscription()
        manager = FakeManager(subscription)
        sdk = SimpleNamespace(
            DeviceType=SimpleNamespace(WujiGlove="glove"),
            HandModel=SimpleNamespace(WujiHand="hand"),
            Handedness=SimpleNamespace(Right="right"),
            RetargetSession=SimpleNamespace(for_hand=lambda *_args, **_kwargs: SimpleNamespace(step=lambda _points: np.zeros(20))),
            SdkManager=SimpleNamespace(instance=lambda: manager),
        )
        with patch.dict(sys.modules, {"wuji_sdk": sdk}), patch("utils.glove_tracker_input.ViveWristTracker", FakeTracker):
            input_device = GloveTrackerInput(glove_serial="glove-1", tracker_serial="tracker-1", startup_valid_frames=2)
            wrist, hand = input_device.step()
            np.testing.assert_allclose(wrist, OPENVR_TO_ROBOT_TRACKING)
            np.testing.assert_allclose(hand, np.clip(np.zeros(20), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]))
            subscription.enabled = False
            input_device.last_glove_time = time.monotonic() - 1.0
            self.assertIsNone(input_device.step())
            input_device.close()
            self.assertTrue(manager.disconnected)
            self.assertTrue(subscription.closed)

    def test_left_glove_selects_left_retargeting(self):
        subscription=FakeSubscription()
        manager=FakeManager(subscription)
        manager.connect=lambda **kwargs: SimpleNamespace(
            hand_side=lambda:SimpleNamespace(get=lambda:"Left"),
            hand_skeleton=lambda:SimpleNamespace(subscribe=lambda:subscription))
        calls=[]
        def retarget(model,side):
            calls.append((model,side))
            return SimpleNamespace(step=lambda points:np.zeros(20))
        sdk=SimpleNamespace(DeviceType=SimpleNamespace(WujiGlove="glove"),
            HandModel=SimpleNamespace(WujiHand="hand"),
            Handedness=SimpleNamespace(Right="right",Left="left"),
            RetargetSession=SimpleNamespace(for_hand=retarget),
            SdkManager=SimpleNamespace(instance=lambda:manager))
        with patch.dict(sys.modules,{"wuji_sdk":sdk}),patch("utils.glove_tracker_input.ViveWristTracker",FakeTracker):
            inp=GloveTrackerInput(glove_serial="glove-1",tracker_serial="tracker-1",hand_side="left",startup_valid_frames=2)
            try:
                self.assertEqual(calls,[("hand","left")])
                self.assertIsNotNone(inp.step())
            finally: inp.close()


if __name__ == "__main__":
    unittest.main()
