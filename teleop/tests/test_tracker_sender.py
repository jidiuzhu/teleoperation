"""Exercise sender startup using Winsock's unbound-receive behavior."""
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from scripts import send_tracker_windows as sender


class WindowsSocket:
    def __init__(self):
        self.bound = False
        self.packets = []
        self.pending = []
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def bind(self, address):
        if address != ('0.0.0.0', 0):
            raise AssertionError('Sender must request its own ephemeral port')
        self.bound = True
    def setblocking(self, value): pass
    def recvfrom(self, size):
        if not self.bound:
            raise OSError(10022, 'Socket must be bound before receive')
        if not self.pending:
            raise BlockingIOError()
        item = self.pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    def sendto(self, payload, destination):
        self.packets.append(json.loads(payload))
        if len(self.packets) == 1:
            self.pending.append((json.dumps({'type':'tracker_probe','nonce':'test-nonce'}).encode(), destination))
        else:
            raise KeyboardInterrupt()


class SenderTest(unittest.TestCase):
    def test_startup_binds_then_sends_pose_and_echoes_challenge(self):
        self.exercise_sender(WindowsSocket())

    def test_windows_udp_reset_does_not_prevent_recovery(self):
        sock = WindowsSocket()
        error = ConnectionResetError('UDP peer unavailable')
        error.winerror = 10054
        sock.pending.append(error)
        self.exercise_sender(sock)

    def exercise_sender(self, sock):
        pose = SimpleNamespace(bDeviceIsConnected=True, bPoseIsValid=True,
            eTrackingResult=200, mDeviceToAbsoluteTracking=[[1,0,0,.1],[0,1,0,.2],[0,0,1,.3]])
        vr = SimpleNamespace(getDeviceToAbsoluteTrackingPose=lambda *a:[pose],
            getTrackedDeviceClass=lambda i:3,
            getStringTrackedDeviceProperty=lambda *a:'TEST')
        shutdown = []
        sdk = SimpleNamespace(VRApplication_Background=3, TrackingUniverseStanding=1,
            k_unMaxTrackedDeviceCount=1, TrackedDeviceClass_GenericTracker=3,
            Prop_SerialNumber_String=1002, TrackingResult_Running_OK=200,
            init=lambda mode:vr, shutdown=lambda:shutdown.append(True))
        with patch.dict(sys.modules, {'openvr':sdk}), patch.object(sys,'argv',
            ['sender','--host','127.0.0.1','--serial','TEST']), \
            patch.object(sender.socket,'socket',return_value=sock), \
            patch.object(sender.time,'sleep'):
            with self.assertRaises(KeyboardInterrupt): sender.main()
        self.assertTrue(sock.packets[0]['valid'])
        self.assertIsNone(sock.packets[0]['echo'])
        self.assertEqual(sock.packets[1]['echo'],'test-nonce')
        self.assertEqual(sock.packets[1]['seq'],1)
        self.assertEqual(shutdown,[True])


if __name__ == '__main__': unittest.main()
