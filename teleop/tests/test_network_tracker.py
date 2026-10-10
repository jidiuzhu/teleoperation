import json
import socket
import time
import unittest
import numpy as np
from utils.network_tracker import NetworkWristTracker


class NetworkTrackerTests(unittest.TestCase):
    def setUp(self):
        self.receiver=NetworkWristTracker('TEST',bind='127.0.0.1',port=0,source_ip='127.0.0.1',timeout_s=.12)
        self.sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.sender.bind(('127.0.0.1',0)); self.sender.settimeout(.2)
        self.target=self.receiver.sock.getsockname()
        self.packet=dict(version=2,session='session1',seq=0,serial='TEST',frame='steamvr_standing',
                         connected=True,valid=True,tracking_result=200,echo=None,
                         matrix34=[[1,0,0,.2],[0,1,0,.3],[0,0,1,.4]])
        self.addCleanup(self.sender.close);self.addCleanup(self.receiver.close)
    def send(self,**kwargs):
        self.packet.update(kwargs)
        self.sender.sendto(json.dumps(self.packet).encode(),self.target)
        return self.receiver.read_world_wrist()
    def handshake(self):
        self.assertIsNone(self.send())
        probe=json.loads(self.sender.recvfrom(2048)[0]);self.packet['echo']=probe['nonce']
        return self.send(seq=1)
    def test_pose_order_invalid_and_recovery(self):
        pose=self.handshake();np.testing.assert_allclose(pose[:3,3],[.2,.3,.4])
        self.send(seq=0,matrix34=[[1,0,0,9],[0,1,0,9],[0,0,1,9]])
        np.testing.assert_allclose(self.receiver.pose[:3,3],[.2,.3,.4])
        self.assertIsNone(self.send(seq=2,valid=False,matrix34=None,tracking_result=101))
        self.assertIsNone(self.receiver.read_world_wrist())
    def test_old_challenge_cannot_refresh_delayed_pose(self):
        self.handshake();time.sleep(.14)
        self.assertIsNone(self.send(seq=20))
        self.assertEqual(self.receiver.seconds_since_valid(),float('inf'))
    def test_malformed_rotation_and_wrong_serial_rejected(self):
        self.assertIsNone(self.send(serial='OTHER'))
        self.packet['serial']='TEST';self.handshake()
        self.assertIsNone(self.send(seq=2,matrix34=[[-1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    def test_legacy_diagnostic_packets_are_not_motion_input(self):
        self.assertIsNone(self.send(version=1))
    def test_sender_restart_requires_new_handshake(self):
        self.handshake();time.sleep(.14)
        self.assertIsNone(self.send(session='session2',seq=0,echo=None))
        # Drain probes and use the current-session challenge.
        nonce=None
        while True:
            try:nonce=json.loads(self.sender.recvfrom(2048)[0])['nonce']
            except socket.timeout:break
            self.sender.settimeout(.001)
        self.assertIsNotNone(self.send(seq=1,echo=nonce))


if __name__=='__main__':unittest.main()
