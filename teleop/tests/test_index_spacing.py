import unittest
from types import SimpleNamespace
import numpy as np
from scripts.adjust_index_spacing import spacing_plan,apply_once
from utils.wuji_hand_spec import hand_spec

class Clock:
    def __init__(self):self.t=0
    def __call__(self):return self.t
    def sleep(self,dt):self.t+=dt

class Hand:
    def __init__(self,start):
        self.q=start.copy();self.mask=np.zeros((5,4),bool);self.enables=[];self.targets=[];self.closed=False
        self.limit=.3;self.fail=False;self.opposite=False
    def read_joint_actual_position(self,**kw):
        if self.fail:raise ConnectionError('feedback failed')
        return self.q.reshape(5,4).copy()
    def read_joint_error_code(self,**kw):return np.zeros((5,4),int)
    def write_joint_enabled(self,value,**kw):
        self.mask=np.full((5,4),value,bool) if isinstance(value,bool) else value.copy()
        self.enables.append(self.mask.copy())
    def finger(self,i):
        assert i==1
        return SimpleNamespace(joint=self.joint)
    def joint(self,i):assert i==1;return self
    def read_joint_effort_limit(self,**kw):return self.limit
    def write_joint_effort_limit(self,v,**kw):self.limit=v
    def realtime_controller(self,**kw):return self
    def set_joint_target_position(self,q):
        self.targets.append(q.copy())
        if self.mask[1,1]:self.q[5]=q[1,1] if not self.opposite else self.q[5]+.01
    def close(self):self.closed=True

class SpacingTests(unittest.TestCase):
    def setup_plan(self):
        q=np.zeros(20);q[5]=.18
        lim=hand_spec('left')[2]
        target,lo,hi,before,after=spacing_plan(q,lim[:,0],lim[:,1],-.02)
        self.assertGreater(after,before)
        return q,target,lo,hi
    def test_only_index_j2_enabled_and_never_returns(self):
        q,t,lo,hi=self.setup_plan();h=Hand(q);c=Clock()
        sdk=SimpleNamespace(filter=SimpleNamespace(LowPass=lambda **kw:None))
        result=apply_once(h,sdk,q,t,lo,hi,clock=c,sleep=c.sleep)
        self.assertAlmostEqual(result[5],t[5],places=4)
        for mask in h.enables:
            self.assertFalse(np.delete(mask.reshape(20),5).any())
        self.assertFalse(h.mask.any());self.assertTrue(h.closed)
        targets=np.array(h.targets).reshape(-1,20)
        self.assertTrue(np.all(np.diff(targets[:,5])<=1e-10))
        np.testing.assert_allclose(targets[:,np.arange(20)!=5],np.tile(q[np.arange(20)!=5],(len(targets),1)))
    def test_opposite_motion_stops_and_disables(self):
        q,t,lo,hi=self.setup_plan();h=Hand(q);h.opposite=True;c=Clock()
        sdk=SimpleNamespace(filter=SimpleNamespace(LowPass=lambda **kw:None))
        with self.assertRaisesRegex(RuntimeError,'opposite'):
            apply_once(h,sdk,q,t,lo,hi,clock=c,sleep=c.sleep)
        self.assertFalse(h.mask.any());self.assertTrue(h.closed)
    def test_setup_feedback_failure_disables(self):
        q,t,lo,hi=self.setup_plan();h=Hand(q);h.fail=True;c=Clock()
        with self.assertRaises(ConnectionError):apply_once(h,None,q,t,lo,hi,clock=c,sleep=c.sleep)
        self.assertFalse(h.mask.any());self.assertEqual(len(h.targets),0)
    def test_delta_restrictions(self):
        q,t,lo,hi=self.setup_plan()
        for delta in [.02,-.1,float('nan'),0]:
            with self.assertRaises(ValueError):spacing_plan(q,lo,hi,delta)

if __name__=='__main__':unittest.main()
