"""Read-only OpenVR to UDP diagnostic bridge. No robot commands."""
import argparse
import json
import math
import socket
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='Ubuntu receiver IP')
    parser.add_argument('--port', type=int, default=6210)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--hz', type=float, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.hz) or not 1 <= args.hz <= 120:
        parser.error('--hz must be between 1 and 120')
    import openvr
    vr = openvr.init(openvr.VRApplication_Background)
    session = uuid.uuid4().hex
    seq = 0
    last_print = 0.0
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            print(f'Sending {args.serial} to {args.host}:{args.port}; Ctrl+C exits')
            while True:
                start = time.monotonic()
                packet = dict(version=1, session=session, seq=seq,
                              serial=args.serial, frame='steamvr_standing',
                              connected=False, valid=False, tracking_result=None,
                              matrix34=None)
                poses = vr.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding, 0,
                    openvr.k_unMaxTrackedDeviceCount)
                for i, pose in enumerate(poses):
                    if not pose.bDeviceIsConnected:
                        continue
                    if vr.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_GenericTracker:
                        continue
                    serial = vr.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
                    if isinstance(serial, bytes):
                        serial = serial.decode('utf-8')
                    if serial != args.serial:
                        continue
                    packet['connected'] = True
                    packet['tracking_result'] = int(pose.eTrackingResult)
                    valid = bool(pose.bPoseIsValid and
                                 pose.eTrackingResult == openvr.TrackingResult_Running_OK)
                    matrix = [[float(pose.mDeviceToAbsoluteTracking[r][c])
                               for c in range(4)] for r in range(3)]
                    packet['valid'] = valid and all(math.isfinite(v) for row in matrix for v in row)
                    if packet['valid']:
                        packet['matrix34'] = matrix
                    break
                sock.sendto(json.dumps(packet, allow_nan=False).encode('utf-8'),
                            (args.host, args.port))
                if start - last_print >= 1:
                    print(f"seq={seq} connected={packet['connected']} valid={packet['valid']}", flush=True)
                    last_print = start
                seq += 1
                time.sleep(max(0, 1 / args.hz - (time.monotonic() - start)))
    finally:
        openvr.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
