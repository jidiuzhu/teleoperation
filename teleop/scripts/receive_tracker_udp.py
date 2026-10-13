"""Display Tracker UDP diagnostics only; never drive hardware.

Receive timeout measures local silence, not end-to-end latency. UDP is
unauthenticated; use a trusted network. Robot integration is separate work.
"""
import argparse
import json
import math
import socket
import time


def validate(packet, serial):
    if not isinstance(packet, dict):
        raise ValueError('Expected object')
    if type(packet.get('version')) is not int or packet['version'] not in (1, 2):
        raise ValueError(f"Unsupported protocol version {packet.get('version')!r}; expected 1 or 2")
    if packet.get('serial') != serial:
        raise ValueError(f"Tracker serial mismatch: expected {serial!r}, received {packet.get('serial')!r}")
    if packet.get('frame') != 'steamvr_standing':
        raise ValueError('Wrong frame')
    if not isinstance(packet.get('session'), str) or not packet['session']:
        raise ValueError('Missing session')
    if type(packet.get('seq')) is not int or packet['seq'] < 0:
        raise ValueError('Invalid sequence')
    if type(packet.get('valid')) is not bool or type(packet.get('connected')) is not bool:
        raise ValueError('Invalid flags')
    if packet['valid']:
        if not packet['connected'] or packet.get('tracking_result') != 200:
            raise ValueError('Inconsistent tracking status')
        m = packet.get('matrix34')
        if not isinstance(m, list) or len(m) != 3:
            raise ValueError('Invalid matrix')
        for row in m:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError('Invalid matrix row')
            if any(type(v) not in (float, int) or not math.isfinite(v) for v in row):
                raise ValueError('Non-finite matrix')
        for i in range(3):
            for j in range(3):
                if abs(sum(m[k][i] * m[k][j] for k in range(3)) - (i == j)) > 0.005:
                    raise ValueError('Invalid rotation')
        det = (m[0][0] * (m[1][1]*m[2][2]-m[1][2]*m[2][1])
               - m[0][1] * (m[1][0]*m[2][2]-m[1][2]*m[2][0])
               + m[0][2] * (m[1][0]*m[2][1]-m[1][1]*m[2][0]))
        if abs(det - 1) > 0.005:
            raise ValueError('Invalid rotation determinant')
    elif packet.get('matrix34') is not None:
        raise ValueError('Invalid tracking must not include pose')
    return packet


class RejectionReporter:
    """Rate-limit rejected-packet output without accepting invalid packets."""
    def __init__(self, interval=5.0):
        self.interval = interval
        self.last_print = None
        self.suppressed = 0

    def report(self, error, now):
        if self.last_print is not None and now-self.last_print < self.interval:
            self.suppressed += 1
            return
        suffix = f" ({self.suppressed} rejection messages suppressed)" if self.suppressed else ""
        print(f"Rejected packet: {error}{suffix}", flush=True)
        self.last_print = now
        self.suppressed = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=6210)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--source-ip', help='Only accept this Windows IP')
    parser.add_argument('--timeout', type=float, default=0.25)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive and finite')
    reporter = RejectionReporter()
    last_rx = None
    last_print = 0.0
    status = None
    active = None
    last_seq = -1
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((args.bind, args.port))
        sock.settimeout(min(args.timeout, 0.1))
        print(f'Listening UDP {args.bind}:{args.port}; diagnostics only', flush=True)
        while True:
            try:
                data, addr = sock.recvfrom(8192)
                if args.source_ip and addr[0] != args.source_ip:
                    continue
                p = validate(json.loads(data), args.serial)
                identity = (addr, p['session'])
                # A new sender/session can take over only after local silence.
                if identity != active:
                    if last_rx is not None and time.monotonic() - last_rx < args.timeout:
                        continue
                    active, last_seq = identity, -1
                if p['seq'] <= last_seq:
                    continue
                gap = max(0, p['seq'] - last_seq - 1) if last_seq >= 0 else 0
                last_seq = p['seq']
                last_rx = time.monotonic()
                current = 'TRACKING' if p['valid'] else 'INVALID'
                if current != status or last_rx - last_print >= 0.5:
                    labels = {1: 'Uninitialized', 100: 'Calibrating_InProgress',
                              101: 'Calibrating_OutOfRange', 200: 'Running_OK',
                              201: 'Running_OutOfRange', 300: 'Fallback_RotationOnly'}
                    result = p.get('tracking_result')
                    label = labels.get(result, 'Unknown') if type(result) is int else 'Unavailable'
                    detail = f" connected={p['connected']} tracking_result={result}({label})"
                    if not p['valid'] and result == 200:
                        detail += ' pose invalid or matrix non-finite at sender'
                    if p['valid']:
                        m = p['matrix34']
                        detail += ' xyz_m=' + str([round(m[r][3], 4) for r in range(3)])
                    print(f"{current} serial={p['serial']} seq={last_seq} gap={gap}" + detail, flush=True)
                    last_print = last_rx
                status = current
            except socket.timeout:
                pass
            except (ValueError, TypeError, UnicodeError) as exc:
                reporter.report(exc, time.monotonic())
            if last_rx is None or time.monotonic() - last_rx >= args.timeout:
                if status != 'STALE':
                    print('STALE: no recent packets; pose unavailable', flush=True)
                    status = 'STALE'


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
