import contextlib
import io
import unittest
from scripts.receive_tracker_udp import validate, RejectionReporter


class DiagnosticsTests(unittest.TestCase):
    def packet(self, version=2):
        return dict(version=version, serial='LHR-C87FB5D3', frame='steamvr_standing',
                    session='test', seq=0, connected=True, valid=False,
                    tracking_result=101, matrix34=None)

    def test_missing_serial_character_reports_both_values(self):
        with self.assertRaisesRegex(ValueError, "expected 'LHR-C87FB5D', received 'LHR-C87FB5D3'"):
            validate(self.packet(), 'LHR-C87FB5D')

    def test_both_diagnostic_versions_accept_correct_serial(self):
        for version in (1, 2):
            self.assertEqual(validate(self.packet(version), 'LHR-C87FB5D3')['serial'], 'LHR-C87FB5D3')

    def test_bad_version_has_distinct_message(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported protocol version'):
            validate(self.packet(3), 'LHR-C87FB5D3')

    def test_rejections_are_throttled_and_counted(self):
        reporter = RejectionReporter()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            reporter.report('serial mismatch', 0)
            for i in range(1, 100):
                reporter.report('serial mismatch', i / 100)
            reporter.report('serial mismatch', 5)
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertIn('99 rejection messages suppressed', output.getvalue())


if __name__ == '__main__': unittest.main()
