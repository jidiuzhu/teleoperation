"""Wuji-specific Vuer preprocessing without changing the XHand path."""

from teleop.Preprocessor import VuerPreprocessor


class VuerPreprocessorWuji(VuerPreprocessor):
    """Use the existing wrist-axis calibration for the identically mounted Wuji hand."""

    def process(self, tv, hand="wuji"):
        if hand != "wuji":
            raise ValueError(f"VuerPreprocessorWuji only supports 'wuji', got {hand!r}")
        # The installation contract states that Wuji and XHand have the same
        # palm position/orientation convention, so hand2xhand remains valid.
        return super().process(tv, "xhand")
