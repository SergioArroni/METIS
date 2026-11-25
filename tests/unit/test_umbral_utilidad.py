"""Utility applies when the TTR gain clears 3 times its noise (2026-10-25, decisions §21)."""

from __future__ import annotations

import math

from metis.infrastructure.metrics.utility import ml_efficiency_metric as mem
from metis.shared.anchors import ANCHOR_Z, NOISE_MULTIPLE, UTILITY_SIGNAL_MULTIPLE


def test_the_utility_threshold_is_three_and_the_calibration_one_stays_five():
    assert UTILITY_SIGNAL_MULTIPLE == 3.0
    assert NOISE_MULTIPLE == 5.0  # excluding calibrated metrics is a different rule


def test_a_gain_between_three_and_five_noises_now_applies():
    noise = 0.01
    assert mem._no_signal_reason(4.0 * noise, noise) is None
    reason = mem._no_signal_reason(2.5 * noise, noise)
    assert reason is not None and "3× its sampling noise" in reason
    assert mem._no_signal_reason(-0.001, noise) is not None


def test_just_above_the_threshold_the_utility_anchors_are_ordered():
    sigma = 0.01
    gain = UTILITY_SIGNAL_MULTIPLE * math.sqrt(2.0) * sigma * 1.0001
    lower, upper = ANCHOR_Z * sigma, gain - ANCHOR_Z * math.sqrt(2.0) * sigma
    assert upper > lower
