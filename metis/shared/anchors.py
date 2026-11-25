"""
Where the 0 and the 1 of a score sit: one rule for calibration and utility.

An anchor is a reference measured with sampling noise (an unseen real sample,
uniform noise, an exact copy, a model without signal).  Placing the 1 at the
*mean* of the upper reference makes an independent real sample score below 1
half of the time, and the one-sided clip turns that into a mean below 1; the
same happens at 0.  So each anchor sits at the one-sided
``ANCHOR_CONFIDENCE`` quantile of its noise, towards the other anchor
(normal approximation, ``mean ∓ z·sd``): a score that cannot be told apart
from the upper reference is 1, one that cannot be told apart from the lower
reference is 0.

A score whose anchors are less than ``NOISE_MULTIPLE`` noise units apart
(their means, noise ``sqrt(sd_lower² + sd_upper²)``) cannot separate them and
is excluded (study ``calib_v2``).  With that margin the quantile anchors stay
ordered: ``z·(sd_l + sd_u) ≤ z·√2·noise ≈ 2.3·noise < 5·noise``.
"""

from scipy import stats

ANCHOR_CONFIDENCE = 0.95
ANCHOR_Z = float(stats.norm.ppf(ANCHOR_CONFIDENCE))
NOISE_MULTIPLE = 5.0

# Whether a real table carries signal for the utility metrics: TTR gain above
# this many times the noise of another real sample's gain (√2·σ).  5 until
# 2026-10-25; 3 since: with the target shuffled (no signal by construction,
# 240 measurements on 4 tables of 300 and 1000 rows) 1 in 240 cleared 3× and
# none 5×, while with the true target at 300 rows airbnb cleared 5× in 47 % of
# the samples and 3× in 80 % (all of them at 1000 rows).  The utility anchors
# stay ordered: the scale ``u − l = gain − z·(1 + √2)·σ`` is positive once
# ``gain > 3·√2·σ ≈ 4.24·σ > z·(1 + √2)·σ ≈ 3.97·σ``.  Decision §21 of the
# method decisions.
UTILITY_SIGNAL_MULTIPLE = 3.0

# Warning outside the score: a calibrated metric more than this many standard
# deviations of its upper reference (another real sample) below that
# reference's mean.  The averages of the hierarchy can hide a defect that one
# metric sees; the warning names it without changing the score.  On the
# defect bank (140 defects, 20 good tables) 3 warned 4 good tables (cardio,
# privacy.t_closeness at 3.0-3.6) and 4 none; 4 warned 116 defects.  The bank
# rebuilt with stratified splits (2026-11-12, 139 defects) gave 3 good tables
# warned at 4 and none at 5, the smallest such threshold, which warns 112
# defects: the same rule, set on the good tables only, now gives 5.  The floor
# keeps a metric whose real samples barely move from warning on rounding.
# Decision §24 of the method decisions.
WARNING_Z = 5.0
WARNING_SD_FLOOR = 0.005
