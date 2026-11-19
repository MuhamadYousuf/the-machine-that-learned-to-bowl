"""SI-unit dimensions and shared physical constants for the AI Bowler project."""

from math import pi

# The coordinate origin is the centre of the bowler's stumps. The batter's
# stumps are one regulation pitch length away in the positive x direction.
PITCH_LENGTH_M = 20.12
POPPING_CREASE_FROM_STUMPS_M = 1.22
BOWLER_STUMPS_X_M = 0.0
BATTER_STUMPS_X_M = BOWLER_STUMPS_X_M + PITCH_LENGTH_M
BATTER_POPPING_CREASE_X_M = BATTER_STUMPS_X_M - POPPING_CREASE_FROM_STUMPS_M
BOWLER_POPPING_CREASE_X_M = BOWLER_STUMPS_X_M - POPPING_CREASE_FROM_STUMPS_M

STUMPS_HEIGHT_M = 0.711
STUMPS_WIDTH_M = 0.2286

# The approximate one-day off-side return/wide guideline is 0.89 m from the
# middle stump line. The leg-side wide limit is simplified to 0.45 m outside
# leg stump. Both are centre-line thresholds at the popping crease; batter
# reach and umpire judgement are not modelled.
WIDE_GUIDELINE_M = 0.89
LEG_SIDE_WIDE_GUIDELINE_M = 0.45

BALL_MASS_KG = 0.156
BALL_RADIUS_M = 0.036
BALL_CROSS_SECTION_M2 = pi * BALL_RADIUS_M**2
GRAVITY_M_S2 = 9.81
AIR_DENSITY_KG_M3 = 1.2

# Release is at the bowler's popping crease, about 2.2 m high and 0.3 m
# off-side of the middle-stump line for a right-arm bowler over the wicket.
RELEASE_POSITION_M = (BOWLER_POPPING_CREASE_X_M, 0.30, 2.20)

INTEGRATION_DT_S = 0.002
DRAG_COEFFICIENT = 0.40
DEFAULT_RESTITUTION = 0.67
DEFAULT_RESTITUTION_IMPACT_SLOPE_PER_RAD = 0.55
DEFAULT_RESTITUTION_REFERENCE_ANGLE_RAD = 0.15
DEFAULT_HORIZONTAL_RETENTION = 0.96

# Acceleration = swing * coefficient * speed**2. The coefficient is tuned
# against the 21 m flight so full swing produces about 0.3-0.6 m of drift.
SWING_ACCEL_COEFFICIENT_PER_M = 0.0023

SPEED_RANGE_KMH = (115.0, 150.0)
ELEVATION_RANGE_DEG = (-12.0, 2.0)
AZIMUTH_OFFSET_RANGE_DEG = (-2.0, 2.0)
