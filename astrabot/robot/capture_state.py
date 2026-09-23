"""Shared stationary-state eligibility for capture and offline image loading."""

CAPTURE_BODY_DRIFT_RAD = 0.005
CAPTURE_HAND_DRIFT_RAW = 2.0


def stationary_capture_state(state):
    """A live task can retain RUNNING ownership with no motion or planning."""
    return (
        state.get("state") in ("DISARMED", "HOLD", "READY", "RUNNING")
        and state.get("remaining_segments") == 0
        and not state.get("planning_active", False)
    )
