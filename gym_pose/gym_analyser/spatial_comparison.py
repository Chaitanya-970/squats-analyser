"""Spatial comparison of two corresponding frames of 3D joint estimates."""

from math import acos, degrees, dist, isfinite
from typing import Mapping, Sequence

Pose = Mapping[str, Sequence[float]]

ANGLES = {
    "left_knee_deg": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_deg": ("right_hip", "right_knee", "right_ankle"),
    "left_hip_deg": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_deg": ("right_shoulder", "right_hip", "right_knee"),
}


def _point(pose: Pose, name: str) -> tuple[float, ...] | None:
    point = pose.get(name)
    if point is None or len(point) != 3 or not all(isfinite(v) for v in point):
        return None
    return tuple(point)


def measure_pose(pose: Pose) -> dict[str, float]:
    """Measure one frame, omitting unavailable or degenerate measurements."""
    result = {}
    for metric, names in ANGLES.items():
        a, center, b = (_point(pose, name) for name in names)
        if a is None or center is None or b is None:
            continue
        length_a, length_b = dist(a, center), dist(b, center)
        if min(length_a, length_b) == 0:
            continue
        cosine = sum(
            ((x - c) / length_a) * ((y - c) / length_b)
            for x, c, y in zip(a, center, b)
        )
        result[metric] = degrees(acos(max(-1.0, min(1.0, cosine))))
    return result


def compare_poses(
    observed: Pose, reference: Pose
) -> dict[str, dict[str, float] | None]:
    """Compare matched frames. Positive differences mean a larger interior angle.

    Use matching anatomical left/right joint names and a common scale across
    axes within each pose. Omit low-confidence joints before calling. Frame
    selection, timestamps and sequence alignment belong to the caller.
    """
    actual = measure_pose(observed)
    target = measure_pose(reference)
    return {
        metric: {
            "observed": actual[metric],
            "reference": target[metric],
            "difference": actual[metric] - target[metric],
        }
        if metric in actual and metric in target else None
        for metric in ANGLES
    }
