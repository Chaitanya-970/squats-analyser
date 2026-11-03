"""Export estimated 3D joint frames for spatial comparison."""

import argparse
import json
from math import isfinite
from pathlib import Path

from .auxiliary_pose import MediaPipeLandmarker
from .spatial_comparison import measure_pose

JOINTS = dict(zip(
    ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
     "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
     "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
     "left_toe", "right_toe"),
    (*range(11, 17), *range(23, 33)),
))



def extract_video(video, model, max_frames):
    import cv2

    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not isfinite(fps) or fps <= 0:
            raise ValueError("Video must open with a valid frame rate")
        with MediaPipeLandmarker(model) as detector:
            for index in range(max_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                result = detector.detect_rgb(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), include_world=True
                )
                joints = {}
                world = result["world_landmarks"]
                if result["person_count"] == len(world) == 1 and len(world[0]) == 33:
                    for name, i in JOINTS.items():
                        point = result["poses"][0][i]
                        xyz = world[0][i]
                        if (point["visibility"] >= .5 and point["presence"] >= .5
                            and 0 <= point["x_px"] < result["width"]
                            and 0 <= point["y_px"] < result["height"]
                            and all(map(isfinite, xyz))):
                            joints[name] = xyz
                frames.append(dict(frame_index=index, time_s=index / fps,
                                   joints=joints, angles_deg=measure_pose(joints)))
        if not frames:
            raise ValueError("Video has no decodable frames")
        return dict(schema_version=1, coordinate_basis="mediapipe_estimated_world",
                    time_basis="frame_index_over_nominal_fps", fps=fps,
                    frame_limit_reached=len(frames) == max_frames, frames=frames)
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    if args.max_frames <= 0 or args.output.exists():
        parser.error("Use a positive frame limit and a fresh output path")
    result = extract_video(args.video, args.model, args.max_frames)
    payload = json.dumps(result, allow_nan=False)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")


if __name__ == "__main__":
    main()
