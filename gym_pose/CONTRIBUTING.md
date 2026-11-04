# Temporal contributor handoff

## Where the project stands

The local web app analyses uploaded squat videos. It estimates 2D joints with
YOLO Pose, counts reps with angle and stability rules, and shows annotated
playback, angle plots and downloads. Optional counting cross-checks use another
YOLO model and MediaPipe when needed. Experimental Teng back and heel review
uses separate logistic regression classifiers. These features have limited
accuracy evidence on new recordings.

The 3D extension is a command-line prototype. The spatial exporter uses
MediaPipe to estimate named 3D joints. The comparison module measures knee and
hip angles for a supplied pair of frames. At commit ff6b435, 39 focused tests
and a mocked export had passed. A real-video export check was pending.

## What we are building

The project aims to compare a user squat with a selected reference squat at
corresponding movement phases. The current division of work is:

- Chaitanya handles spatial work: 3D joint extraction, frame measurements and geometry.
- The temporal contributor handles pairing frames across the two recordings.

Later work, including synchronized playback, a rotatable skeleton view and
possible ST-GCN experiments, is open for discussion. Ownership of those items
will be decided together later.

## Interface for the temporal work

- Input is two version 1 exports from
  [spatial_export.py](gym_analyser/spatial_export.py). See the
  [root README](../README.md) for the command.
- Each entry in `frames` has `frame_index`, `time_s`, `joints` and
  `angles_deg`. Indices are zero based and timestamps use nominal FPS.
  Recordings may have different frame rates and durations.
- Missing joints and angles are omitted. An empty joints map is unavailable
  tracking. Keep its frame index. The export's frame limit can cut off a
  repetition, so confirm the selected windows contain the intended movement.
- Start with one explicitly selected repetition in each clip. Pair comparable
  points in the descent, bottom and ascent. Keep source timestamps so we can
  still report differences in pace.
- Proposed handoff is an ordered list of
  `{observed_frame_index, reference_frame_index}` pairs. Allow repeated
  indices if an alignment needs them and keep both index sequences
  nondecreasing. Report skipped ranges and coverage alongside the pairs.
- For each pair, pass the two `joints` maps to
  [compare_poses()](gym_analyser/spatial_comparison.py). It returns observed
  and reference interior angles and observed-minus-reference differences in
  degrees. A null metric means one of the two poses lacks a usable angle.
- Check identical sequences, different speeds or frame rates, missing
  observations, and a clip with an incomplete repetition. Small synthetic
  exports are enough to begin developing the temporal logic.

## How to contribute

1. Fork the public upstream repository and clone your fork.
2. Start a feature branch such as `temporal/phase-alignment` from upstream
   `main`. Keep the first PR scoped to alignment and its integration contract.
3. Push to your fork and open a pull request targeting upstream `main`.
   Include the input/output example, matching method, tests and known limits.
4. The maintainer reviews, runs checks and merges. Coordinate interface
   changes in the PR before changing shared spatial or web modules.

Keep personal recordings, model weights, datasets, local outputs and
credentials out of Git. Use small synthetic fixtures for review.
See [GitHub's fork pull request guide](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request-from-a-fork).
