# Bundled black-glove processing

This directory contains the code and default model used by the
`black_glove_hand` workflow module. It is intentionally inside `Data
Acquisition` so the worker does not import the legacy repository-level
black-glove demo.

Contents:

- `glove_detector.py`: YOLO-World box detection, RTMPose hand keypoints and
  temporal stabilization.
- `glove_package/`: local tracker and YOLO-World wrapper.
- `d435_hands_demo.py`: local RGB-D alignment/lifting helper used when a
  compatible depth stream is available.
- `d435_tracking.py`: local, NumPy-only depth jump gate and alpha-beta slot
  predictor. It is the production workflow's extracted v1.0 tracking logic;
  it does not import the standalone v1.0 project. The workflow adapter also
  keeps a slot-level depth-centre EMA, per-joint gate recovery, and label/2D
  geometry fallback when a detector track id is unavailable.
- `hand_landmarker.task`: MediaPipe model used by the RGB-D/Hand 3D helper.
- `weights/yolov8m-worldv2.pt`: default black-glove detector weight.

Before publishing a public repository, verify the redistribution terms for the
bundled model files. If the model terms do not allow redistribution, keep the
code layout and replace the files with a documented download step.

The Python dependencies are declared in the parent `Data Acquisition`
`requirements.txt`. RTMPose model assets may be downloaded by `rtmlib` on
first use and cached by the Python environment; this is a package/runtime
dependency, not a dependency on another repository directory.

## Output semantics

The `hand_3d/*.parquet` artifact keeps both preview continuity and data
quality explicit for each slot:

- `real`: current YOLO/RTMPose + depth observation accepted;
- `propagated`: bounded D435 alpha-beta prediction, for preview continuity;
- `absent`: no observation and no bounded prediction.

The RGB renderer hides detector rectangles, but the detector still uses its
internal ROI. `propagated` frames are retained in this raw artifact for a
continuous preview; they are excluded from the default training export.

The raw artifact keeps propagated rows so the 3D viewer can remain continuous.
LeRobot/HDF5 training export filters `state != "real"` by default, so predicted
points are not silently used as labels.

## Future detector fine-tuning augmentation

The current repository has no detector-training command, so augmentation is
not applied to inference code. When fine-tuning a glove-specific detector or
21-point pose model, build the training set from whole video sequences and add
motion blur, random frame exposure, black-object/background swaps, partial
occlusion, crop/scale jitter, and hand-edge truncation. Split validation by
sequence/person, not by random frames, and keep a clean unaugmented test set.
