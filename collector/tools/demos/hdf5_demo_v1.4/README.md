# HDF5 Data Viewer (hdf5_demo)

Self-contained single-file viewer for hdf5 recordings. All UI text is
English. Shows **stereo video / rendered video / bionic hand / tactile
heatmap / IMU waveform** in one window, with sensor panes appearing only
when the file actually contains that data.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. On Linux a desktop environment (X11/Wayland) is needed.

## Run

```bash
python hdf5_demo.py your_data.h5
```

Without an argument, click "Open h5" after startup to pick a file.

## Layout

| Position | Content |
|---|---|
| Top-left | Stereo camera video (left \| right side by side) |
| Top-right | Pre-rendered video from the h5 (videos/hand_skeleton MP4, or auto-detected by keywords) |
| Bottom-left | Bionic hand: 16x16 tactile matrix mapped onto a hand figure (L \| R) |
| Bottom-right | Tactile heatmap (16x16 pressure matrix, L \| R) |
| Bottom | IMU waveform: accel X/Y/Z (top) and gyro X/Y/Z (bottom), current frame marked by a white line |

Controls: Play/Pause, \|< and >| single-frame step, slider to seek.

## Sensor detection

- Tactile panes are shown only when the tactile datasets exist **and**
  contain non-zero samples (all-zero = sensor never touched = hidden)
- The IMU pane is shown only when `observation/imu` exists
- Hidden panes are removed from the layout (no blank gap); the remaining
  panes expand to fill the window

## Rendered video auto-detection

The top-right pane plays a video **already rendered inside the h5** - no
rendering is done by the viewer. Priority:

1. `<episode>/videos/hand_skeleton` - MP4 bytes, decoded via a temporary
   file + cv2.VideoCapture (sequential read on playback, seek + cache on jumps)
2. Any dataset in `images` whose name contains: `preview, keypoint, annot,
   render, overlay, visual, vis, kp, skeleton, slam`

Extend `RENDERED_KEYWORDS` at the top of the script for other names.

## Datasets read (inside an episode_* group)

```
<episode>/observation/images/stereo_left   (N,800,1280,3) uint8 RGB  [required]
<episode>/observation/images/stereo_right  (N,800,1280,3) uint8 RGB  [required]
<episode>/observation/tactile/left  (N,16,16) float   [optional]
<episode>/observation/tactile/right (N,16,16) float   [optional]
<episode>/observation/imu           (M,6) float [acc xyz, gyr xyz]   [optional]
<episode>/observation/imu_frame_index (M,) int32                     [optional]
```

Missing optional datasets are tolerated: the corresponding pane is hidden
and the rest keeps playing. Playback rate follows `meta/info` attrs
(`capture_fps` preferred, else file `fps`), defaulting to 25 fps.
