# Public examples

These are two short examples derived from the server sessions:

- `D435---glove_sensor_AI`: RGB, raw 12-bit depth and left/right glove arrays.
- `D435---bare_hand_3d_keypoints_AI`: RGB, raw 12-bit depth and the source frame table.

Each example contains the first 90 frames only. RGB is downscaled and blurred for publication; audio is removed. The depth MP4 remains a single-plane `gray12le` HEVC stream, so it is raw depth code data rather than a pseudo-color rendering. The episode metadata uses relative timestamps and public sample device names.
