"""Shared hand skeleton renderer used by all hand-processing modules."""


FINGERS = {
    "Thumb": ([1, 2, 3, 4], (255, 128, 0)),
    "Index": ([5, 6, 7, 8], (0, 255, 0)),
    "Middle": ([9, 10, 11, 12], (0, 255, 255)),
    "Ring": ([13, 14, 15, 16], (255, 0, 255)),
    "Pinky": ([17, 18, 19, 20], (0, 128, 255)),
}

PALM_EDGES = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]


def hand_style_scale(keypoints):
    """Return a hand-size-aware visual scale for the skeleton overlay.

    Recordings can be 640x480 or 1280x720 while the hand can occupy very
    different portions of the image. Fixed radii therefore look oversized on
    close/far views. Keep a small lower bound so distant hands remain visible.
    """
    import numpy as np

    points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if len(points) < 21:
        return 1.0
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return 1.0
    visible = points[finite]
    span = float(np.max(np.ptp(visible, axis=0)))
    return max(0.45, min(1.0, span / 160.0))


def draw_demo_style(frame, keypoints):
    """Draw the canonical five-colour hand style onto a BGR OpenCV frame.

    ``keypoints`` contains pixel coordinates in MediaPipe's standard 21-point
    order.  Keeping this renderer shared prevents mono and stereo output
    videos from drifting apart visually.
    """
    import cv2

    if len(keypoints) < 21:
        return
    points = [(int(x), int(y)) for x, y in keypoints[:21]]
    scale = hand_style_scale(keypoints)
    palm_width = max(1, int(round(2 * scale)))
    finger_width = max(1, int(round(3 * scale)))
    for start, end in PALM_EDGES:
        cv2.line(frame, points[start], points[end], (200, 200, 200), palm_width,
                 cv2.LINE_AA)
    for finger, (ids, color) in FINGERS.items():
        chain = ids if finger == "Thumb" else [0] + ids
        for i in range(len(chain) - 1):
            cv2.line(frame, points[chain[i]], points[chain[i + 1]], color,
                     finger_width,
                     cv2.LINE_AA)
        for index in ids:
            radius = max(2, int(round((7 if index == ids[-1] else 5) * scale)))
            cv2.circle(frame, points[index], radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, points[index], radius, (30, 30, 30), 1,
                       cv2.LINE_AA)
    wrist_radius = max(3, int(round(9 * scale)))
    cv2.circle(frame, points[0], wrist_radius, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, points[0], wrist_radius, (40, 40, 40),
               max(1, int(round(2 * scale))), cv2.LINE_AA)
