# ============================================================
# Causal Relative Kinematics Module
# Shared by extract_dinov2_features.py AND run_bee_vit_lstm.py
#
# Properties:
#   1. Strictly causal: Trailing Exponential Moving Average (EMA)
#      smoothing only (alpha=0.6). Zero future peeking.
#   2. Scale and position invariant:
#      - No absolute x, y coordinates
#      - No absolute pixel dimensions
#      - Speed and acceleration in bee body lengths / second
#      - Features relative to entrance polygon boundary and center
#   3. Identical offline and online frame-gap rule (max_gap=5 frames)
# ============================================================

from __future__ import annotations

import cv2
import numpy as np


KINEMATIC_FEATURE_NAMES = [
    "speed_bee_lengths_sec",
    "accel_bee_lengths_sec2",
    "movement_sin",
    "movement_cos",
    "box_aspect_ratio",
    "signed_dist_to_polygon",
    "heading_to_entrance_cos",
]

NUM_KINEMATIC_FEATURES = len(KINEMATIC_FEATURE_NAMES)  # 7


class CausalKinematicsTracker:
    """
    Maintains causal kinematic state per track_id.
    Can be used online frame-by-frame or offline across sequences.
    """

    def __init__(
        self,
        polygon: np.ndarray,
        video_w: int,
        video_h: int,
        fps: float = 50.0,
        alpha: float = 0.6,
        max_gap_frames: int = 5,
    ):
        self.polygon = polygon.astype(np.int32)
        self.video_w = float(video_w)
        self.video_h = float(video_h)
        self.fps = float(fps)
        self.alpha = float(alpha)
        self.max_gap_frames = int(max_gap_frames)

        # Polygon geometry
        rect = cv2.boundingRect(self.polygon)
        self.poly_diag = float(max(1.0, np.hypot(rect[2], rect[3])))
        # Polygon center
        M = cv2.moments(self.polygon)
        if M["m00"] != 0:
            self.poly_cx = float(M["m10"] / M["m00"])
            self.poly_cy = float(M["m01"] / M["m00"])
        else:
            self.poly_cx = float(rect[0] + rect[2] / 2.0)
            self.poly_cy = float(rect[1] + rect[3] / 2.0)

        # State storage: track_id -> dict
        self._states: dict[int, dict] = {}

    def reset_track(self, track_id: int) -> None:
        if track_id in self._states:
            del self._states[track_id]

    def prune_old_tracks(self, current_frame: int, max_unseen_frames: int = 100) -> None:
        """Prune tracks that have not been updated for > max_unseen_frames."""
        stale = [
            tid
            for tid, st in self._states.items()
            if (current_frame - st["last_frame"]) > max_unseen_frames
        ]
        for tid in stale:
            del self._states[tid]

    def update(
        self,
        track_id: int,
        frame_id: int,
        norm_x: float,
        norm_y: float,
        norm_w: float,
        norm_h: float,
    ) -> np.ndarray:
        """
        Ingests a new detection, updates state, and returns the 7-d relative kinematic feature vector.
        """
        # Pixel coordinates
        cx = float(norm_x * self.video_w)
        cy = float(norm_y * self.video_h)
        bw = float(max(1.0, norm_w * self.video_w))
        bh = float(max(1.0, norm_h * self.video_h))

        bee_length = float(max(5.0, max(bw, bh)))
        aspect = float(bw / bh)

        # Polygon relative calculations
        # Signed distance: positive inside polygon, negative outside
        signed_dist = float(cv2.pointPolygonTest(self.polygon, (cx, cy), True) / self.poly_diag)

        # Vector from bee to polygon center
        to_center_x = self.poly_cx - cx
        to_center_y = self.poly_cy - cy
        center_dist = float(np.hypot(to_center_x, to_center_y))
        if center_dist > 1e-4:
            uc_x = to_center_x / center_dist
            uc_y = to_center_y / center_dist
        else:
            uc_x, uc_y = 0.0, 0.0

        state = self._states.get(track_id)

        if state is None:
            # First observation of this track
            cx_smooth = cx
            cy_smooth = cy
            speed_bee_len = 0.0
            accel_bee_len = 0.0
            sin_dir = 0.0
            cos_dir = 1.0
            heading_cos = 0.0
        else:
            dt_frames = frame_id - state["last_frame"]
            if dt_frames <= 0 or dt_frames > self.max_gap_frames:
                # Discontinuity / gap reset
                cx_smooth = cx
                cy_smooth = cy
                speed_bee_len = 0.0
                accel_bee_len = 0.0
                sin_dir = 0.0
                cos_dir = 1.0
                heading_cos = 0.0
            else:
                # Causal EMA smoothing
                cx_smooth = self.alpha * cx + (1.0 - self.alpha) * state["cx_smooth"]
                cy_smooth = self.alpha * cy + (1.0 - self.alpha) * state["cy_smooth"]

                dx = cx_smooth - state["cx_smooth"]
                dy = cy_smooth - state["cy_smooth"]
                dist = float(np.hypot(dx, dy))
                dt_sec = float(dt_frames / self.fps)

                # Speed in bee lengths / second
                speed_bee_len = float(dist / (bee_length * dt_sec))
                # Acceleration in bee lengths / second^2
                accel_bee_len = float((speed_bee_len - state["last_speed"]) / dt_sec)

                if dist > 1e-4:
                    sin_dir = float(dy / dist)
                    cos_dir = float(dx / dist)
                else:
                    sin_dir = 0.0
                    cos_dir = 1.0

                # Alignment cosine with entrance center direction
                heading_cos = float(cos_dir * uc_x + sin_dir * uc_y)

        # Update state
        self._states[track_id] = {
            "cx_smooth": cx_smooth,
            "cy_smooth": cy_smooth,
            "last_frame": frame_id,
            "last_speed": speed_bee_len,
        }

        # Pack 7-d feature vector
        features = np.array([
            speed_bee_len,
            accel_bee_len,
            sin_dir,
            cos_dir,
            aspect,
            signed_dist,
            heading_cos,
        ], dtype=np.float32)

        return features


def compute_causal_kinematics_offline(
    observations: list[dict],
    polygon: np.ndarray,
    video_w: int,
    video_h: int,
    fps: float = 50.0,
    alpha: float = 0.6,
    max_gap_frames: int = 5,
) -> list[dict]:
    """
    Offline helper to enrich a list of sorted observations for a single track
    using the exact same CausalKinematicsTracker used online.
    """
    tracker = CausalKinematicsTracker(
        polygon=polygon,
        video_w=video_w,
        video_h=video_h,
        fps=fps,
        alpha=alpha,
        max_gap_frames=max_gap_frames,
    )
    enriched = []
    track_id = observations[0]["track_id"] if observations else 0

    for obs in observations:
        k_vec = tracker.update(
            track_id=track_id,
            frame_id=obs["frame_id"],
            norm_x=obs["x"],
            norm_y=obs["y"],
            norm_w=obs["w"],
            norm_h=obs["h"],
        )
        obs_copy = obs.copy()
        obs_copy["kinematics"] = k_vec
        enriched.append(obs_copy)

    return enriched
