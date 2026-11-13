import os
import json
import math
import argparse
from collections import defaultdict
import numpy as np
import cv2
import librosa
from scipy.signal import savgol_filter, find_peaks
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Module-level reference populated by suppress_static_balls so analyze_proximity_and_events
# can access player pose & racket detections without changing run_pipeline.py's call signature.
_CACHED_DET_FRAMES = {}


def suppress_static_balls(frames_dict, fps, bucket_px=8.0, min_static_sec=0.8):
    """
    Pre-tracking static ball & player-shoe false-positive filter:
    1. Rejects oversize boxes (min(w,h) > 45px) and tiny noise (< 4px).
    2. Masks out low-confidence false positives sitting directly on the player's shoes/ankles.
    3. Drops static background marks appearing in the same spatial bucket across >= 0.8s.
    """
    global _CACHED_DET_FRAMES
    _CACHED_DET_FRAMES = frames_dict

    bucket_Frames = defaultdict(set)
    parsed_frames = {}

    for f_str, f_data in frames_dict.items():
        f_idx = int(f_str)
        balls = f_data.get("balls", [])
        players = f_data.get("players", [])

        # Collect ankle/foot coordinates (kp 15, 16) to suppress white-shoe false positives
        foot_pts = []
        for p in players:
            kpts = p.get("keypoints", [])
            for ankle_idx in (15, 16):
                if ankle_idx < len(kpts) and kpts[ankle_idx][2] > 0.25:
                    foot_pts.append(
                        (float(kpts[ankle_idx][0]), float(kpts[ankle_idx][1]))
                    )

        valid_balls = []
        for b in balls:
            x1, y1, x2, y2, conf = b
            w = x2 - x1
            h = y2 - y1
            if w < 4.0 or h < 4.0:
                continue
            if min(w, h) > 45.0 or max(w, h) > 68.0:
                continue
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect > 4.2:
                continue

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            # Reject low/medium-confidence detections sitting right on the player's white shoes
            on_shoe = False
            for fx, fy in foot_pts:
                if math.hypot(cx - fx, cy - fy) < 38.0 and conf < 0.62:
                    on_shoe = True
                    break
            if on_shoe:
                continue

            bx = int(round(cx / bucket_px))
            by = int(round(cy / bucket_px))
            bucket_Frames[(bx, by)].add(f_idx)
            valid_balls.append([x1, y1, x2, y2, conf, bx, by])
        parsed_frames[f_idx] = valid_balls

    min_frames_thresh = max(5, int(round(min_static_sec * fps)))
    static_buckets = {
        k for k, f_set in bucket_Frames.items() if len(f_set) >= min_frames_thresh
    }

    cleaned_balls = {}
    removed_count = 0
    kept_count = 0

    for f_idx, b_list in parsed_frames.items():
        clean_list = []
        for b in b_list:
            x1, y1, x2, y2, conf, bx, by = b
            is_static = any(
                (bx + dx, by + dy) in static_buckets
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            )
            if is_static:
                removed_count += 1
            else:
                clean_list.append([x1, y1, x2, y2, conf])
                kept_count += 1
        cleaned_balls[f_idx] = clean_list

    return cleaned_balls, removed_count, kept_count


class AsymmetricBallKalmanTracker:
    """
    4-Rule Asymmetric Kalman Filter with Downward Gravity Bias & Tight Spatial Gating:
    State vector: [cx, cy, vx, vy]^T
    """

    def __init__(self, width, height, fps):
        self.width = float(width)
        self.height = float(height)
        self.fps = float(fps)
        self.dt = 1.0 / max(1.0, self.fps)
        self.gravity_px_per_frame2 = 980.0 * (self.dt**2)

        self.active = False
        self.coasted_frames = 0
        self.max_coast = max(6, int(round(0.20 * self.fps)))
        self.high_conf_init = 0.45
        self.low_conf_cont = 0.12

        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.last_d = np.nan

        self.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.985, 0.0],
                [0.0, 0.0, 0.0, 0.985],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        self.R = np.diag([9.0, 9.0])
        self.Q = np.diag([4.0, 4.0, 22.0, 22.0])

    @staticmethod
    def compute_blur_compensated_diameter(x1, y1, x2, y2, vx, vy):
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        speed = math.hypot(vx, vy)
        if speed > 8.0:
            return float(min(w, h))
        return float(math.sqrt(w * h))

    def _reset(self):
        self.active = False
        self.coasted_frames = 0
        self.last_d = np.nan

    def step(self, candidates):
        if not self.active:
            high_cands = [
                c
                for c in candidates
                if c[4] >= self.high_conf_init
                and (0.5 * (c[0] + c[2])) < self.width * 0.85
            ]
            if not high_cands:
                return {
                    "tracked": False,
                    "coasted": False,
                    "cx": np.nan,
                    "cy": np.nan,
                    "vx": 0.0,
                    "vy": 0.0,
                    "d": np.nan,
                    "conf": 0.0,
                }
            best = max(high_cands, key=lambda c: c[4])
            cx = 0.5 * (best[0] + best[2])
            cy = 0.5 * (best[1] + best[3])
            self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
            self.P = np.diag([16.0, 16.0, 150.0, 150.0])
            self.active = True
            self.coasted_frames = 0
            self.last_d = self.compute_blur_compensated_diameter(
                best[0], best[1], best[2], best[3], 0.0, 0.0
            )
            return {
                "tracked": True,
                "coasted": False,
                "cx": round(float(cx), 2),
                "cy": round(float(cy), 2),
                "vx": 0.0,
                "vy": 0.0,
                "d": round(float(self.last_d), 2),
                "conf": round(float(best[4]), 4),
            }

        # Predict next state with downward gravity bias
        self.x = self.F @ self.x
        self.x[3, 0] += self.gravity_px_per_frame2
        self.P = self.F @ self.P @ self.F.T + self.Q

        pred_x = float(self.x[0, 0])
        pred_y = float(self.x[1, 0])

        # Tighter spatial gate prevents jumping to stray court/background glints
        max_gate_px = 110.0 + 24.0 * self.coasted_frames
        valid_cands = []
        for c in candidates:
            if c[4] < self.low_conf_cont:
                continue
            cx = 0.5 * (c[0] + c[2])
            cy = 0.5 * (c[1] + c[3])
            dist = math.hypot(cx - pred_x, cy - pred_y)
            # Require high confidence if a candidate is near the outer half of the gate
            if dist <= max_gate_px and (dist <= 65.0 or c[4] >= 0.35):
                valid_cands.append((dist, cx, cy, c))

        if valid_cands:
            valid_cands.sort(key=lambda item: item[0])
            _, meas_x, meas_y, best_c = valid_cands[0]
            z = np.array([[meas_x], [meas_y]], dtype=np.float64)
            y_res = z - (self.H @ self.x)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + (K @ y_res)
            self.P = (np.eye(4) - K @ self.H) @ self.P

            self.coasted_frames = 0
            vx = float(self.x[2, 0])
            vy = float(self.x[3, 0])
            self.last_d = self.compute_blur_compensated_diameter(
                best_c[0], best_c[1], best_c[2], best_c[3], vx, vy
            )
            return {
                "tracked": True,
                "coasted": False,
                "cx": round(float(self.x[0, 0]), 2),
                "cy": round(float(self.x[1, 0]), 2),
                "vx": round(vx, 2),
                "vy": round(vy, 2),
                "d": round(float(self.last_d), 2),
                "conf": round(float(best_c[4]), 4),
            }

        # Coast on gravity prediction when ball is occluded or motion-blurred
        self.coasted_frames += 1
        out_of_bounds = (
            pred_x < 0.0 or pred_x > self.width or pred_y < 0.0 or pred_y > self.height
        )
        if self.coasted_frames > self.max_coast or out_of_bounds:
            self._reset()
            return {
                "tracked": False,
                "coasted": False,
                "cx": np.nan,
                "cy": np.nan,
                "vx": 0.0,
                "vy": 0.0,
                "d": np.nan,
                "conf": 0.0,
            }

        return {
            "tracked": True,
            "coasted": True,
            "cx": round(pred_x, 2),
            "cy": round(pred_y, 2),
            "vx": round(float(self.x[2, 0]), 2),
            "vy": round(float(self.x[3, 0]), 2),
            "d": round(float(self.last_d), 2),
            "conf": 0.0,
        }


def smooth_ball_trajectory(track_list, max_interp_gap=4):
    """
    Outlier despiker + Zero-phase Savitzky-Golay trajectory smoother:
    1. Removes isolated 1-3 frame teleport spikes (e.g., stray court glints or shoe snaps).
    2. Bridges tiny 1-4 frame coasting dropouts linearly.
    3. Smooths (cx, cy, d) across continuous tracked segments.
    """
    N = len(track_list)
    cx = np.array([t["cx"] for t in track_list], dtype=np.float64)
    cy = np.array([t["cy"] for t in track_list], dtype=np.float64)
    d = np.array([t["d"] for t in track_list], dtype=np.float64)

    # 1. Despike isolated 1-2 frame spatial jumps (> 115px away from neighbors)
    for k in range(2, N - 2):
        if np.isnan(cx[k]) or np.isnan(cy[k]):
            continue
        prev_idx = k - 1 if not np.isnan(cx[k - 1]) else k - 2
        next_idx = k + 1 if not np.isnan(cx[k + 1]) else k + 2
        if not np.isnan(cx[prev_idx]) and not np.isnan(cx[next_idx]):
            mid_x = 0.5 * (cx[prev_idx] + cx[next_idx])
            mid_y = 0.5 * (cy[prev_idx] + cy[next_idx])
            neighbor_span = math.hypot(
                cx[next_idx] - cx[prev_idx], cy[next_idx] - cy[prev_idx]
            )
            dev = math.hypot(cx[k] - mid_x, cy[k] - mid_y)
            if dev > 95.0 and dev > 1.8 * neighbor_span:
                cx[k] = np.nan
                cy[k] = np.nan
                d[k] = np.nan
                track_list[k]["tracked"] = False
                track_list[k]["coasted"] = False

    # 2. Bridge tiny gaps (<= max_interp_gap frames) between valid detections
    valid = ~np.isnan(cx) & ~np.isnan(cy)
    if np.sum(valid) < 10:
        return track_list

    valid_idx = np.where(valid)[0]
    for i in range(len(valid_idx) - 1):
        s, e = valid_idx[i], valid_idx[i + 1]
        gap = e - s - 1
        if 0 < gap <= max_interp_gap:
            dist_gap = math.hypot(cx[e] - cx[s], cy[e] - cy[s])
            if dist_gap <= 65.0 * (gap + 1):
                for k in range(s + 1, e):
                    alpha = (k - s) / float(e - s)
                    cx[k] = (1.0 - alpha) * cx[s] + alpha * cx[e]
                    cy[k] = (1.0 - alpha) * cy[s] + alpha * cy[e]
                    d[k] = (1.0 - alpha) * d[s] + alpha * d[e]
                    track_list[k]["tracked"] = True
                    track_list[k]["coasted"] = True

    # 3. Apply Savitzky-Golay filter (window=7, polyorder=2) over contiguous tracked runs
    is_trk = ~np.isnan(cx) & ~np.isnan(cy)
    run_start = None
    for i in range(N + 1):
        if i < N and is_trk[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= 7:
                    cx[run_start:i] = savgol_filter(
                        cx[run_start:i], window_length=7, polyorder=2
                    )
                    cy[run_start:i] = savgol_filter(
                        cy[run_start:i], window_length=7, polyorder=2
                    )
                    d[run_start:i] = np.clip(
                        savgol_filter(d[run_start:i], window_length=7, polyorder=2),
                        4.0,
                        50.0,
                    )
                run_start = None

    for i in range(N):
        if is_trk[i]:
            track_list[i]["cx"] = round(float(cx[i]), 2)
            track_list[i]["cy"] = round(float(cy[i]), 2)
            track_list[i]["d"] = round(float(d[i]), 2)

    return track_list


def parabolic_subframe(signal_arr, idx):
    """3-point parabolic sub-frame peak/trough refinement."""
    if idx <= 0 or idx >= len(signal_arr) - 1:
        return float(idx)
    y0 = float(signal_arr[idx - 1])
    y1 = float(signal_arr[idx])
    y2 = float(signal_arr[idx + 1])
    denom = 2.0 * (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-6:
        return float(idx)
    delta = (y0 - y2) / denom
    return float(idx + np.clip(delta, -0.5, 0.5))


def extract_audio_onsets(video_path, sr=22050, hop_length=256):
    """Extract acoustic onset timestamps (seconds) if the video contains an audio stream."""
    if video_path is None or not os.path.exists(video_path):
        return np.array([], dtype=np.float64)
    try:
        y, sr_loaded = librosa.load(video_path, sr=sr, mono=True)
        if y is None or len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            return np.array([], dtype=np.float64)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr_loaded, hop_length=hop_length, backtrack=False, units="frames"
        )
        return librosa.frames_to_time(onset_frames, sr=sr_loaded, hop_length=hop_length)
    except Exception:
        return np.array([], dtype=np.float64)


def fuse_visual_and_audio_time(
    vis_frame_float, fps, audio_onsets_sec, max_tol_frames=2.5
):
    """Snap visual sub-frame timestamp toward acoustic transient if within +-2.5 frames."""
    vis_sec = vis_frame_float / float(fps)
    if audio_onsets_sec is None or len(audio_onsets_sec) == 0:
        return vis_sec, False
    diffs = np.abs(audio_onsets_sec - vis_sec)
    min_i = int(np.argmin(diffs))
    if diffs[min_i] <= (max_tol_frames / float(fps)):
        aud_sec = float(audio_onsets_sec[min_i])
        fused_sec = 0.55 * aud_sec + 0.45 * vis_sec
        return fused_sec, True
    return vis_sec, False


def _get_player_wrist_and_racket_info(det_frames, f_idx):
    """Return primary player's wrists, shoulders, hips, box, and detected racket center at f_idx."""
    if not det_frames:
        return None
    f_data = det_frames.get(str(f_idx), {})
    players = f_data.get("players", [])
    if not players:
        return None
    primary = max(
        players, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1])
    )
    kpts = primary["keypoints"]
    wrists = []
    for w_i in (9, 10):
        if w_i < len(kpts) and kpts[w_i][2] > 0.20:
            wrists.append((float(kpts[w_i][0]), float(kpts[w_i][1])))

    sh_y = [float(kpts[i][1]) for i in (5, 6) if i < len(kpts) and kpts[i][2] > 0.20]
    hip_y = [float(kpts[i][1]) for i in (11, 12) if i < len(kpts) and kpts[i][2] > 0.20]
    sh_x = [float(kpts[i][0]) for i in (5, 6) if i < len(kpts) and kpts[i][2] > 0.20]
    hip_x = [float(kpts[i][0]) for i in (11, 12) if i < len(kpts) and kpts[i][2] > 0.20]

    torso_cx = (
        float(np.mean(sh_x + hip_x))
        if (sh_x or hip_x)
        else 0.5 * (primary["box"][0] + primary["box"][2])
    )
    top_sh_y = float(min(sh_y)) if sh_y else float(primary["box"][1])
    bot_hip_y = (
        float(max(hip_y))
        if hip_y
        else float(0.5 * (primary["box"][1] + primary["box"][3]))
    )
    spine_len = max(32.0, bot_hip_y - top_sh_y)

    rackets = f_data.get("rackets", [])
    racket_centers = [
        (0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3])) for r in rackets if r[4] >= 0.20
    ]

    return {
        "box": primary["box"],
        "wrists": wrists,
        "rackets": racket_centers,
        "torso_cx": torso_cx,
        "top_sh_y": top_sh_y,
        "bot_hip_y": bot_hip_y,
        "spine_len": spine_len,
    }


def snap_bounce_to_racket_contact(
    cand_peak_idx, search_end_idx, cx, cy, prox_s, det_frames, wall_corners_px, fps
):
    """
    Fixes the Ground-Bounce vs. Racket-Contact timing bug & rejects phantom mid-air peaks:
    In wall practice, the ball hits the court floor (max cy / proximity peak) ~0.25-0.45s
    BEFORE rising to waist height where the player's racket strikes it.
    This function scans forward from the ground bounce to find the exact frame where:
      1. The ball comes closest to the player's wrists (kp 9, 10) or tennis-racket box.
      2. The ball immediately accelerates upward (dy < 0) toward the wall.
      3. Rejects the candidate completely if the ball is never within physical reach of the player
         (e.g., Shot #06 fence bounce or Shot #14 mid-air flight spike).
    """
    N = len(cx)
    wall_arr = np.array(wall_corners_px, dtype=np.float64)
    wall_top_y = float(np.min(wall_arr[:, 1]))

    # Search from 3 frames before the proximity peak up to 0.45s after (representing post-bounce rise to racket)
    w_start = max(2, cand_peak_idx - 3)
    w_end = min(N - 5, search_end_idx, cand_peak_idx + int(round(0.45 * fps)))
    if w_end <= w_start:
        w_end = min(N - 5, w_start + 4)

    best_f = None
    best_score = -1e9
    best_min_dist = 1e9

    for f in range(w_start, w_end + 1):
        bx, by = float(cx[f]), float(cy[f])
        # Outbound upward launch velocity over the next 4 frames (negative dy = flying up toward wall)
        f_ahead = min(N - 1, f + 4)
        upward_launch = (by - float(cy[f_ahead])) / max(
            1, f_ahead - f
        )  # positive when moving UP

        p_info = _get_player_wrist_and_racket_info(det_frames, f)
        if p_info is None:
            continue

        # Measure shortest distance from ball to hitting wrists or racket head
        strike_targets = list(p_info["wrists"]) + list(p_info["rackets"])
        if strike_targets:
            min_dist = min(math.hypot(bx - tx, by - ty) for tx, ty in strike_targets)
        else:
            min_dist = math.hypot(bx - p_info["torso_cx"], by - p_info["bot_hip_y"])

        spine = p_info["spine_len"]
        norm_dist = min_dist / spine

        # Ball at racket impact should be near waist/chest height (not high above the wall/fence)
        if by < wall_top_y + 25.0:
            continue

        # Score combines closeness to wrist/racket + immediate upward launch toward the wall
        # Ground bounces have large norm_dist (> 1.6x spine) and slower initial upward speed;
        # True racket contact has small norm_dist (0.15x - 1.15x spine) and fast upward launch.
        launch_bonus = 0.08 * np.clip(upward_launch, -10.0, 35.0)
        prox_bonus = 0.15 * float(prox_s[f])
        score = -1.65 * norm_dist + launch_bonus + prox_bonus

        # Extra bonus if the ball is actively transitioning into rapid upward flight right after contact
        if upward_launch > 3.5 and norm_dist < 1.35:
            score += 0.85

        if score > best_score:
            best_score = score
            best_f = f
            best_min_dist = norm_dist

    if best_f is None:
        return None

    # Physical Reach Gate: Reject phantom mid-air peaks where the ball never approached the player/racket
    if best_min_dist > 1.85:
        return None

    return int(best_f)


def analyze_proximity_and_events(
    track_list, wall_corners_px, fps, video_path=None, det_frames=None
):
    """
    1. Smooths trajectory in-place (despiking + Savitzky-Golay).
    2. Computes 1D Depth-Proximity signal P(t) = z_y(t) + z_d(t) and autocorrelation rally period.
    3. Detects player turnarounds, snaps ground bounces forward to TRUE racket/wrist contact,
       filters out phantom mid-air/fence peaks, and pairs each stroke with its true wall impact.
    """
    if det_frames is None:
        det_frames = _CACHED_DET_FRAMES

    # Ensure trajectory is despiked and smoothed even if caller didn't call smooth_ball_trajectory
    smooth_ball_trajectory(track_list)

    N = len(track_list)
    idx_arr = np.arange(N)
    raw_cx = np.array([t["cx"] for t in track_list], dtype=np.float64)
    raw_cy = np.array([t["cy"] for t in track_list], dtype=np.float64)
    raw_d = np.array([t["d"] for t in track_list], dtype=np.float64)

    valid_mask = ~np.isnan(raw_cy) & ~np.isnan(raw_d)
    if np.sum(valid_mask) < 15:
        raise RuntimeError(
            "Insufficient valid ball track points to analyze rally events."
        )

    cx_interp = np.interp(idx_arr, idx_arr[valid_mask], raw_cx[valid_mask])
    cy_interp = np.interp(idx_arr, idx_arr[valid_mask], raw_cy[valid_mask])
    d_interp = np.interp(idx_arr, idx_arr[valid_mask], raw_d[valid_mask])

    win = max(7, int(round(0.18 * fps)) | 1)
    cy_s = savgol_filter(cy_interp, window_length=win, polyorder=2)
    d_s = savgol_filter(d_interp, window_length=win, polyorder=2)

    corr_yd = float(np.corrcoef(cy_interp[valid_mask], d_interp[valid_mask])[0, 1])

    z_y = (cy_s - np.mean(cy_s)) / (np.std(cy_s) + 1e-6)
    z_d = (d_s - np.mean(d_s)) / (np.std(d_s) + 1e-6)
    prox_raw = z_y + z_d
    prox_s = savgol_filter(prox_raw, window_length=win, polyorder=2)

    # Autocorrelation for natural rally period
    p_zero = prox_s - np.mean(prox_s)
    acf_full = np.correlate(p_zero, p_zero, mode="full")
    acf = acf_full[N - 1 :] / (acf_full[N - 1] + 1e-6)

    min_lag = int(round(0.65 * fps))
    max_lag = min(N - 1, int(round(3.0 * fps)))
    acf_peaks, _ = find_peaks(acf[min_lag:max_lag], prominence=0.08)
    if len(acf_peaks) > 0:
        best_lag = int(acf_peaks[0] + min_lag)
    else:
        best_lag = int(np.argmax(acf[min_lag:max_lag]) + min_lag)
    rally_period_sec = float(best_lag / fps)

    # Step 1: Detect initial candidate player-side peaks and wall-side troughs
    min_peak_dist = max(int(round(0.45 * best_lag)), int(round(0.55 * fps)))
    raw_stroke_peaks, _ = find_peaks(prox_s, distance=min_peak_dist, prominence=0.32)
    raw_wall_troughs, _ = find_peaks(-prox_s, distance=min_peak_dist, prominence=0.25)

    # Step 2: Snap each candidate peak from the ground bounce to TRUE racket contact
    # and reject phantom mid-air/fence peaks where the ball isn't near the player
    snapped_strokes = []
    for k, p_idx in enumerate(raw_stroke_peaks):
        p_idx = int(p_idx)
        # Bound forward search before the next wall trough or next peak
        next_troughs = [int(w) for w in raw_wall_troughs if w > p_idx + 4]
        search_limit = next_troughs[0] - 4 if next_troughs else (N - 5)
        if k + 1 < len(raw_stroke_peaks):
            search_limit = min(search_limit, int(raw_stroke_peaks[k + 1]) - 4)

        contact_f = snap_bounce_to_racket_contact(
            p_idx,
            search_limit,
            cx_interp,
            cy_interp,
            prox_s,
            det_frames,
            wall_corners_px,
            fps,
        )
        if contact_f is not None:
            snapped_strokes.append(contact_f)

    # Deduplicate & enforce minimum physical refractory period between consecutive strokes (>= 0.92s)
    min_stroke_gap_frames = max(int(round(0.62 * best_lag)), int(round(0.92 * fps)))
    filtered_strokes = []
    for s_f in snapped_strokes:
        if not filtered_strokes:
            filtered_strokes.append(s_f)
        else:
            prev_f = filtered_strokes[-1]
            if s_f - prev_f < min_stroke_gap_frames:
                # Keep the candidate where the ball is closer to the player's wrist/racket
                p_prev = _get_player_wrist_and_racket_info(det_frames, prev_f)
                p_curr = _get_player_wrist_and_racket_info(det_frames, s_f)
                d_prev = (
                    min(
                        [
                            math.hypot(cx_interp[prev_f] - tx, cy_interp[prev_f] - ty)
                            for tx, ty in (p_prev["wrists"] + p_prev["rackets"])
                        ]
                        or [999.0]
                    )
                    if p_prev
                    else 999.0
                )
                d_curr = (
                    min(
                        [
                            math.hypot(cx_interp[s_f] - tx, cy_interp[s_f] - ty)
                            for tx, ty in (p_curr["wrists"] + p_curr["rackets"])
                        ]
                        or [999.0]
                    )
                    if p_curr
                    else 999.0
                )
                if d_curr < d_prev:
                    filtered_strokes[-1] = s_f
            else:
                filtered_strokes.append(s_f)

    audio_onsets_sec = extract_audio_onsets(video_path)

    wall_arr = np.array(wall_corners_px, dtype=np.float64)
    wall_cy_min = float(np.min(wall_arr[:, 1])) - 80.0
    wall_cy_max = float(np.max(wall_arr[:, 1])) + 95.0

    min_flight_f = max(5, int(round(0.16 * fps)))
    max_flight_f = int(round(1.15 * fps))

    paired_rallies = []
    audio_snapped_count = 0

    # Step 3: Pair each validated racket contact with the true wall bounce trough that follows it
    for idx_s, s_idx in enumerate(filtered_strokes):
        next_s_idx = (
            filtered_strokes[idx_s + 1]
            if (idx_s + 1 < len(filtered_strokes))
            else (N - 1)
        )
        win_start = s_idx + min_flight_f
        win_end = min(next_s_idx - 4, s_idx + max_flight_f, N - 2)
        if win_end <= win_start:
            continue

        # Look for trough candidates in [win_start, win_end]; fallback to minimum cy_s (highest point on wall)
        cand_troughs = [int(w) for w in raw_wall_troughs if win_start <= w <= win_end]
        if cand_troughs:
            # Pick the deepest wall trough (lowest cy_s + proximity)
            w_idx = min(cand_troughs, key=lambda w: cy_s[w] + 15.0 * prox_s[w])
        else:
            w_idx = int(win_start + np.argmin(cy_s[win_start : win_end + 1]))

        # Verify that the ball actually reached the upper wall zone at w_idx
        w_cy = float(cy_interp[w_idx])
        s_cy = float(cy_interp[s_idx])
        if not (wall_cy_min <= w_cy <= wall_cy_max):
            continue
        if (s_cy - w_cy) < 35.0:
            # Ball must travel meaningfully upward from the racket to the wall
            continue

        s_sub = parabolic_subframe(prox_s, s_idx)
        w_sub = parabolic_subframe(-cy_s, w_idx)

        s_time, s_aud = fuse_visual_and_audio_time(s_sub, fps, audio_onsets_sec)
        w_time, w_aud = fuse_visual_and_audio_time(w_sub, fps, audio_onsets_sec)
        if w_time <= s_time + 0.14:
            continue

        if s_aud or w_aud:
            audio_snapped_count += 1

        # Measure robust ball pixel diameters around racket contact and wall bounce
        s_d_slice = raw_d[max(0, s_idx - 2) : min(N, s_idx + 3)]
        w_d_slice = raw_d[max(0, w_idx - 2) : min(N, w_idx + 3)]
        s_d_val = (
            float(np.nanmedian(s_d_slice))
            if np.any(~np.isnan(s_d_slice))
            else float(d_interp[s_idx])
        )
        w_d_val = (
            float(np.nanmedian(w_d_slice))
            if np.any(~np.isnan(w_d_slice))
            else float(d_interp[w_idx])
        )

        paired_rallies.append(
            {
                "shot_number": len(paired_rallies) + 1,
                "stroke_frame": int(s_idx),
                "stroke_subframe": round(float(s_sub), 3),
                "stroke_time_sec": round(float(s_time), 4),
                "stroke_pos_px": [
                    round(float(cx_interp[s_idx]), 2),
                    round(float(cy_interp[s_idx]), 2),
                ],
                "stroke_ball_d_px": round(s_d_val, 2),
                "stroke_d_px": round(s_d_val, 2),
                "impact_frame": int(w_idx),
                "impact_subframe": round(float(w_sub), 3),
                "impact_time_sec": round(float(w_time), 4),
                "impact_pos_px": [
                    round(float(cx_interp[w_idx]), 2),
                    round(float(cy_interp[w_idx]), 2),
                ],
                "impact_ball_d_px": round(w_d_val, 2),
                "impact_d_px": round(w_d_val, 2),
                "flight_time_sec": round(float(w_time - s_time), 4),
                "audio_verified": bool(s_aud or w_aud),
            }
        )

    diagnostics = {
        "corr_y_diameter": round(corr_yd, 4),
        "rally_period_frames": int(best_lag),
        "rally_period_sec": round(rally_period_sec, 3),
        "audio_onsets_detected": int(len(audio_onsets_sec)),
        "audio_snapped_shots": int(audio_snapped_count),
        "proximity_signal": [round(float(x), 4) for x in prox_s],
        "cy_smoothed": [round(float(x), 2) for x in cy_s],
        "d_smoothed": [round(float(x), 2) for x in d_s],
        "acf": [round(float(x), 4) for x in acf[:max_lag]],
    }
    return paired_rallies, diagnostics


def save_diagnostic_plot(track_list, paired_rallies, diag, fps, out_png):
    """Render 3-panel diagnostic figure: Vertical Trajectory, Proximity Signal, and Autocorrelation."""
    N = len(track_list)
    t_sec = np.arange(N) / float(fps)
    cy_s = np.array(diag["cy_smoothed"])
    prox_s = np.array(diag["proximity_signal"])
    acf = np.array(diag["acf"])
    acf_t = np.arange(len(acf)) / float(fps)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), constrained_layout=True)

    axes[0].plot(t_sec, cy_s, color="#1f77b4", lw=1.5, label="Smoothed Ball Y (px)")
    axes[0].invert_yaxis()
    axes[0].set_title(
        f"1. Ball Vertical Trajectory (y-d Corr = {diag['corr_y_diameter']:+.3f}) | Inverted Y: Top = Wall, Bottom = Player",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_ylabel("Image Y (px)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        t_sec, prox_s, color="#2ca02c", lw=1.6, label="Proximity Signal (z_y + z_d)"
    )
    for r in paired_rallies:
        st_f = r["stroke_frame"]
        im_f = r["impact_frame"]
        axes[1].plot(st_f / fps, prox_s[st_f], "ro", ms=7)
        axes[1].plot(im_f / fps, prox_s[im_f], "bs", ms=6)
    axes[1].plot([], [], "ro", label="Racket Stroke (Snapped to Wrist/Racket)")
    axes[1].plot([], [], "bs", label="Wall Impact (Trough)")
    axes[1].set_title(
        f"2. Combined Depth-Proximity Signal & Validated Rally Pairs (Total = {len(paired_rallies)} shots)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_ylabel("Z-Score Sum")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(acf_t, acf, color="#9467bd", lw=1.8, label="Autocorrelation")
    axes[2].axvline(
        diag["rally_period_sec"],
        color="#d62728",
        ls="--",
        lw=2.0,
        label=f"Rally Period = {diag['rally_period_sec']:.2f}s ({diag['rally_period_frames']} frames)",
    )
    axes[2].set_title(
        "3. Autocorrelation Rally Period Estimation", fontsize=11, fontweight="bold"
    )
    axes[2].set_xlabel("Lag Time (seconds)")
    axes[2].set_ylabel("Normalized ACF")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/raw_videos/sample.mp4")
    args = parser.parse_args()

    clip_name = os.path.splitext(os.path.basename(args.video))[0]
    det_cache_file = f"cache/{clip_name}_detections.json"
    quad_file = f"cache/{clip_name}_wall_quad.json"
    out_track_file = f"cache/{clip_name}_tracks_and_events.json"
    out_plot_file = f"outputs/{clip_name}_diagnostics.png"

    with open(det_cache_file, "r") as f:
        det_payload = json.load(f)
    with open(quad_file, "r") as f:
        quad_payload = json.load(f)

    fps = det_payload["fps"]
    width = det_payload["width"]
    height = det_payload["height"]
    total_frames = det_payload["total_frames"]

    cleaned_balls, removed_static, kept_dynamic = suppress_static_balls(
        det_payload["frames"], fps
    )

    tracker = AsymmetricBallKalmanTracker(width, height, fps)
    track_list = []
    for idx in range(total_frames):
        state = tracker.step(cleaned_balls.get(idx, []))
        state["frame_idx"] = idx
        track_list.append(state)

    track_list = smooth_ball_trajectory(track_list)

    tracked_count = sum(1 for t in track_list if t["tracked"])
    coasted_count = sum(1 for t in track_list if t["coasted"])

    paired_rallies, diag = analyze_proximity_and_events(
        track_list,
        quad_payload["wall_corners_px"],
        fps,
        video_path=args.video,
        det_frames=det_payload["frames"],
    )
    save_diagnostic_plot(track_list, paired_rallies, diag, fps, out_plot_file)

    print("\n" + "=" * 74)
    print("PASS 2A: BALL TRACKING & RACKET-SNAPPED RALLY EVENT SUMMARY")
    print("=" * 74)
    print(
        f"Static / Invalid Boxes Dropped     : {removed_static} (Kept {kept_dynamic} moving candidates)"
    )
    print(
        f"Kalman Track Coverage              : {tracked_count}/{total_frames} frames ({100.0 * tracked_count / max(1, total_frames):.1f}% | Coasted: {coasted_count})"
    )
    print(
        f"Ball Y vs. Diameter Correlation    : {diag['corr_y_diameter']:+.3f} (README ref: +0.660)"
    )
    print(
        f"Autocorrelation Rally Period       : {diag['rally_period_sec']:.2f} sec ({diag['rally_period_frames']} frames)"
    )
    print(
        f"Audio-Visual Fused Contacts        : {diag['audio_snapped_shots']}/{len(paired_rallies)} shots snapped to acoustic transients"
    )
    print("-" * 74)
    for r in paired_rallies:
        tag = "AUD+VIS" if r["audio_verified"] else "VIS-ONLY"
        print(
            f"  Shot #{r['shot_number']:02d} [{tag}] | Stroke @ {r['stroke_time_sec']:5.2f}s (d={r['stroke_d_px']:4.1f}px) "
            f"--> Wall @ {r['impact_time_sec']:5.2f}s (d={r['impact_d_px']:4.1f}px) | Flight: {r['flight_time_sec'] * 1000:4.0f} ms"
        )
    print("=" * 74)

    out_payload = {
        "clip_name": clip_name,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "static_balls_removed": removed_static,
        "tracked_frames": tracked_count,
        "coasted_frames": coasted_count,
        "diagnostics": diag,
        "rallies": paired_rallies,
        "ball_track": track_list,
    }
    os.makedirs(os.path.dirname(out_track_file), exist_ok=True)
    with open(out_track_file, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[Saved] Track & Events JSON : {out_track_file}")
    print(f"[Saved] Diagnostic Plot PNG : {out_plot_file}")
