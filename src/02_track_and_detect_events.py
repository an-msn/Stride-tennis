import os
import json
import warnings
import argparse
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks, savgol_filter


def suppress_static_balls(frames_dict, fps, grid_px=8.0, min_duration_sec=0.8):
    """
    Rule 1 (Step 5): Any position holding a ball detection for ~1 second or more
    is static scenery (e.g., pale mark on the wall). Drop it before tracking.
    Also rejects non-ball giant boxes (>45px minor axis) and tiny noise (<4px).
    """
    bucket_frames = defaultdict(set)
    total_frames = len(frames_dict)

    for f_str, f_data in frames_dict.items():
        f_idx = int(f_str)
        for b in f_data["balls"]:
            if (
                min(b[2] - b[0], b[3] - b[1]) > 45.0
                or max(b[2] - b[0], b[3] - b[1]) > 65.0
            ):
                continue

            cx = 0.5 * (b[0] + b[2])
            cy = 0.5 * (b[1] + b[3])
            bx, by = int(round(cx / grid_px)), int(round(cy / grid_px))
            bucket_frames[(bx, by)].add(f_idx)

    # Identify static buckets (including immediate 3x3 neighbor buckets)
    min_frames = max(4, int(round(min_duration_sec * fps)))
    static_buckets = set()
    for (bx, by), f_set in bucket_frames.items():
        combined = set(f_set)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                combined |= bucket_frames.get((bx + dx, by + dy), set())
        if len(combined) >= min_frames:
            span = max(combined) - min(combined) + 1
            if len(combined) >= 0.25 * total_frames or (
                span >= min_frames and len(combined) / max(1, span) > 0.35
            ):
                static_buckets.add((bx, by))

    cleaned_balls = {}
    removed_count = 0
    kept_count = 0

    for f_str, f_data in frames_dict.items():
        f_idx = int(f_str)
        valid = []
        for b in f_data["balls"]:
            w, h = b[2] - b[0], b[3] - b[1]
            if min(w, h) > 45.0 or max(w, h) > 65.0 or max(w, h) < 4.0:
                removed_count += 1
                continue
            cx = 0.5 * (b[0] + b[2])
            cy = 0.5 * (b[1] + b[3])
            bx, by = int(round(cx / grid_px)), int(round(cy / grid_px))
            if (bx, by) in static_buckets:
                removed_count += 1
            else:
                valid.append(b)
                kept_count += 1
        cleaned_balls[f_idx] = valid

    return cleaned_balls, removed_count, kept_count


class AsymmetricBallKalmanTracker:
    """
    Step 5: Constant-velocity Kalman filter with downward gravity bias,
    nearest-to-prediction association, asymmetric confidence thresholds,
    and frame-bounds respect.
    """

    def __init__(self, width, height, fps, conf_init=0.50, conf_keep=0.12, max_coast=8):
        self.width = width
        self.height = height
        self.fps = fps
        self.conf_init = conf_init
        self.conf_keep = conf_keep
        self.max_coast = max_coast
        self.max_gate_dist = max(90.0, 1100.0 / max(24.0, fps))
        self.gravity_px = 450.0 / (fps**2)
        self.reset()

    def reset(self):
        self.active = False
        self.coast_count = 0
        self.x = np.zeros((4, 1), dtype=np.float64)  # [cx, cy, vx, vy]
        self.P = np.eye(4, dtype=np.float64) * 500.0
        self.last_d = np.nan

    def predict(self):
        F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0.98, 0], [0, 0, 0, 0.98]],
            dtype=np.float64,
        )
        Q = np.diag([4.0, 4.0, 25.0, 25.0])
        self.x = F @ self.x
        self.x[1, 0] += 0.5 * self.gravity_px
        self.x[3, 0] += self.gravity_px
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, cx, cy):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        R = np.diag([9.0, 9.0])
        z = np.array([[cx], [cy]], dtype=np.float64)
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.coast_count = 0

    @staticmethod
    def compute_blur_compensated_diameter(w, h, vx, vy):
        """
        Upgrade: Motion blur stretches the bounding box along velocity (vx, vy).
        The dimension perpendicular to motion (minor axis) reflects true ball diameter.
        """
        speed = np.hypot(vx, vy)
        if speed > 8.0:
            return float(min(w, h))
        return float(np.sqrt(w * h))

    def step(self, candidates):
        if not self.active:
            strong = [c for c in candidates if c[4] >= self.conf_init]
            if not strong:
                return {
                    "tracked": False,
                    "coasted": False,
                    "cx": np.nan,
                    "cy": np.nan,
                    "d": np.nan,
                    "conf": np.nan,
                }
            best = max(strong, key=lambda c: c[4])
            cx, cy = 0.5 * (best[0] + best[2]), 0.5 * (best[1] + best[3])
            w, h = best[2] - best[0], best[3] - best[1]
            self.active = True
            self.coast_count = 0
            self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
            self.P = np.diag([16.0, 16.0, 200.0, 200.0])
            self.last_d = float(np.sqrt(w * h))
            return {
                "tracked": True,
                "coasted": False,
                "cx": cx,
                "cy": cy,
                "d": self.last_d,
                "conf": best[4],
            }

        pred_x, pred_y = self.predict()

        if not (0 <= pred_x < self.width and 0 <= pred_y < self.height):
            self.reset()
            return {
                "tracked": False,
                "coasted": False,
                "cx": np.nan,
                "cy": np.nan,
                "d": np.nan,
                "conf": np.nan,
            }

        valid_cands = [c for c in candidates if c[4] >= self.conf_keep]
        best_cand = None
        best_dist = float("inf")
        gate = self.max_gate_dist * (1.0 + 0.25 * self.coast_count)

        for c in valid_cands:
            cx, cy = 0.5 * (c[0] + c[2]), 0.5 * (c[1] + c[3])
            dist = np.hypot(cx - pred_x, cy - pred_y)
            if dist < best_dist and dist <= gate:
                best_dist = dist
                best_cand = c

        if best_cand is not None:
            cx, cy = (
                0.5 * (best_cand[0] + best_cand[2]),
                0.5 * (best_cand[1] + best_cand[3]),
            )
            w, h = best_cand[2] - best_cand[0], best_cand[3] - best_cand[1]
            self.update(cx, cy)
            vx, vy = float(self.x[2, 0]), float(self.x[3, 0])
            d = self.compute_blur_compensated_diameter(w, h, vx, vy)
            self.last_d = d
            return {
                "tracked": True,
                "coasted": False,
                "cx": float(self.x[0, 0]),
                "cy": float(self.x[1, 0]),
                "vx": vx,
                "vy": vy,
                "d": d,
                "raw_w": w,
                "raw_h": h,
                "conf": best_cand[4],
            }
        else:
            self.coast_count += 1
            if self.coast_count > self.max_coast:
                self.reset()
                return {
                    "tracked": False,
                    "coasted": False,
                    "cx": np.nan,
                    "cy": np.nan,
                    "d": np.nan,
                    "conf": np.nan,
                }
            return {
                "tracked": True,
                "coasted": True,
                "cx": pred_x,
                "cy": pred_y,
                "vx": float(self.x[2, 0]),
                "vy": float(self.x[3, 0]),
                "d": self.last_d,
                "conf": 0.0,
            }


def parabolic_subframe(signal, idx):
    """
    Step 6: Fit a parabola through the 3 samples around a turning point
    to locate the slope's zero crossing to a fraction of a frame.
    """
    if idx <= 0 or idx >= len(signal) - 1:
        return float(idx)
    s_m1, s_0, s_p1 = float(signal[idx - 1]), float(signal[idx]), float(signal[idx + 1])
    denom = 2.0 * (s_m1 - 2.0 * s_0 + s_p1)
    if abs(denom) < 1e-6:
        return float(idx)
    delta = (s_m1 - s_p1) / denom
    return float(idx) + float(np.clip(delta, -0.5, 0.5))


def interpolate_at_subframe(series, t_star):
    idx0 = int(np.clip(np.floor(t_star), 0, len(series) - 1))
    idx1 = min(len(series) - 1, idx0 + 1)
    alpha = float(np.clip(t_star - idx0, 0.0, 1.0))
    v0, v1 = series[idx0], series[idx1]
    if np.isnan(v0):
        return float(v1)
    if np.isnan(v1):
        return float(v0)
    return float((1.0 - alpha) * v0 + alpha * v1)


def extract_audio_onsets(video_path, fps):
    """
    Active Audio-Visual Fusion Upgrade:
    Extract high-resolution acoustic onset transients (hop_length=256 -> ~11.6ms)
    from the video's audio track to snap visual contact timestamps.
    """
    try:
        import librosa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_aud, sr = librosa.load(video_path, sr=22050, mono=True)
            if len(y_aud) == 0:
                return np.array([])
            onset_times = librosa.onset.onset_detect(
                y=y_aud, sr=sr, hop_length=256, units="time", backtrack=False
            )
        return onset_times * fps
    except Exception:
        return np.array([])


def fuse_visual_and_audio_time(
    t_vis_frame, audio_onsets_frames, max_window_frames=2.5, audio_weight=0.55
):
    """
    If a crisp acoustic transient (racket strike or wall impact) occurs within
    +-max_window_frames of the visual parabolic turning point, fuse the two
    timestamps to sharpen sub-frame contact timing.
    """
    if len(audio_onsets_frames) == 0:
        return t_vis_frame, False
    diffs = np.abs(audio_onsets_frames - t_vis_frame)
    min_idx = int(np.argmin(diffs))
    if diffs[min_idx] <= max_window_frames:
        t_aud = float(audio_onsets_frames[min_idx])
        t_fused = audio_weight * t_aud + (1.0 - audio_weight) * t_vis_frame
        return float(t_fused), True
    return t_vis_frame, False


def smooth_ball_trajectory(track_list, max_interp_gap=4):
    """
    Zero-phase Savitzky-Golay trajectory smoother:
    1. Bridges tiny 1-4 frame coasting dropouts linearly.
    2. Smooths (cx, cy, d) across continuous tracked segments so the AR comet
       trail and proximity signal are free of single-frame bounding-box jitter.
    """
    N = len(track_list)
    cx = np.array([t["cx"] for t in track_list], dtype=np.float64)
    cy = np.array([t["cy"] for t in track_list], dtype=np.float64)
    d = np.array([t["d"] for t in track_list], dtype=np.float64)

    # Bridge tiny gaps (<= max_interp_gap frames) between valid detections
    valid = ~np.isnan(cx) & ~np.isnan(cy)
    if np.sum(valid) < 10:
        return track_list

    valid_idx = np.where(valid)[0]
    for i in range(len(valid_idx) - 1):
        s, e = valid_idx[i], valid_idx[i + 1]
        gap = e - s - 1
        if 0 < gap <= max_interp_gap:
            for k in range(s + 1, e):
                alpha = (k - s) / float(e - s)
                cx[k] = (1.0 - alpha) * cx[s] + alpha * cx[e]
                cy[k] = (1.0 - alpha) * cy[s] + alpha * cy[e]
                d[k] = (1.0 - alpha) * d[s] + alpha * d[e]
                track_list[k]["tracked"] = True
                track_list[k]["coasted"] = True

    # Apply Savitzky-Golay filter (window=7, polyorder=2) over contiguous tracked runs
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


def analyze_proximity_and_events(track_list, wall_corners, fps, video_path=None):
    """
    Step 6 + Active Audio-Visual Fusion:
    Build 1D proximity signal from z-scores of ball vertical position (cy)
    and apparent size (d), find rally period via autocorrelation, detect
    validated stroke maxima & wall impact minima, and fuse with audio transients.
    """
    N = len(track_list)
    cx_arr = np.array([t["cx"] for t in track_list], dtype=np.float64)
    cy_arr = np.array([t["cy"] for t in track_list], dtype=np.float64)
    d_arr = np.array([t["d"] for t in track_list], dtype=np.float64)

    idx_all = np.arange(N)
    valid_mask = ~np.isnan(cy_arr) & ~np.isnan(d_arr)
    if np.sum(valid_mask) < 10:
        raise RuntimeError("Too few tracked ball frames to compute proximity signal.")

    cx_interp = np.interp(idx_all, idx_all[valid_mask], cx_arr[valid_mask])
    cy_interp = np.interp(idx_all, idx_all[valid_mask], cy_arr[valid_mask])
    d_interp = np.interp(idx_all, idx_all[valid_mask], d_arr[valid_mask])

    corr_yd = float(np.corrcoef(cy_interp[valid_mask], d_interp[valid_mask])[0, 1])

    z_y = (cy_interp - np.mean(cy_interp[valid_mask])) / max(
        1e-6, np.std(cy_interp[valid_mask])
    )
    z_d = (d_interp - np.mean(d_interp[valid_mask])) / max(
        1e-6, np.std(d_interp[valid_mask])
    )
    raw_prox = z_y + z_d

    win = max(5, int(round(0.15 * fps)) | 1)
    prox_smooth = savgol_filter(raw_prox, window_length=win, polyorder=2)

    prox_zero = prox_smooth - np.mean(prox_smooth)
    ac = correlate(prox_zero, prox_zero, mode="full")[N - 1 :]
    ac = ac / max(1e-6, ac[0])

    min_lag = int(round(0.6 * fps))
    max_lag = min(N - 1, int(round(3.0 * fps)))
    ac_peaks, _ = find_peaks(ac[min_lag:max_lag], prominence=0.08)
    if len(ac_peaks) > 0:
        rally_period_frames = int(ac_peaks[0] + min_lag)
    else:
        rally_period_frames = int(round(1.5 * fps))
    rally_period_sec = rally_period_frames / fps

    min_sep = max(6, int(round(0.45 * rally_period_frames)))

    stroke_cands, _ = find_peaks(prox_smooth, distance=min_sep, prominence=0.35)
    impact_cands, _ = find_peaks(-prox_smooth, distance=min_sep, prominence=0.35)

    stroke_cands = [int(i) for i in stroke_cands if valid_mask[i]]
    impact_cands = [int(i) for i in impact_cands if valid_mask[i]]

    if len(impact_cands) > 0 and len(stroke_cands) > 0:
        med_wall_d = float(np.median(d_interp[impact_cands]))
        med_stroke_d = float(np.median(d_interp[stroke_cands]))
        size_ceiling = med_wall_d + 0.60 * max(2.0, (med_stroke_d - med_wall_d))
    else:
        size_ceiling = float("inf")

    wall_poly = np.array(wall_corners, dtype=np.float32)
    wall_y_max = float(np.max(wall_poly[:, 1])) + 0.15 * (
        np.max(wall_poly[:, 1]) - np.min(wall_poly[:, 1])
    )

    valid_impacts = []
    for imp_idx in impact_cands:
        if d_interp[imp_idx] <= size_ceiling and cy_interp[imp_idx] <= wall_y_max:
            valid_impacts.append(imp_idx)

    audio_onsets = extract_audio_onsets(video_path, fps) if video_path else np.array([])
    paired_rallies = []
    max_flight_frames = int(round(1.15 * rally_period_frames))
    min_flight_frames = max(3, int(round(0.12 * rally_period_frames)))

    for k, imp_idx in enumerate(valid_impacts):
        prev_imp = valid_impacts[k - 1] if k > 0 else -1
        preceding_strokes = [
            s
            for s in stroke_cands
            if prev_imp < s < imp_idx
            and min_flight_frames <= (imp_idx - s) <= max_flight_frames
        ]
        if not preceding_strokes:
            continue

        s_idx = preceding_strokes[-1]

        # 1. Visual sub-frame timing via 3-point parabolic interpolation
        t_stroke_vis = parabolic_subframe(prox_smooth, s_idx)
        t_impact_vis = parabolic_subframe(-prox_smooth, imp_idx)

        # 2. Active Audio-Visual Sensor Fusion (snap toward acoustic onset within +-2.5 frames)
        t_stroke_star, s_snapped = fuse_visual_and_audio_time(
            t_stroke_vis, audio_onsets, max_window_frames=2.5
        )
        t_impact_star, w_snapped = fuse_visual_and_audio_time(
            t_impact_vis, audio_onsets, max_window_frames=2.5
        )

        if t_impact_star <= t_stroke_star + 2.0:
            t_stroke_star, t_impact_star = t_stroke_vis, t_impact_vis
            s_snapped, w_snapped = False, False

        audio_fused = bool(s_snapped or w_snapped)

        stroke_x = interpolate_at_subframe(cx_interp, t_stroke_star)
        stroke_y = interpolate_at_subframe(cy_interp, t_stroke_star)
        stroke_d = interpolate_at_subframe(d_interp, t_stroke_star)

        impact_x = interpolate_at_subframe(cx_interp, t_impact_star)
        impact_y = interpolate_at_subframe(cy_interp, t_impact_star)
        impact_d = interpolate_at_subframe(d_interp, t_impact_star)

        paired_rallies.append(
            {
                "shot_number": len(paired_rallies) + 1,
                "stroke_frame": int(s_idx),
                "stroke_frame_sub": round(t_stroke_star, 3),
                "stroke_time_sec": round(t_stroke_star / fps, 3),
                "stroke_pos_px": [round(stroke_x, 2), round(stroke_y, 2)],
                "stroke_ball_d_px": round(stroke_d, 2),
                "impact_frame": int(imp_idx),
                "impact_frame_sub": round(t_impact_star, 3),
                "impact_time_sec": round(t_impact_star / fps, 3),
                "impact_pos_px": [round(impact_x, 2), round(impact_y, 2)],
                "impact_ball_d_px": round(impact_d, 2),
                "flight_frames": round(t_impact_star - t_stroke_star, 3),
                "flight_time_sec": round((t_impact_star - t_stroke_star) / fps, 4),
                "audio_verified": audio_fused,
                "audio_snapped_stroke": s_snapped,
                "audio_snapped_impact": w_snapped,
            }
        )

    diagnostics = {
        "corr_y_diameter": round(corr_yd, 3),
        "rally_period_frames": rally_period_frames,
        "rally_period_sec": round(rally_period_sec, 3),
        "audio_onsets_detected": int(len(audio_onsets)),
        "prox_raw": raw_prox.tolist(),
        "prox_smooth": prox_smooth.tolist(),
        "autocorr": ac[:max_lag].tolist(),
    }
    return paired_rallies, diagnostics


def save_diagnostic_plot(track_list, paired_rallies, diag, fps, out_png):
    N = len(track_list)
    t_sec = np.arange(N) / fps
    cy = np.array([t["cy"] for t in track_list])
    prox = np.array(diag["prox_smooth"])
    ac = np.array(diag["autocorr"])
    ac_t = np.arange(len(ac)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)

    axes[0].plot(t_sec, cy, color="#1f77b4", lw=1.5, label="Kalman Tracked Ball Y (px)")
    axes[0].invert_yaxis()
    axes[0].set_title(
        "1. Kalman-Tracked Ball Vertical Trajectory (Top = Near Wall, Bottom = Near Racket)"
    )
    axes[0].set_ylabel("Image Y (px)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        t_sec, prox, color="#2ca02c", lw=1.5, label="Proximity Signal (z_y + z_d)"
    )
    s_times = [r["stroke_time_sec"] for r in paired_rallies]
    s_vals = [prox[r["stroke_frame"]] for r in paired_rallies]
    w_times = [r["impact_time_sec"] for r in paired_rallies]
    w_vals = [prox[r["impact_frame"]] for r in paired_rallies]

    axes[1].scatter(
        s_times,
        s_vals,
        color="red",
        s=55,
        zorder=5,
        label=f"Audio-Visual Fused Strokes ({len(s_times)})",
    )
    axes[1].scatter(
        w_times,
        w_vals,
        color="blue",
        s=55,
        zorder=5,
        label=f"Audio-Visual Fused Wall Impacts ({len(w_times)})",
    )
    axes[1].set_title(
        f"2. 1D Depth Proximity Signal (y-d Correlation = {diag['corr_y_diameter']:+.2f} | "
        f"Measured Rally Period = {diag['rally_period_sec']:.2f}s)"
    )
    axes[1].set_ylabel("Proximity Z-Score")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ac_t, ac, color="#9467bd", lw=1.8, label="Autocorrelation")
    axes[2].axvline(
        diag["rally_period_sec"],
        color="red",
        ls="--",
        lw=1.5,
        label=f"First Rally Period Peak ({diag['rally_period_sec']:.2f}s)",
    )
    axes[2].set_title(
        "3. Proximity Signal Autocorrelation (Dynamic Rally Period Estimation)"
    )
    axes[2].set_xlabel("Lag Time (seconds)")
    axes[2].set_ylabel("Correlation")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/raw_videos/sample.mp4")
    args = parser.parse_args()

    clip_name = os.path.splitext(os.path.basename(args.video))[0]
    quad_file = f"cache/{clip_name}_wall_quad.json"
    det_cache_file = f"cache/{clip_name}_detections.json"
    out_track_file = f"cache/{clip_name}_tracks_and_events.json"
    out_plot_file = f"outputs/{clip_name}_diagnostics.png"

    with open(quad_file, "r") as f:
        quad_data = json.load(f)
    with open(det_cache_file, "r") as f:
        det_data = json.load(f)

    fps = det_data["fps"]
    width = det_data["width"]
    height = det_data["height"]
    total_frames = det_data["total_frames"]

    cleaned_balls, removed_static, kept_dynamic = suppress_static_balls(
        det_data["frames"], fps
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
        track_list, quad_data["wall_corners_px"], fps, video_path=args.video
    )

    save_diagnostic_plot(track_list, paired_rallies, diag, fps, out_plot_file)

    output_payload = {
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
    with open(out_track_file, "w") as f:
        json.dump(output_payload, f, indent=2)

    audio_fused_count = sum(1 for r in paired_rallies if r["audio_verified"])

    print("\n" + "=" * 74)
    print("PASS 2A: BALL TRACKING & AUDIO-VISUAL RALLY EVENT SUMMARY")
    print("=" * 74)
    print(
        f"Static / Invalid Boxes Dropped      : {removed_static} (Kept {kept_dynamic} moving candidates)"
    )
    print(
        f"Kalman Track Coverage               : {tracked_count}/{total_frames} frames ({100 * tracked_count / total_frames:.1f}% | Coasted: {coasted_count})"
    )
    print(
        f"Ball Y vs. Diameter Correlation     : {diag['corr_y_diameter']:+.3f} (README ref: +0.660)"
    )
    print(
        f"Autocorrelation Rally Period        : {diag['rally_period_sec']:.2f} sec ({diag['rally_period_frames']} frames)"
    )
    print(
        f"Audio-Visual Fused Contacts         : {audio_fused_count}/{len(paired_rallies)} shots snapped to acoustic transients"
    )
    print("-" * 74)
    for r in paired_rallies[:12]:
        aud_tag = "[AUD+VIS]" if r["audio_verified"] else "[VIS-ONLY]"
        print(
            f"  Shot #{r['shot_number']:02d} {aud_tag} | Stroke @ {r['stroke_time_sec']:5.2f}s (d={r['stroke_ball_d_px']:4.1f}px) "
            f"--> Wall @ {r['impact_time_sec']:5.2f}s (d={r['impact_ball_d_px']:4.1f}px) | "
            f"Flight: {r['flight_time_sec'] * 1000:4.0f} ms"
        )
    print("=" * 74)
    print(f"[Saved] Track & Events JSON : {out_track_file}")
    print(f"[Saved] Diagnostic Plot PNG : {out_plot_file}")
