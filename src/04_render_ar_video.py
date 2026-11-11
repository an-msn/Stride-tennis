import os
import json
import math
import argparse
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm


SKELETON_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def is_valid_num(val):
    return val is not None and not (
        isinstance(val, float) and (math.isnan(val) or math.isinf(val))
    )


def draw_translucent_box(
    img, x1, y1, x2, y2, color=(20, 24, 30), alpha=0.72, border_color=None
):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub
    if border_color is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)


def draw_wall_target_ar(frame, wall_corners_scaled, completed_impacts, flash_alpha=0.0):
    """Draw the 4-corner perspective target grid on the wall and pin historical shot impacts."""
    pts = np.array(wall_corners_scaled, dtype=np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()

    fill_color = (80, 230, 120) if flash_alpha > 0.1 else (200, 180, 50)
    cv2.fillPoly(overlay, [pts], fill_color)
    blend = 0.10 + 0.22 * flash_alpha
    cv2.addWeighted(overlay, blend, frame, 1.0 - blend, 0, frame)

    border_col = (100, 255, 150) if flash_alpha > 0.1 else (255, 220, 80)
    cv2.polylines(
        frame, [pts], isClosed=True, color=border_col, thickness=2, lineType=cv2.LINE_AA
    )

    c = np.array(wall_corners_scaled, dtype=np.float32)  # TL, TR, BR, BL
    for frac in (1.0 / 3.0, 2.0 / 3.0):
        top_pt = (1.0 - frac) * c[0] + frac * c[1]
        bot_pt = (1.0 - frac) * c[3] + frac * c[2]
        cv2.line(
            frame,
            tuple(top_pt.astype(int)),
            tuple(bot_pt.astype(int)),
            (200, 200, 120),
            1,
            cv2.LINE_AA,
        )

        left_pt = (1.0 - frac) * c[0] + frac * c[3]
        right_pt = (1.0 - frac) * c[1] + frac * c[2]
        cv2.line(
            frame,
            tuple(left_pt.astype(int)),
            tuple(right_pt.astype(int)),
            (200, 200, 120),
            1,
            cv2.LINE_AA,
        )

    for shot in completed_impacts:
        ix, iy = int(shot["impact_pos_scaled"][0]), int(shot["impact_pos_scaled"][1])
        is_fh = shot["stroke_type"] == "Forehand"
        col = (
            (255, 220, 0) if is_fh else (0, 165, 255)
        )  # Cyan = Forehand, Orange = Backhand
        cv2.circle(frame, (ix, iy), 7, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (ix, iy), 9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(shot["shot_number"]),
            (ix + 10, iy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_player_skeleton_smoothed(frame, players_frame, scale, ema_state, alpha=0.45):
    """Render primary player's pose skeleton using Exponential Moving Average (EMA) smoothing."""
    if not players_frame:
        return
    primary = max(
        players_frame,
        key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]),
    )
    raw_kpts = np.array(primary["keypoints"], dtype=np.float64)

    if ema_state["kpts"] is None:
        ema_state["kpts"] = raw_kpts.copy()
    else:
        for idx in range(len(raw_kpts)):
            if raw_kpts[idx, 2] > 0.25:
                if ema_state["kpts"][idx, 2] > 0.25:
                    ema_state["kpts"][idx, :2] = (
                        alpha * raw_kpts[idx, :2]
                        + (1.0 - alpha) * ema_state["kpts"][idx, :2]
                    )
                else:
                    ema_state["kpts"][idx, :2] = raw_kpts[idx, :2]
                ema_state["kpts"][idx, 2] = raw_kpts[idx, 2]

    kpts = ema_state["kpts"]

    for i, j in SKELETON_EDGES:
        if kpts[i][2] > 0.30 and kpts[j][2] > 0.30:
            p1 = (int(round(kpts[i][0] * scale)), int(round(kpts[i][1] * scale)))
            p2 = (int(round(kpts[j][0] * scale)), int(round(kpts[j][1] * scale)))
            cv2.line(frame, p1, p2, (120, 255, 160), 2, cv2.LINE_AA)

    for idx in range(5, 17):
        if kpts[idx][2] > 0.30:
            px, py = int(round(kpts[idx][0] * scale)), int(round(kpts[idx][1] * scale))
            cv2.circle(frame, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)

    if kpts[5][2] > 0.25 and kpts[6][2] > 0.25:
        mx = int(round(0.5 * (kpts[5][0] + kpts[6][0]) * scale))
        my = int(round(0.5 * (kpts[5][1] + kpts[6][1]) * scale))
        cv2.circle(frame, (mx, my), 6, (0, 255, 255), -1, cv2.LINE_AA)


def draw_hud_dashboard(frame, completed_shots, active_shot, all_shots, summary, calib):
    """Render Top-Left Telemetry HUD, Top-Right 2D Wall Target Mini-Map, and Bottom-Right Speed Chart."""
    h, w = frame.shape[:2]

    # --- 1. TOP-LEFT TELEMETRY PANEL ---
    px1, py1, px2, py2 = 24, 24, 465, 255
    draw_translucent_box(
        frame,
        px1,
        py1,
        px2,
        py2,
        color=(18, 22, 28),
        alpha=0.78,
        border_color=(90, 110, 130),
    )

    cv2.putText(
        frame,
        "STRIDE TENNIS AI | MONOCULAR CV",
        (px1 + 16, py1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 230, 255),
        2,
        cv2.LINE_AA,
    )

    total_now = len(completed_shots)
    fh_now = sum(1 for s in completed_shots if s["stroke_type"] == "Forehand")
    bh_now = total_now - fh_now

    cv2.putText(
        frame,
        f"STROKES : {total_now:02d}  (FH: {fh_now} | BH: {bh_now})",
        (px1 + 16, py1 + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    latest = (
        active_shot
        if active_shot is not None
        else (completed_shots[-1] if completed_shots else None)
    )
    if latest is not None:
        st_type = latest["stroke_type"].upper()
        st_col = (255, 220, 0) if st_type == "FOREHAND" else (0, 165, 255)
        cv2.putText(
            frame,
            f"LAST SHOT: {st_type} @ {latest['speed_kmh']:.1f} km/h",
            (px1 + 16, py1 + 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            st_col,
            2,
            cv2.LINE_AA,
        )

        knee_val = latest.get("knee_angle_deg")
        knee_txt = f"{knee_val:.0f} deg" if is_valid_num(knee_val) else "N/A"
        spc_val = latest.get("rel_spacing_ratio")
        spc_txt = f"{spc_val:.2f}x" if is_valid_num(spc_val) else "N/A"
        cv2.putText(
            frame,
            f"LOAD KNEE: {knee_txt} | SPACING: {spc_txt}",
            (px1 + 16, py1 + 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"COACH CUE: {latest['posture_note']}",
            (px1 + 16, py1 + 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (120, 255, 150),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "WAITING FOR FIRST RALLY STROKE...",
            (px1 + 16, py1 + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"CLIP MEDIAN: {summary['median_speed_kmh']:.1f} km/h | MAX: {summary['max_speed_kmh']:.1f} km/h",
        (px1 + 16, py1 + 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 190, 200),
        1,
        cv2.LINE_AA,
    )

    # --- 2. TOP-RIGHT 2D WALL PLACEMENT MINI-MAP ---
    map_size = 210
    mx1 = w - map_size - 28
    my1 = 24
    mx2 = mx1 + map_size
    my2 = my1 + map_size + 36
    draw_translucent_box(
        frame,
        mx1,
        my1,
        mx2,
        my2,
        color=(18, 22, 28),
        alpha=0.78,
        border_color=(90, 110, 130),
    )
    cv2.putText(
        frame,
        "2D WALL TARGET MAP",
        (mx1 + 18, my1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 220, 80),
        1,
        cv2.LINE_AA,
    )

    pad = 22
    sq_x1, sq_y1 = mx1 + pad, my1 + 36 + pad // 2
    sq_w = map_size - 2 * pad
    sq_h = map_size - 2 * pad
    cv2.rectangle(
        frame,
        (sq_x1, sq_y1),
        (sq_x1 + sq_w, sq_y1 + sq_h),
        (220, 200, 80),
        2,
        cv2.LINE_AA,
    )

    for frac in (1.0 / 3.0, 2.0 / 3.0):
        gx = int(sq_x1 + frac * sq_w)
        gy = int(sq_y1 + frac * sq_h)
        cv2.line(frame, (gx, sq_y1), (gx, sq_y1 + sq_h), (70, 80, 90), 1)
        cv2.line(frame, (sq_x1, gy), (sq_x1 + sq_w, gy), (70, 80, 90), 1)

    for s in completed_shots:
        u_norm, v_norm = s["wall_target_uv"]
        u_clamped = float(np.clip(u_norm, -0.12, 1.12))
        v_clamped = float(np.clip(v_norm, -0.12, 1.12))
        px = int(sq_x1 + u_clamped * sq_w)
        py = int(sq_y1 + v_clamped * sq_h)
        col = (255, 220, 0) if s["stroke_type"] == "Forehand" else (0, 165, 255)
        cv2.circle(frame, (px, py), 6, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)

    # --- 3. BOTTOM-RIGHT SHOT SPEED BAR CHART ---
    if len(all_shots) > 0:
        cw = min(380, max(240, len(all_shots) * 28 + 40))
        ch = 135
        cx1 = w - cw - 28
        cy1 = h - ch - 24
        cx2, cy2 = cx1 + cw, cy1 + ch
        draw_translucent_box(
            frame,
            cx1,
            cy1,
            cx2,
            cy2,
            color=(18, 22, 28),
            alpha=0.78,
            border_color=(90, 110, 130),
        )
        cv2.putText(
            frame,
            "SHOT SPEED (KM/H)  [CYAN=FH | ORG=BH]",
            (cx1 + 14, cy1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )

        max_spd = max(60.0, float(summary.get("max_speed_kmh", 60.0)) * 1.15)
        bar_area_w = cw - 28
        bar_max_h = ch - 52
        base_y = cy2 - 20
        step_x = bar_area_w / max(1, len(all_shots))
        bar_w = max(8, int(step_x * 0.65))

        done_ids = {s["shot_number"] for s in completed_shots}
        if active_shot is not None:
            done_ids.add(active_shot["shot_number"])

        for idx_s, s in enumerate(all_shots):
            bx1 = int(cx1 + 14 + idx_s * step_x)
            bx2 = bx1 + bar_w
            bh_px = int(np.clip((s["speed_kmh"] / max_spd) * bar_max_h, 4, bar_max_h))
            by1 = base_y - bh_px

            if s["shot_number"] in done_ids:
                col = (255, 220, 0) if s["stroke_type"] == "Forehand" else (0, 165, 255)
                cv2.rectangle(frame, (bx1, by1), (bx2, base_y), col, -1)
                cv2.putText(
                    frame,
                    f"{s['speed_kmh']:.0f}",
                    (bx1 - 1, by1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(frame, (bx1, base_y - 4), (bx2, base_y), (60, 70, 80), -1)

            cv2.putText(
                frame,
                str(s["shot_number"]),
                (bx1 + 2, base_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (160, 170, 180),
                1,
                cv2.LINE_AA,
            )


def create_smooth_video_writer(out_mp4, fps, width, height):
    """Prefer H.264 (avc1) for butter-smooth 60 FPS playback; fallback to mp4v."""
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_mp4, fourcc, fps, (width, height))
        if writer.isOpened():
            print(
                f"[Pass 3] Initialized VideoWriter with codec: '{codec}' @ {fps:.2f} FPS"
            )
            return writer
    raise RuntimeError(f"Could not open VideoWriter for {out_mp4}")


def render_annotated_video(
    video_path, det_payload, track_payload, quad_payload, analysis_payload, out_mp4
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_h = max(1080, orig_h) if orig_h < 1080 else orig_h
    scale = target_h / float(orig_h)
    target_w = int(round(orig_w * scale))
    target_w += target_w % 2
    target_h += target_h % 2

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    writer = create_smooth_video_writer(out_mp4, fps, target_w, target_h)

    wall_corners_scaled = [
        [pt[0] * scale, pt[1] * scale] for pt in quad_payload["wall_corners_px"]
    ]
    ball_track = track_payload["ball_track"]
    det_frames = det_payload["frames"]
    shots = analysis_payload["shots"]
    summary = analysis_payload["summary"]
    calib = analysis_payload["calibration"]

    for s in shots:
        s["stroke_pos_scaled"] = [
            s["stroke_pos_px"][0] * scale,
            s["stroke_pos_px"][1] * scale,
        ]
        s["impact_pos_scaled"] = [
            s["impact_pos_px"][0] * scale,
            s["impact_pos_px"][1] * scale,
        ]

    comet_trail = deque(maxlen=int(round(0.42 * fps)))
    ema_pose_state = {"kpts": None}

    print("\n" + "=" * 70)
    print(
        f"PASS 3: RENDERING SMOOTH 1080P AR VIDEO ({target_w}x{target_h} @ {fps:.1f} FPS)"
    )
    print("=" * 70)

    pbar = tqdm(total=total_frames, desc="Rendering AR Video", unit="frame")
    f_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if scale != 1.0:
            frame = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        completed_impacts = [s for s in shots if f_idx >= s["impact_frame"]]
        active_strokes = [
            s for s in shots if s["stroke_frame"] <= f_idx < s["impact_frame"]
        ]
        active_shot = active_strokes[-1] if active_strokes else None

        flash_alpha = 0.0
        for s in shots:
            dt_imp = f_idx - s["impact_frame"]
            if 0 <= dt_imp <= int(0.25 * fps):
                flash_alpha = max(flash_alpha, 1.0 - dt_imp / max(1.0, 0.25 * fps))

        # 1. Draw 3D Perspective Wall Target & Pinned Shot Placement Marks
        draw_wall_target_ar(
            frame, wall_corners_scaled, completed_impacts, flash_alpha=flash_alpha
        )

        # 2. Draw EMA-Smoothed Player Pose Skeleton
        f_det = det_frames.get(str(f_idx), {})
        draw_player_skeleton_smoothed(
            frame, f_det.get("players", []), scale, ema_pose_state, alpha=0.45
        )

        # 3. Update & Draw Smoothed Ball Comet Trail
        if f_idx < len(ball_track) and ball_track[f_idx]["tracked"]:
            raw_cx = ball_track[f_idx].get("cx")
            raw_cy = ball_track[f_idx].get("cy")
            raw_d = ball_track[f_idx].get("d")
            if is_valid_num(raw_cx) and is_valid_num(raw_cy):
                bx = raw_cx * scale
                by = raw_cy * scale
                bd = max(6.0, (raw_d if is_valid_num(raw_d) else 12.0) * scale)
                is_coasted = ball_track[f_idx]["coasted"]
                comet_trail.append((int(round(bx)), int(round(by)), is_coasted))

                ball_col = (0, 215, 255) if is_coasted else (0, 255, 180)
                cv2.circle(
                    frame,
                    (int(round(bx)), int(round(by))),
                    int(max(5, bd * 0.55)),
                    ball_col,
                    2,
                    cv2.LINE_AA,
                )
            else:
                comet_trail.append(None)
        else:
            comet_trail.append(None)

        valid_pts = [
            (idx_p, pt) for idx_p, pt in enumerate(comet_trail) if pt is not None
        ]
        for k in range(1, len(valid_pts)):
            idx_prev, p1 = valid_pts[k - 1]
            idx_curr, p2 = valid_pts[k]
            if idx_curr - idx_prev <= 4:
                age_frac = idx_curr / max(1, len(comet_trail))
                thickness = max(1, int(round(1 + 5 * age_frac)))
                col = (int(50 * age_frac), int(220 * age_frac), int(255 * age_frac))
                cv2.line(
                    frame, (p1[0], p1[1]), (p2[0], p2[1]), col, thickness, cv2.LINE_AA
                )

        # 4. Highlight Racket Strike Callout Ring
        for s in shots:
            dt_stroke = f_idx - s["stroke_frame"]
            if 0 <= dt_stroke <= int(0.35 * fps):
                sx, sy = int(s["stroke_pos_scaled"][0]), int(s["stroke_pos_scaled"][1])
                ring_r = int(16 + dt_stroke * 2.2)
                st_col = (
                    (255, 220, 0) if s["stroke_type"] == "Forehand" else (0, 165, 255)
                )
                cv2.circle(frame, (sx, sy), ring_r, st_col, 2, cv2.LINE_AA)
                label = f"{s['stroke_type'].upper()} {s['speed_kmh']:.0f} km/h"
                cv2.putText(
                    frame,
                    label,
                    (sx - 55, max(35, sy - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    st_col,
                    2,
                    cv2.LINE_AA,
                )

        # 5. Draw Top-Left Telemetry HUD, Top-Right 2D Mini-Map & Bottom-Right Speed Chart
        draw_hud_dashboard(frame, completed_impacts, active_shot, shots, summary, calib)

        writer.write(frame)
        f_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"\n[Done!] Smooth Annotated AR Video saved to: {out_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/raw_videos/sample.mp4")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 path")
    args = parser.parse_args()

    clip_name = os.path.splitext(os.path.basename(args.video))[0]
    quad_file = f"cache/{clip_name}_wall_quad.json"
    det_file = f"cache/{clip_name}_detections.json"
    track_file = f"cache/{clip_name}_tracks_and_events.json"
    analysis_file = f"cache/{clip_name}_analysis.json"
    out_mp4 = args.out or f"outputs/{clip_name}_annotated.mp4"

    with open(quad_file, "r") as f:
        quad_payload = json.load(f)
    with open(det_file, "r") as f:
        det_payload = json.load(f)
    with open(track_file, "r") as f:
        track_payload = json.load(f)
    with open(analysis_file, "r") as f:
        analysis_payload = json.load(f)

    render_annotated_video(
        args.video, det_payload, track_payload, quad_payload, analysis_payload, out_mp4
    )
