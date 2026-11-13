import os
import json
import csv
import math
import argparse
import numpy as np
import cv2

BALL_DIAMETER_M = 0.067  # Standard tennis ball diameter: 6.7 cm (Step 7)


def is_valid_num(val):
    return val is not None and not (
        isinstance(val, float) and (math.isnan(val) or math.isinf(val))
    )


def project_to_wall_square(pt_px, H_wall):
    """Map image (u, v) coordinate onto the normalized [0, 1] x [0, 1] wall target."""
    vec = np.array([[[float(pt_px[0]), float(pt_px[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(vec, np.array(H_wall, dtype=np.float64))[0, 0]
    return [round(float(dst[0]), 3), round(float(dst[1]), 3)]


def filter_valid_rallies(rallies, H_wall=None):
    """
    Step 6 Rule 1 + Homography Wall Gate:
    1. A ball at the racket must be physically larger in image space than when it
       reaches the wall (d_stroke > d_wall). Rejects ground-bounce or mid-air artifacts.
    2. If H_wall is provided, rejects extreme non-wall impacts (e.g., high fence hits).
    """
    valid = []
    dropped = 0
    for r in rallies:
        d_s = r.get("stroke_ball_d_px", r.get("stroke_d_px", 22.0))
        d_w = r.get("impact_ball_d_px", r.get("impact_d_px", 11.0))
        if d_s <= d_w * 1.05:
            dropped += 1
            continue

        if H_wall is not None:
            uv = project_to_wall_square(r["impact_pos_px"], H_wall)
            if not (-0.45 <= uv[0] <= 1.45 and -0.45 <= uv[1] <= 1.45):
                dropped += 1
                continue

        r_copy = dict(r)
        r_copy["stroke_ball_d_px"] = d_s
        r_copy["impact_ball_d_px"] = d_w
        r_copy["shot_number"] = len(valid) + 1
        valid.append(r_copy)

    # Fallback if strict filtering dropped everything
    if not valid and rallies:
        for idx, r in enumerate(rallies):
            r_copy = dict(r)
            r_copy["stroke_ball_d_px"] = r.get(
                "stroke_ball_d_px", r.get("stroke_d_px", 22.0)
            )
            r_copy["impact_ball_d_px"] = r.get(
                "impact_ball_d_px", r.get("impact_d_px", 11.0)
            )
            r_copy["shot_number"] = idx + 1
            valid.append(r_copy)

    return valid, dropped


def calibrate_camera(rallies, det_frames, wall_distance_m):
    """
    Step 7: Pin down focal length (f) using median ball diameter at strokes
    and wall impacts: BALL_D * f / d_wall - BALL_D * f / d_racket = wall_distance
    """
    stroke_ds = [
        r["stroke_ball_d_px"] for r in rallies if 8.0 <= r["stroke_ball_d_px"] <= 48.0
    ]
    wall_ds = [
        r["impact_ball_d_px"] for r in rallies if 4.0 <= r["impact_ball_d_px"] <= 30.0
    ]

    if not stroke_ds or not wall_ds:
        stroke_ds = [
            r["stroke_ball_d_px"]
            for r in rallies
            if is_valid_num(r["stroke_ball_d_px"])
        ]
        wall_ds = [
            r["impact_ball_d_px"]
            for r in rallies
            if is_valid_num(r["impact_ball_d_px"])
        ]
        if not stroke_ds or not wall_ds:
            raise RuntimeError(
                "Insufficient valid ball diameter samples to calibrate camera."
            )

    d_racket_med = float(np.median(stroke_ds))
    d_wall_med = float(np.median(wall_ds))

    if d_wall_med >= d_racket_med:
        d_racket_med = max(d_wall_med * 1.65, d_racket_med)

    inv_diff = (1.0 / d_wall_med) - (1.0 / d_racket_med)
    focal_px = wall_distance_m / (BALL_DIAMETER_M * inv_diff)

    z_racket_m = (BALL_DIAMETER_M * focal_px) / d_racket_med
    z_wall_m = (BALL_DIAMETER_M * focal_px) / d_wall_med

    player_heights_px = []
    for r in rallies:
        f_str = str(r["stroke_frame"])
        players = det_frames.get(f_str, {}).get("players", [])
        if players:
            p_box = max(
                players,
                key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]),
            )["box"]
            player_heights_px.append(p_box[3] - p_box[1])

    if player_heights_px:
        med_p_h = float(np.median(player_heights_px))
        z_player_standing_m = round(float((1.75 * focal_px) / med_p_h), 2)
        z_player_crouched_m = round(float((1.38 * focal_px) / med_p_h), 2)
        depth_agreement_pct = round(
            float(
                abs(z_racket_m - z_player_crouched_m) / max(1e-6, z_racket_m) * 100.0
            ),
            2,
        )
    else:
        z_player_standing_m = None
        z_player_crouched_m = None
        depth_agreement_pct = None

    return {
        "focal_length_px": round(focal_px, 2),
        "d_racket_median_px": round(d_racket_med, 2),
        "d_wall_median_px": round(d_wall_med, 2),
        "z_racket_m": round(z_racket_m, 2),
        "z_wall_m": round(z_wall_m, 2),
        "wall_distance_m": round(wall_distance_m, 2),
        "z_player_standing_check_m": z_player_standing_m,
        "z_player_crouched_check_m": z_player_crouched_m,
        "player_depth_agreement_pct": depth_agreement_pct,
    }


def compute_joint_angle(a, b, c):
    """Compute interior angle (degrees) at joint b given 2D points a-b-c."""
    ba = np.array(a[:2], dtype=np.float64) - np.array(b[:2], dtype=np.float64)
    bc = np.array(c[:2], dtype=np.float64) - np.array(b[:2], dtype=np.float64)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-5:
        return None
    cos_ang = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def extract_frame_knee_angle(player_data):
    """Return the active (more flexed) knee angle for a single frame's player keypoints."""
    kpts = player_data["keypoints"]
    angles = []
    if kpts[11][2] > 0.30 and kpts[13][2] > 0.30 and kpts[15][2] > 0.30:
        ang_l = compute_joint_angle(kpts[11], kpts[13], kpts[15])
        if ang_l is not None:
            angles.append(ang_l)
    if kpts[12][2] > 0.30 and kpts[14][2] > 0.30 and kpts[16][2] > 0.30:
        ang_r = compute_joint_angle(kpts[12], kpts[14], kpts[16])
        if ang_r is not None:
            angles.append(ang_r)
    return min(angles) if angles else None


def classify_stroke_and_posture(
    stroke_frame, ball_pos_px, det_frames, fps, handedness="right"
):
    """
    Step 9 + Upgraded 3D-Aware Torso-Spine, Multi-Frame Contact & Takeback Classifier:
    1. Evaluates contact-frame geometry (Shoulders kp 5,6, Hips kp 11,12, Hitting Wrist kp 10/9,
       and Racket box) at the true racket-snapped stroke_frame.
    2. Scans the contact window [stroke_frame - 4, stroke_frame + 2] for hitting-wrist side,
       dominant-shoulder relative offset, and two-handed backhand wrist coupling.
    3. Scans the pre-contact takeback window [stroke_frame - 10, stroke_frame - 2] for backhand
       cross-chest coiling so diagonal camera angles cannot flip Backhands into Forehands.
    4. Preserves all original output keys (shoulder_mid_px, contact_offset_px, wrist_offset_px,
       unit_turn_ratio, rel_spacing_ratio, knee_angle_deg, posture_note).
    """
    search_frames = [
        stroke_frame,
        stroke_frame - 1,
        stroke_frame + 1,
        stroke_frame - 2,
        stroke_frame + 2,
    ]
    player_data = None
    rackets = []

    for sf in search_frames:
        f_obj = det_frames.get(str(sf), {})
        if not rackets and f_obj.get("rackets"):
            rackets = f_obj["rackets"]
        if player_data is None and f_obj.get("players"):
            player_data = max(
                f_obj["players"],
                key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]),
            )

    if player_data is None:
        return {
            "stroke_type": "Forehand",
            "shoulder_mid_px": [None, None],
            "contact_offset_px": None,
            "wrist_offset_px": None,
            "unit_turn_ratio": None,
            "rel_spacing_ratio": None,
            "knee_angle_deg": None,
            "posture_note": "Pose unavailable",
        }

    # COCO 17 keypoints: 5=L_sh, 6=R_sh, 7=L_elb, 8=R_elb, 9=L_wrist, 10=R_wrist, 11=L_hip, 12=R_hip
    kpts = player_data["keypoints"]
    p_box = player_data["box"]
    box_h = max(40.0, p_box[3] - p_box[1])

    l_sh, r_sh = kpts[5], kpts[6]
    l_wr, r_wr = kpts[9], kpts[10]
    l_hip, r_hip = kpts[11], kpts[12]

    has_shoulders = l_sh[2] > 0.25 and r_sh[2] > 0.25
    has_hips = l_hip[2] > 0.25 and r_hip[2] > 0.25

    if has_shoulders:
        sh_mid_x = 0.5 * (l_sh[0] + r_sh[0])
        sh_mid_y = 0.5 * (l_sh[1] + r_sh[1])
        shoulder_span_x = abs(r_sh[0] - l_sh[0])
    else:
        sh_mid_x = 0.5 * (p_box[0] + p_box[2])
        sh_mid_y = p_box[1] + 0.24 * box_h
        shoulder_span_x = 0.35 * (p_box[2] - p_box[0])

    if has_hips:
        hip_mid_x = 0.5 * (l_hip[0] + r_hip[0])
        hip_mid_y = 0.5 * (l_hip[1] + r_hip[1])
        torso_center_x = 0.55 * sh_mid_x + 0.45 * hip_mid_x
    else:
        hip_mid_x = sh_mid_x
        hip_mid_y = p_box[1] + 0.54 * box_h
        torso_center_x = sh_mid_x

    # 1. Rotation-invariant vertical torso spine length (shoulder midpoint to hip midpoint)
    if has_shoulders and has_hips:
        spine_len_px = float(np.hypot(hip_mid_x - sh_mid_x, hip_mid_y - sh_mid_y))
        spine_len_px = max(spine_len_px, 0.22 * box_h)
    else:
        spine_len_px = 0.30 * box_h

    # 2. 3D Torso Unit-Turn Ratio: lower ratio (<0.45) indicates strong sideways shoulder rotation
    unit_turn_ratio = round(float(shoulder_span_x / max(20.0, spine_len_px)), 2)

    # 3. Infer facing orientation (facing wall away from camera = +1.0; facing camera = -1.0)
    facing_sign = 1.0
    if has_shoulders and has_hips:
        if (r_sh[0] - l_sh[0]) < -25.0 and (r_hip[0] - l_hip[0]) < -18.0:
            facing_sign = -1.0

    is_right_handed = handedness.lower().startswith("r")
    hand_sign = (1.0 if is_right_handed else -1.0) * facing_sign
    dom_wrist_idx = 10 if is_right_handed else 9
    non_wrist_idx = 9 if is_right_handed else 10
    dom_elbow_idx = 8 if is_right_handed else 7
    dom_sh_idx = 6 if is_right_handed else 5

    dom_wrist = r_wr if is_right_handed else l_wr
    wrist_offset_x = (dom_wrist[0] - torso_center_x) if dom_wrist[2] > 0.25 else None

    # 4. Contact Ball & Racket Offsets (normalized by spine_len_px)
    ball_x = float(ball_pos_px[0])
    ball_offset_x = ball_x - torso_center_x
    ball_rel = hand_sign * (ball_offset_x / max(28.0, spine_len_px))

    racket_offset_x = None
    if rackets:
        r_box = max(rackets, key=lambda r: r[4])
        racket_cx = 0.5 * (r_box[0] + r_box[2])
        racket_offset_x = racket_cx - torso_center_x
    racket_rel = (
        hand_sign * (racket_offset_x / max(28.0, spine_len_px))
        if racket_offset_x is not None
        else ball_rel
    )

    # 5. Scan 220ms Pre-Contact Window for Multi-Frame Wrist Side, Takeback Coiling & Knee Flexion
    load_win = max(6, int(round(0.22 * fps)))
    window_knee_angles = []
    contact_window_scores = []
    takeback_scores = []

    for f_idx in range(max(0, stroke_frame - load_win), stroke_frame + 3):
        f_players = det_frames.get(str(f_idx), {}).get("players", [])
        if not f_players:
            continue
        p_curr = max(
            f_players,
            key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]),
        )
        k_ang = extract_frame_knee_angle(p_curr)
        if k_ang is not None and f_idx <= stroke_frame + 1:
            window_knee_angles.append(k_ang)

        kp_c = p_curr["keypoints"]
        sh_pts = [kp_c[i][:2] for i in (5, 6) if kp_c[i][2] > 0.25]
        hip_pts = [kp_c[i][:2] for i in (11, 12) if kp_c[i][2] > 0.25]
        if not sh_pts or not hip_pts:
            continue

        m_sh = np.mean(sh_pts, axis=0)
        m_hip = np.mean(hip_pts, axis=0)
        t_cx = float(0.55 * m_sh[0] + 0.45 * m_hip[0])
        sp_len = float(max(28.0, np.linalg.norm(m_sh - m_hip)))
        d_sh_x = (
            float(kp_c[dom_sh_idx][0])
            if kp_c[dom_sh_idx][2] > 0.25
            else (t_cx + hand_sign * 0.22 * sp_len)
        )

        # Contact window [-4, +2]: hitting wrist offset + 2-handed backhand wrist coupling
        if stroke_frame - 4 <= f_idx <= stroke_frame + 2:
            if kp_c[dom_wrist_idx][2] > 0.25:
                wx = float(kp_c[dom_wrist_idx][0])
                w_rel_torso = hand_sign * (wx - t_cx) / sp_len
                w_rel_sh = hand_sign * (wx - d_sh_x) / sp_len
                c_score = 0.65 * w_rel_torso + 0.35 * w_rel_sh

                if kp_c[non_wrist_idx][2] > 0.25:
                    nwx = float(kp_c[non_wrist_idx][0])
                    wrist_gap = abs(wx - nwx) / sp_len
                    mid_w_rel = hand_sign * (0.5 * (wx + nwx) - t_cx) / sp_len
                    if wrist_gap < 0.38 and mid_w_rel < 0.05:
                        c_score -= 0.45
                contact_window_scores.append(c_score)

        # Takeback window [-10, -2]: dominant wrist & elbow coiling across body
        if stroke_frame - 10 <= f_idx <= stroke_frame - 2:
            if kp_c[dom_wrist_idx][2] > 0.25:
                wx = float(kp_c[dom_wrist_idx][0])
                tb_score = hand_sign * (wx - t_cx) / sp_len
                if kp_c[dom_elbow_idx][2] > 0.25:
                    ex = float(kp_c[dom_elbow_idx][0])
                    tb_score = 0.70 * tb_score + 0.30 * (
                        hand_sign * (ex - t_cx) / sp_len
                    )
                takeback_scores.append(tb_score)

    wrist_contact_rel = (
        float(np.median(contact_window_scores)) if contact_window_scores else ball_rel
    )
    takeback_rel = (
        float(np.median(takeback_scores)) if takeback_scores else wrist_contact_rel
    )

    combined_score = (
        0.45 * ball_rel
        + 0.30 * wrist_contact_rel
        + 0.15 * racket_rel
        + 0.10 * takeback_rel
    )
    stroke_type = "Forehand" if combined_score >= 0.0 else "Backhand"

    # Compute equivalent pixel fused offset for JSON telemetry compatibility
    fused_offset_x = float(combined_score * spine_len_px * hand_sign)

    if window_knee_angles:
        knee_angle = round(float(np.percentile(window_knee_angles, 15)), 1)
    else:
        knee_angle = None

    rel_ext = abs(ball_offset_x) / max(20.0, spine_len_px)

    if knee_angle is not None and knee_angle > 158.0:
        posture_note = "Bend knees more in prep (stiff legs)"
    elif rel_ext < 0.55:
        posture_note = "Jammed close to body (create spacing)"
    elif rel_ext > 1.55:
        posture_note = "Reaching outside base (move feet)"
    elif knee_angle is not None and knee_angle <= 138.0:
        posture_note = "Strong knee load & unit turn"
    else:
        posture_note = "Good athletic spacing & base"

    return {
        "stroke_type": stroke_type,
        "shoulder_mid_px": [round(float(torso_center_x), 1), round(float(sh_mid_y), 1)],
        "contact_offset_px": round(float(fused_offset_x), 1),
        "wrist_offset_px": round(float(wrist_offset_x), 1)
        if wrist_offset_x is not None
        else None,
        "unit_turn_ratio": unit_turn_ratio,
        "rel_spacing_ratio": round(float(rel_ext), 2),
        "knee_angle_deg": knee_angle,
        "posture_note": posture_note,
    }


def analyze_rallies(
    track_payload, det_payload, quad_payload, wall_distance_m, handedness="right"
):
    raw_rallies = track_payload["rallies"]
    H_wall = quad_payload["homography_to_unit_square"]
    rallies, dropped_bounces = filter_valid_rallies(raw_rallies, H_wall=H_wall)

    det_frames = det_payload["frames"]
    fps = track_payload["fps"]
    width = track_payload["width"]
    height = track_payload["height"]
    cx_prin, cy_prin = width / 2.0, height / 2.0

    calib = calibrate_camera(rallies, det_frames, wall_distance_m)
    calib["dropped_ground_bounces"] = dropped_bounces
    f_px = calib["focal_length_px"]
    z_racket = calib["z_racket_m"]
    z_wall = calib["z_wall_m"]

    analyzed_shots = []
    speeds_kmh = []

    for r in rallies:
        u_r, v_r = r["stroke_pos_px"]
        u_w, v_w = r["impact_pos_px"]
        dt_sec = r["flight_time_sec"]

        x_r = (u_r - cx_prin) * z_racket / f_px
        y_r = (v_r - cy_prin) * z_racket / f_px
        x_w = (u_w - cx_prin) * z_wall / f_px
        y_w = (v_w - cy_prin) * z_wall / f_px

        dx_m = x_w - x_r
        dy_m = y_w - y_r
        dz_m = wall_distance_m

        dist_3d_m = float(np.sqrt(dx_m**2 + dy_m**2 + dz_m**2))
        speed_kmh = float(np.clip((dist_3d_m / max(0.18, dt_sec)) * 3.6, 16.0, 115.0))
        speeds_kmh.append(speed_kmh)

        pose_info = classify_stroke_and_posture(
            r["stroke_frame"],
            r["stroke_pos_px"],
            det_frames,
            fps=fps,
            handedness=handedness,
        )

        wall_uv = project_to_wall_square(r["impact_pos_px"], H_wall)
        in_target = (0.0 <= wall_uv[0] <= 1.0) and (0.0 <= wall_uv[1] <= 1.0)

        shot_entry = {
            **r,
            "speed_kmh": round(speed_kmh, 1),
            "distance_3d_m": round(dist_3d_m, 2),
            "stroke_type": pose_info["stroke_type"],
            "shoulder_mid_px": pose_info["shoulder_mid_px"],
            "contact_offset_px": pose_info["contact_offset_px"],
            "wrist_offset_px": pose_info["wrist_offset_px"],
            "unit_turn_ratio": pose_info["unit_turn_ratio"],
            "rel_spacing_ratio": pose_info["rel_spacing_ratio"],
            "knee_angle_deg": pose_info["knee_angle_deg"],
            "posture_note": pose_info["posture_note"],
            "wall_target_uv": wall_uv,
            "inside_target_quad": in_target,
        }
        analyzed_shots.append(shot_entry)

    rally_period_s = track_payload["diagnostics"]["rally_period_sec"]
    expected_speed_kmh = (wall_distance_m / (rally_period_s / 2.0)) * 3.6
    median_speed_kmh = float(np.median(speeds_kmh)) if speeds_kmh else 0.0

    fh_count = sum(1 for s in analyzed_shots if s["stroke_type"] == "Forehand")
    bh_count = sum(1 for s in analyzed_shots if s["stroke_type"] == "Backhand")

    summary = {
        "total_strokes": len(analyzed_shots),
        "forehand_count": fh_count,
        "backhand_count": bh_count,
        "median_speed_kmh": round(median_speed_kmh, 1),
        "max_speed_kmh": round(float(np.max(speeds_kmh)), 1) if speeds_kmh else 0.0,
        "sanity_check_expected_kmh": round(expected_speed_kmh, 1),
        "target_hit_rate_pct": round(
            100.0
            * sum(1 for s in analyzed_shots if s["inside_target_quad"])
            / max(1, len(analyzed_shots)),
            1,
        ),
    }

    return calib, summary, analyzed_shots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/raw_videos/sample.mp4")
    parser.add_argument(
        "--wall-dist", type=float, default=6.5, help="Player-to-wall distance in meters"
    )
    parser.add_argument(
        "--handedness", type=str, default="right", choices=["right", "left"]
    )
    args = parser.parse_args()

    clip_name = os.path.splitext(os.path.basename(args.video))[0]
    quad_file = f"cache/{clip_name}_wall_quad.json"
    det_file = f"cache/{clip_name}_detections.json"
    track_file = f"cache/{clip_name}_tracks_and_events.json"
    out_analysis_file = f"cache/{clip_name}_analysis.json"
    out_csv_file = f"outputs/{clip_name}_metrics.csv"

    with open(quad_file, "r") as f:
        quad_payload = json.load(f)
    with open(det_file, "r") as f:
        det_payload = json.load(f)
    with open(track_file, "r") as f:
        track_payload = json.load(f)

    calib, summary, shots = analyze_rallies(
        track_payload,
        det_payload,
        quad_payload,
        wall_distance_m=args.wall_dist,
        handedness=args.handedness,
    )

    full_output = {
        "clip_name": clip_name,
        "calibration": calib,
        "summary": summary,
        "shots": shots,
    }
    with open(out_analysis_file, "w") as f:
        json.dump(full_output, f, indent=2)

    os.makedirs(os.path.dirname(out_csv_file), exist_ok=True)
    if shots:
        csv_keys = [
            "shot_number",
            "stroke_time_sec",
            "impact_time_sec",
            "flight_time_sec",
            "audio_verified",
            "stroke_type",
            "speed_kmh",
            "wall_target_uv",
            "inside_target_quad",
            "unit_turn_ratio",
            "rel_spacing_ratio",
            "knee_angle_deg",
            "posture_note",
        ]
        with open(out_csv_file, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(shots)

    print("\n" + "=" * 96)
    print("PASS 2B: CAMERA CALIBRATION, SPEED & 3D TORSO-WRIST BIOMECHANICAL ANALYSIS")
    print("=" * 96)
    print(
        f"Solved Camera Focal Length (f) : {calib['focal_length_px']:.1f} px (Filtered {calib['dropped_ground_bounces']} non-wall bounce)"
    )
    print(
        f"Median Ball Size (Racket/Wall) : {calib['d_racket_median_px']:.1f} px (Z={calib['z_racket_m']:.2f}m) -> {calib['d_wall_median_px']:.1f} px (Z={calib['z_wall_m']:.2f}m)"
    )
    if calib["z_player_crouched_check_m"] is not None:
        print(
            f"Player Box Depth Sanity Check  : {calib['z_player_crouched_check_m']:.2f}m vs Racket {calib['z_racket_m']:.2f}m ({calib['player_depth_agreement_pct']:.1f}% diff)"
        )
    print(
        f"Speed Sanity Check (Step 8)    : Median = {summary['median_speed_kmh']:.1f} km/h | Theoretical = {summary['sanity_check_expected_kmh']:.1f} km/h"
    )
    print(
        f"Stroke Breakdown               : {summary['total_strokes']} Total ({summary['forehand_count']} Forehands, {summary['backhand_count']} Backhands)"
    )
    print("-" * 96)
    for s in shots:
        knee_str = (
            f"{s['knee_angle_deg']:5.1f} deg"
            if is_valid_num(s["knee_angle_deg"])
            else "    N/A  "
        )
        spc_str = (
            f"{s['rel_spacing_ratio']:4.2f}x"
            if is_valid_num(s.get("rel_spacing_ratio"))
            else " N/A "
        )
        ut_str = (
            f"{s['unit_turn_ratio']:4.2f}"
            if is_valid_num(s.get("unit_turn_ratio"))
            else " N/A"
        )
        uv_x, uv_y = s["wall_target_uv"]
        print(
            f"  #{s['shot_number']:02d} | {s['stroke_type']:8s} | {s['speed_kmh']:5.1f} km/h | "
            f"Wall: ({uv_x:+.2f},{uv_y:+.2f}) | Knee: {knee_str} | Space: {spc_str} | Turn: {ut_str} | {s['posture_note']}"
        )
    print("=" * 96)
    print(f"[Saved] Analysis JSON : {out_analysis_file}")
    print(f"[Saved] Metrics CSV   : {out_csv_file}")
