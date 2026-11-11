import os
import sys
import json
import csv
import time
import argparse
import importlib.util


def load_stage_module(module_name, file_path):
    """Dynamically load a stage script whose filename starts with digits."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required stage script not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_src_dir():
    """Locate the src/ directory whether run from project root or inside src/."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here) == "src":
        return here
    candidate = os.path.join(here, "src")
    if os.path.isdir(candidate):
        return candidate
    return "src"


def main():
    parser = argparse.ArgumentParser(
        description="STRIDE Tennis CV: Unified End-to-End Monocular Pipeline Runner"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw_videos/sample.mp4",
        help="Path to input wall-practice video (.mp4)",
    )
    parser.add_argument(
        "--wall-dist",
        type=float,
        default=6.5,
        help="Measured player-to-wall distance in meters (default: 6.5)",
    )
    parser.add_argument(
        "--handedness",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Player handedness for Forehand/Backhand classification (default: right)",
    )
    parser.add_argument(
        "--det-weights",
        type=str,
        default="models/yolo26s.pt",
        help="Path to YOLO26 ball/racket detector weights",
    )
    parser.add_argument(
        "--pose-weights",
        type=str,
        default="models/yolo26s-pose.pt",
        help="Path to YOLO26-Pose player keypoint weights",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="GPU batch size for Pass 1 inference caching (default: 8)",
    )
    parser.add_argument(
        "--reset-wall",
        action="store_true",
        help="Force re-clicking the 4 target wall corners on Frame 0",
    )
    parser.add_argument(
        "--rerun-cache",
        action="store_true",
        help="Force re-running GPU detection & pose inference instead of using cache",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Run tracking, calibration, and CSV analysis only (skip MP4 rendering)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Custom output path for the annotated MP4 video",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Input video not found at: {args.video}")

    src_dir = resolve_src_dir()
    stage01 = load_stage_module(
        "stage01_setup_cache", os.path.join(src_dir, "01_setup_and_cache.py")
    )
    stage02 = load_stage_module(
        "stage02_track_events", os.path.join(src_dir, "02_track_and_detect_events.py")
    )
    stage03 = load_stage_module(
        "stage03_calibrate_analyze",
        os.path.join(src_dir, "03_calibrate_and_analyze.py"),
    )
    stage04 = load_stage_module(
        "stage04_render_ar", os.path.join(src_dir, "04_render_ar_video.py")
    )

    clip_name = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs("cache", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    quad_file = f"cache/{clip_name}_wall_quad.json"
    det_cache_file = f"cache/{clip_name}_detections.json"
    track_file = f"cache/{clip_name}_tracks_and_events.json"
    analysis_file = f"cache/{clip_name}_analysis.json"
    out_plot_file = f"outputs/{clip_name}_diagnostics.png"
    out_csv_file = f"outputs/{clip_name}_metrics.csv"
    out_mp4_file = args.out or f"outputs/{clip_name}_annotated.mp4"

    t_pipeline_start = time.perf_counter()
    timings = {}

    print("\n" + "#" * 86)
    print(
        f"STRIDE TENNIS AI PIPELINE | CLIP: {clip_name} | WALL DIST: {args.wall_dist:.2f}m | HAND: {args.handedness.upper()}"
    )
    print("#" * 86)

    # -------------------------------------------------------------------------
    # PASS 0: 4-Corner Wall Target Selection
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    quad_payload = stage01.select_wall_quad(
        args.video, quad_file, force_reselect=args.reset_wall
    )
    timings["Pass 0 (Wall Quad)"] = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # PASS 1: Batched GPU Detection & Pose Caching
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    det_payload = stage01.run_and_cache_inference(
        args.video,
        args.det_weights,
        args.pose_weights,
        det_cache_file,
        batch_size=args.batch,
        force_rerun=args.rerun_cache,
    )
    timings["Pass 1 (GPU Cache)"] = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # PASS 2A: Static Ball Suppression, Kalman Tracking & Audio-Visual Events
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    fps = det_payload["fps"]
    width = det_payload["width"]
    height = det_payload["height"]
    total_frames = det_payload["total_frames"]

    cleaned_balls, removed_static, kept_dynamic = stage02.suppress_static_balls(
        det_payload["frames"], fps
    )

    tracker = stage02.AsymmetricBallKalmanTracker(width, height, fps)
    track_list = []
    for idx in range(total_frames):
        state = tracker.step(cleaned_balls.get(idx, []))
        state["frame_idx"] = idx
        track_list.append(state)

    track_list = stage02.smooth_ball_trajectory(track_list)
    tracked_count = sum(1 for t in track_list if t["tracked"])
    coasted_count = sum(1 for t in track_list if t["coasted"])

    paired_rallies, diag = stage02.analyze_proximity_and_events(
        track_list, quad_payload["wall_corners_px"], fps, video_path=args.video
    )
    stage02.save_diagnostic_plot(track_list, paired_rallies, diag, fps, out_plot_file)

    track_payload = {
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
    with open(track_file, "w") as f:
        json.dump(track_payload, f, indent=2)
    timings["Pass 2A (Tracking & Signal)"] = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # PASS 2B: Camera Calibration, 3D Speed & Torso-Wrist Biomechanics
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    calib, summary, shots = stage03.analyze_rallies(
        track_payload,
        det_payload,
        quad_payload,
        wall_distance_m=args.wall_dist,
        handedness=args.handedness,
    )

    analysis_payload = {
        "clip_name": clip_name,
        "calibration": calib,
        "summary": summary,
        "shots": shots,
    }
    with open(analysis_file, "w") as f:
        json.dump(analysis_payload, f, indent=2)

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
    timings["Pass 2B (Calibration & Bio)"] = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # PASS 3: 1080p AR Video & Telemetry HUD Rendering
    # -------------------------------------------------------------------------
    if not args.skip_render:
        t0 = time.perf_counter()
        stage04.render_annotated_video(
            args.video,
            det_payload,
            track_payload,
            quad_payload,
            analysis_payload,
            out_mp4_file,
        )
        timings["Pass 3 (AR Video Render)"] = time.perf_counter() - t0

    total_elapsed = time.perf_counter() - t_pipeline_start

    # -------------------------------------------------------------------------
    # MASTER PIPELINE REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("STRIDE MASTER PIPELINE REPORT")
    print("=" * 96)
    print(
        f"Video Clip & Resolution        : {args.video} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)"
    )
    print(
        f"Kalman Ball Track Coverage     : {tracked_count}/{total_frames} frames ({100.0 * tracked_count / max(1, total_frames):.1f}% | Static dropped: {removed_static})"
    )
    print(
        f"Proximity y-d Correlation      : {diag['corr_y_diameter']:+.3f} | Autocorrelation Rally Period: {diag['rally_period_sec']:.2f}s"
    )
    print(
        f"Calibrated Focal Length (f)    : {calib['focal_length_px']:.1f} px (Racket Z={calib['z_racket_m']:.2f}m -> Wall Z={calib['z_wall_m']:.2f}m)"
    )
    print(
        f"Validated Strokes              : {summary['total_strokes']} Total ({summary['forehand_count']} Forehands, {summary['backhand_count']} Backhands)"
    )
    print(
        f"Outbound Shot Speed            : Median = {summary['median_speed_kmh']:.1f} km/h | Max = {summary['max_speed_kmh']:.1f} km/h | Target Hit Rate = {summary['target_hit_rate_pct']:.1f}%"
    )
    print("-" * 96)
    for s in shots:
        knee_val = s.get("knee_angle_deg")
        knee_str = (
            f"{knee_val:5.1f} deg" if stage03.is_valid_num(knee_val) else "    N/A  "
        )
        spc_val = s.get("rel_spacing_ratio")
        spc_str = f"{spc_val:4.2f}x" if stage03.is_valid_num(spc_val) else " N/A "
        ut_val = s.get("unit_turn_ratio")
        ut_str = f"{ut_val:4.2f}" if stage03.is_valid_num(ut_val) else " N/A"
        uv_x, uv_y = s["wall_target_uv"]
        print(
            f"  #{s['shot_number']:02d} | {s['stroke_type']:8s} | {s['speed_kmh']:5.1f} km/h | "
            f"Wall: ({uv_x:+.2f},{uv_y:+.2f}) | Knee: {knee_str} | Space: {spc_str} | Turn: {ut_str} | {s['posture_note']}"
        )
    print("-" * 96)
    print("STAGE EXECUTION TIMINGS:")
    for stage_label, sec in timings.items():
        print(f"  - {stage_label:28s}: {sec:6.2f} s")
    print(f"  - {'Total Pipeline Time':28s}: {total_elapsed:6.2f} s")
    print("-" * 96)
    print("GENERATED ARTIFACTS:")
    print(f"  [1] Diagnostic Signal Plot   : {out_plot_file}")
    print(f"  [2] Shot Telemetry CSV       : {out_csv_file}")
    print(f"  [3] Full Analysis JSON       : {analysis_file}")
    if not args.skip_render:
        print(f"  [4] Annotated 1080p AR Video : {out_mp4_file}")
    print("=" * 96)


if __name__ == "__main__":
    main()
