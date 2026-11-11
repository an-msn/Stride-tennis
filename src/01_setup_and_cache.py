import os
import json
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


def select_wall_quad(video_path, out_json_path, force_reselect=False):
    """
    Pass 0: Interactive 4-corner wall target selector (TL, TR, BR, BL).
    Computes the 3x3 planar homography matrix mapping image pixels to the [0,1]x[0,1] unit square.
    """
    if os.path.exists(out_json_path) and not force_reselect:
        with open(out_json_path, "r") as f:
            data = json.load(f)
        print(f"[Pass 0] Loaded cached wall target quad from: {out_json_path}")
        return data

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read Frame 0 from {video_path}")

    points = []
    win_name = "PASS 0: Click 4 Wall Target Corners (1:TL, 2:TR, 3:BR, 4:BL) | 'r'=Reset | 'Enter'=Confirm"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([float(x), float(y)])

    cv2.setMouseCallback(win_name, mouse_cb)

    while True:
        vis = frame.copy()
        labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]
        for i, pt in enumerate(points):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                vis,
                labels[i],
                (px + 10, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if i > 0:
                p_prev = (int(points[i - 1][0]), int(points[i - 1][1]))
                cv2.line(vis, p_prev, (px, py), (0, 255, 180), 2, cv2.LINE_AA)
        if len(points) == 4:
            cv2.line(
                vis,
                (int(points[3][0]), int(points[3][1])),
                (int(points[0][0]), int(points[0][1])),
                (0, 255, 180),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(win_name, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key in (13, 32) and len(points) == 4:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Wall corner selection aborted by user.")

    cv2.destroyAllWindows()

    src_pts = np.array(points, dtype=np.float32)
    dst_pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    H_to_unit = cv2.getPerspectiveTransform(src_pts, dst_pts)
    H_from_unit = cv2.getPerspectiveTransform(dst_pts, src_pts)

    payload = {
        "video_path": video_path,
        "wall_corners_px": points,
        "homography_to_unit_square": H_to_unit.tolist(),
        "homography_from_unit_square": H_from_unit.tolist(),
    }
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Pass 0] Saved wall target quad & homography to: {out_json_path}")
    return payload


def resolve_ball_and_racket_classes(det_model):
    """Automatically detect whether det_model is our custom 2-class model (0,1) or COCO (32,38)."""
    names = det_model.names
    ball_ids = [
        int(k) for k, v in names.items() if v in ("tennis-ball", "sports ball", "ball")
    ]
    racket_ids = [
        int(k)
        for k, v in names.items()
        if v in ("tennis-racket", "tennis racket", "racket")
    ]
    if not ball_ids:
        ball_ids = [0] if len(names) <= 2 else [32]
    if not racket_ids:
        racket_ids = [1] if len(names) <= 2 else [38]
    return ball_ids, racket_ids


def run_and_cache_inference(
    video_path,
    det_weights,
    pose_weights,
    out_cache_path,
    batch_size=8,
    force_rerun=False,
):
    """
    Pass 1: Run batched 1024x1024 stretched detection + 640px pose inference once and cache to JSON.
    """
    if os.path.exists(out_cache_path) and not force_rerun:
        with open(out_cache_path, "r") as f:
            data = json.load(f)
        print(f"[Pass 1] Loaded cached detections & pose from: {out_cache_path}")
        return data

    device = 0 if torch.cuda.is_available() else "cpu"
    det_model = YOLO(det_weights)
    pose_model = YOLO(pose_weights)

    ball_cls_ids, racket_cls_ids = resolve_ball_and_racket_classes(det_model)
    target_cls_ids = ball_cls_ids + racket_cls_ids
    print(
        f"[Pass 1] Using Detector: {det_weights} | Ball IDs: {ball_cls_ids} | Racket IDs: {racket_cls_ids}"
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sx = width / 1024.0
    sy = height / 1024.0

    frames_dict = {}
    batch_orig = []
    batch_stretched = []
    batch_indices = []

    pbar = tqdm(
        total=total_frames, desc="Pass 1: Caching Detections & Pose", unit="frame"
    )
    f_idx = 0

    def process_batch():
        if not batch_orig:
            return
        det_results = det_model.predict(
            batch_stretched,
            imgsz=1024,
            rect=False,
            half=False,
            conf=0.10,
            classes=target_cls_ids,
            device=device,
            verbose=False,
        )
        pose_results = pose_model.predict(
            batch_orig, imgsz=640, half=False, conf=0.35, device=device, verbose=False
        )

        for b_i, frame_id in enumerate(batch_indices):
            d_res = det_results[b_i]
            p_res = pose_results[b_i]

            balls = []
            rackets = []
            if d_res.boxes is not None and len(d_res.boxes) > 0:
                for row in d_res.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls_id = row
                    cls_id = int(cls_id)
                    rx1, rx2 = round(float(x1 * sx), 2), round(float(x2 * sx), 2)
                    ry1, ry2 = round(float(y1 * sy), 2), round(float(y2 * sy), 2)
                    rconf = round(float(conf), 4)
                    if cls_id in ball_cls_ids:
                        balls.append([rx1, ry1, rx2, ry2, rconf])
                    elif cls_id in racket_cls_ids and rconf >= 0.20:
                        rackets.append([rx1, ry1, rx2, ry2, rconf])

            players = []
            if (
                p_res.boxes is not None
                and p_res.keypoints is not None
                and len(p_res.boxes) > 0
            ):
                boxes_np = p_res.boxes.data.cpu().numpy()
                kpts_np = p_res.keypoints.data.cpu().numpy()
                for p_i in range(len(boxes_np)):
                    bx1, by1, bx2, by2, bconf = boxes_np[p_i][:5]
                    kpts_list = [
                        [
                            round(float(k[0]), 2),
                            round(float(k[1]), 2),
                            round(float(k[2]), 3),
                        ]
                        for k in kpts_np[p_i]
                    ]
                    players.append(
                        {
                            "box": [
                                round(float(bx1), 2),
                                round(float(by1), 2),
                                round(float(bx2), 2),
                                round(float(by2), 2),
                            ],
                            "conf": round(float(bconf), 4),
                            "keypoints": kpts_list,
                        }
                    )

            frames_dict[str(frame_id)] = {
                "balls": balls,
                "rackets": rackets,
                "players": players,
            }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        stretched = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        batch_orig.append(frame)
        batch_stretched.append(stretched)
        batch_indices.append(f_idx)
        f_idx += 1

        if len(batch_orig) >= batch_size:
            process_batch()
            pbar.update(len(batch_orig))
            batch_orig.clear()
            batch_stretched.clear()
            batch_indices.clear()

    if batch_orig:
        process_batch()
        pbar.update(len(batch_orig))
        batch_orig.clear()
        batch_stretched.clear()
        batch_indices.clear()

    pbar.close()
    cap.release()

    pose_frames = sum(1 for v in frames_dict.values() if len(v["players"]) > 0)
    ball_frames = sum(1 for v in frames_dict.values() if len(v["balls"]) > 0)

    print("\n" + "=" * 65)
    print("PASS 0 & PASS 1 DIAGNOSTIC SUMMARY")
    print("=" * 65)
    print(f"Total Frames Processed      : {len(frames_dict)}")
    print(
        f"Frames with Player Pose     : {pose_frames}/{len(frames_dict)} ({100.0 * pose_frames / max(1, len(frames_dict)):.1f}%)"
    )
    print(
        f"Frames with Raw Ball Cand.  : {ball_frames}/{len(frames_dict)} ({100.0 * ball_frames / max(1, len(frames_dict)):.1f}%)"
    )
    print("=" * 65)

    payload = {
        "video_path": video_path,
        "det_weights": det_weights,
        "pose_weights": pose_weights,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": len(frames_dict),
        "frames": frames_dict,
    }
    os.makedirs(os.path.dirname(out_cache_path), exist_ok=True)
    with open(out_cache_path, "w") as f:
        json.dump(payload, f)
    print(f"[Pass 1] Saved detection & pose cache to: {out_cache_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/raw_videos/sample.mp4")
    parser.add_argument("--det-weights", type=str, default="models/yolo26s.pt")
    parser.add_argument("--pose-weights", type=str, default="models/yolo26s-pose.pt")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--reset-wall", action="store_true")
    parser.add_argument("--rerun-cache", action="store_true")
    args = parser.parse_args()

    clip_name = os.path.splitext(os.path.basename(args.video))[0]
    quad_file = f"cache/{clip_name}_wall_quad.json"
    det_cache_file = f"cache/{clip_name}_detections.json"

    select_wall_quad(args.video, quad_file, force_reselect=args.reset_wall)
    run_and_cache_inference(
        args.video,
        args.det_weights,
        args.pose_weights,
        det_cache_file,
        batch_size=args.batch,
        force_rerun=args.rerun_cache,
    )
