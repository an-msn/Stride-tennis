import os
import glob
import json
import shutil
import random
import argparse
from collections import defaultdict
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Custom 2-class schema: focuses 100% of detector capacity on fast-moving targets
CLASS_NAMES = {0: "tennis-ball", 1: "tennis-racket"}


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel [x1, y1, x2, y2] to normalized YOLO [cx, cy, w, h]."""
    cx = np.clip((0.5 * (x1 + x2)) / img_w, 0.0, 1.0)
    cy = np.clip((0.5 * (y1 + y2)) / img_h, 0.0, 1.0)
    bw = np.clip((x2 - x1) / img_w, 0.0, 1.0)
    bh = np.clip((y2 - y1) / img_h, 0.0, 1.0)
    return float(cx), float(cy), float(bw), float(bh)


def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    """Convert normalized YOLO [cx, cy, w, h] back to pixel [x1, y1, x2, y2]."""
    w = bw * img_w
    h = bh * img_h
    x1 = int(round((cx * img_w) - 0.5 * w))
    y1 = int(round((cy * img_h) - 0.5 * h))
    x2 = int(round((cx * img_w) + 0.5 * w))
    y2 = int(round((cy * img_h) + 0.5 * h))
    return x1, y1, x2, y2


def sample_and_autolabel(video_paths, raw_out_dir, det_weights, sample_fps=2.0):
    """
    Steps 2 & 3 of README:
    1. Sample frames out of video clips at 2 FPS.
    2. Auto-label tennis-ball (cls 0) and tennis-racket (cls 1) at 1024x1024 stretched.
    3. Automatically drop static "invented balls" appearing at the identical pixel across frames.
    """
    img_dir = os.path.join(raw_out_dir, "images")
    lbl_dir = os.path.join(raw_out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    det_model = YOLO(det_weights)
    print("\n" + "=" * 72)
    print(f"STEP 2 & 3A: SAMPLING AT {sample_fps:.1f} FPS & AUTO-LABELING")
    print("=" * 72)

    all_sampled = []  # [(stem, img_w, img_h, balls_xyxy, rackets_xyxy)]

    for v_path in video_paths:
        if not os.path.exists(v_path):
            print(f"[Warning] Skipping missing video: {v_path}")
            continue

        clip_name = os.path.splitext(os.path.basename(v_path))[0]
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, int(round(fps / sample_fps)))

        # Track spatial buckets per clip to filter static false-positive balls
        bucket_hits = defaultdict(list)
        clip_records = []
        f_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % step == 0:
                stem = f"{clip_name}_f{f_idx:05d}"
                img_path = os.path.join(img_dir, f"{stem}.jpg")
                cv2.imwrite(img_path, frame)

                # Run stretched 1024x1024 inference for initial auto-labels
                stretched = cv2.resize(
                    frame, (1024, 1024), interpolation=cv2.INTER_LINEAR
                )
                res = det_model.predict(
                    stretched,
                    imgsz=1024,
                    rect=False,
                    half=False,
                    conf=0.20,
                    classes=[32, 38],
                    verbose=False,
                )[0]

                sx, sy = img_w / 1024.0, img_h / 1024.0
                balls, rackets = [], []

                if res.boxes is not None and len(res.boxes) > 0:
                    for b in res.boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls_id = b
                        x1, x2 = x1 * sx, x2 * sx
                        y1, y2 = y1 * sy, y2 * sy
                        bw, bh = x2 - x1, y2 - y1
                        cls_id = int(cls_id)

                        if cls_id == 32:  # sports ball
                            # Physical ball size filter (reject giant shoe/racket boxes)
                            if (
                                min(bw, bh) > 45.0
                                or max(bw, bh) > 65.0
                                or max(bw, bh) < 4.0
                            ):
                                continue
                            balls.append([x1, y1, x2, y2, conf])
                            bx = int(round(0.5 * (x1 + x2) / 6.0))
                            by = int(round(0.5 * (y1 + y2) / 6.0))
                            bucket_hits[(bx, by)].append(stem)
                        elif cls_id == 38 and conf >= 0.30:  # tennis racket
                            rackets.append([x1, y1, x2, y2, conf])

                clip_records.append((stem, img_w, img_h, balls, rackets))
            f_idx += 1
        cap.release()

        # Identify static buckets (boxes appearing at identical pixel across >= 3 sampled frames)
        static_buckets = {k for k, stems in bucket_hits.items() if len(stems) >= 3}
        static_dropped = 0

        for stem, w, h, balls, rackets in clip_records:
            clean_balls = []
            for b in balls:
                bx = int(round(0.5 * (b[0] + b[2]) / 6.0))
                by = int(round(0.5 * (b[1] + b[3]) / 6.0))
                if (bx, by) in static_buckets:
                    static_dropped += 1
                else:
                    clean_balls.append(b)
            # Keep at most 1 highest-confidence ball and 1 racket per frame
            if len(clean_balls) > 1:
                clean_balls = [max(clean_balls, key=lambda x: x[4])]
            if len(rackets) > 1:
                rackets = [max(rackets, key=lambda x: x[4])]

            all_sampled.append((stem, w, h, clean_balls, rackets))

        print(
            f"  Clip '{clip_name}': Sampled {len(clip_records)} frames | Dropped {static_dropped} static ball boxes"
        )

    # Save YOLO format .txt label files
    missing_ball_frames = 0
    for stem, w, h, balls, rackets in all_sampled:
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        lines = []
        if not balls:
            missing_ball_frames += 1
        for b in balls:
            cx, cy, bw, bh = xyxy_to_yolo(b[0], b[1], b[2], b[3], w, h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        for r in rackets:
            cx, cy, bw, bh = xyxy_to_yolo(r[0], r[1], r[2], r[3], w, h)
            lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    print("-" * 72)
    print(f"Total Sampled Frames  : {len(all_sampled)} (Saved to {raw_out_dir})")
    print(
        f"Frames Missing a Ball : {missing_ball_frames}/{len(all_sampled)} (Usually motion-blurred contact frames!)"
    )
    print(
        "Next: Run '--mode review' to inspect frames and draw boxes on any missed balls."
    )
    print("=" * 72)


def interactive_review_ui(raw_out_dir):
    """
    Step 3B of README: Interactive OpenCV Label Reviewer & Motion-Blur Fixer.
    - Left-click & drag  : Draw 'tennis-ball' (Class 0 - Green)
    - Right-click & drag : Draw 'tennis-racket' (Class 1 - Orange)
    - 'd' key            : Delete last box on current frame
    - 'c' key            : Clear all ball boxes on current frame
    - 'n' or Space/Enter : Save & go to NEXT frame
    - 'p' key            : Save & go to PREVIOUS frame
    - 'm' key            : Jump directly to next frame MISSING a ball
    - 'q' or ESC         : Save & Finish review
    """
    img_dir = os.path.join(raw_out_dir, "images")
    lbl_dir = os.path.join(raw_out_dir, "labels")
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_files:
        raise RuntimeError(
            f"No sampled images found in {img_dir}. Run '--mode sample' first."
        )

    def load_labels(lbl_path, w, h):
        boxes = []  # [[cls_id, x1, y1, x2, y2]]
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
                        boxes.append([cls_id, x1, y1, x2, y2])
        return boxes

    def save_labels(lbl_path, boxes, w, h):
        lines = []
        for cls_id, x1, y1, x2, y2 in boxes:
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    win_name = "STEP 3 REVIEW: L-Drag=Ball | R-Drag=Racket | d=Undo | m=Next Missing | Space=Next | q=Save&Exit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 760)

    state = {
        "drawing": False,
        "cls": 0,
        "ix": 0,
        "iy": 0,
        "cx": 0,
        "cy": 0,
        "boxes": [],
    }

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["cls"] = 0
            state["ix"], state["iy"] = x, y
            state["cx"], state["cy"] = x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["drawing"] = True
            state["cls"] = 1
            state["ix"], state["iy"] = x, y
            state["cx"], state["cy"] = x, y
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["cx"], state["cy"] = x, y
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and state["drawing"]:
            state["drawing"] = False
            x1, x2 = sorted([state["ix"], x])
            y1, y2 = sorted([state["iy"], y])
            if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                # If adding a ball, replace existing ball so there's 1 clean ball per frame
                if state["cls"] == 0:
                    state["boxes"] = [b for b in state["boxes"] if b[0] != 0]
                state["boxes"].append([state["cls"], x1, y1, x2, y2])

    cv2.setMouseCallback(win_name, mouse_cb)

    idx = 0
    total = len(img_files)
    while 0 <= idx < total:
        img_path = img_files[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")

        orig = cv2.imread(img_path)
        h, w = orig.shape[:2]
        state["boxes"] = load_labels(lbl_path, w, h)

        while True:
            vis = orig.copy()
            has_ball = any(b[0] == 0 for b in state["boxes"])

            # Top banner
            banner_col = (25, 95, 25) if has_ball else (20, 30, 160)
            cv2.rectangle(vis, (0, 0), (w, 42), banner_col, -1)
            status_txt = (
                "BALL LABELED"
                if has_ball
                else "MISSING BALL (DRAW WITH LEFT-CLICK DRAG IF VISIBLE)"
            )
            cv2.putText(
                vis,
                f"[{idx + 1}/{total}] {stem} | {status_txt} | [Space]=Next [p]=Prev [m]=JumpMissing [d]=Undo [q]=Finish",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw existing boxes
            for cls_id, x1, y1, x2, y2 in state["boxes"]:
                col = (0, 255, 100) if cls_id == 0 else (0, 165, 255)
                name = "ball" if cls_id == 0 else "racket"
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
                cv2.putText(
                    vis,
                    name,
                    (x1, max(55, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    col,
                    2,
                    cv2.LINE_AA,
                )

            # Draw active rubber-band box
            if state["drawing"]:
                col = (0, 255, 100) if state["cls"] == 0 else (0, 165, 255)
                cv2.rectangle(
                    vis, (state["ix"], state["iy"]), (state["cx"], state["cy"]), col, 2
                )

            cv2.imshow(win_name, vis)
            key = cv2.waitKey(20) & 0xFF

            if key in (32, 13, ord("n")):  # Space, Enter, or 'n' -> Next
                save_labels(lbl_path, state["boxes"], w, h)
                idx += 1
                break
            elif key == ord("p"):  # Previous
                save_labels(lbl_path, state["boxes"], w, h)
                idx = max(0, idx - 1)
                break
            elif key == ord("d") and state["boxes"]:  # Undo last box
                state["boxes"].pop()
            elif key == ord("c"):  # Clear ball box
                state["boxes"] = [b for b in state["boxes"] if b[0] != 0]
            elif key == ord("m"):  # Jump to next frame missing a ball
                save_labels(lbl_path, state["boxes"], w, h)
                found_next = False
                for j in range(idx + 1, total):
                    cand_stem = os.path.splitext(os.path.basename(img_files[j]))[0]
                    cand_lbl = os.path.join(lbl_dir, f"{cand_stem}.txt")
                    cand_boxes = load_labels(cand_lbl, w, h)
                    if not any(b[0] == 0 for b in cand_boxes):
                        idx = j
                        found_next = True
                        break
                if not found_next:
                    print("[Info] No more frames with missing balls ahead!")
                break
            elif key in (27, ord("q")):  # ESC or 'q' -> Finish
                save_labels(lbl_path, state["boxes"], w, h)
                idx = total
                break

    cv2.destroyAllWindows()
    print("[Saved] All manual label reviews written to disk.")


def build_split_and_train(
    raw_out_dir, dataset_dir, base_weights, epochs=35, batch_size=2
):
    """
    Step 4 of README:
    1. Split reviewed 2 FPS frames 80% train / 20% val.
    2. Train YOLO26s at 1024x1024 stretched (rect=False) with brightness/exposure jitter.
    3. Use native FP32 (amp=False) on GTX 1650 to prevent FP16 NaN gradient instability.
    """
    img_files = sorted(glob.glob(os.path.join(raw_out_dir, "images", "*.jpg")))
    if len(img_files) < 4:
        raise RuntimeError(
            f"Need at least 4 labeled images in {raw_out_dir}/images to train."
        )

    for split in ("train", "val"):
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    rng = random.Random(42)
    shuffled = list(img_files)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(0.20 * len(shuffled))))
    val_set = set(shuffled[:val_count])

    train_n, val_n = 0, 0
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(raw_out_dir, "labels", f"{stem}.txt")
        split = "val" if img_path in val_set else "train"
        if split == "train":
            train_n += 1
        else:
            val_n += 1

        shutil.copy2(
            img_path, os.path.join(dataset_dir, "images", split, f"{stem}.jpg")
        )
        if os.path.exists(lbl_path):
            shutil.copy2(
                lbl_path, os.path.join(dataset_dir, "labels", split, f"{stem}.txt")
            )

    yaml_path = os.path.abspath(os.path.join(dataset_dir, "tennis_wall.yaml"))
    abs_root = os.path.abspath(dataset_dir).replace("\\", "/")
    yaml_content = (
        f"path: {abs_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: tennis-ball\n"
        f"  1: tennis-racket\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\n" + "=" * 72)
    print(
        f"STEP 4: FINE-TUNING YOLO26s AT 1024x1024 STRETCHED ({train_n} Train / {val_n} Val)"
    )
    print("=" * 72)

    model = YOLO(base_weights)
    results = model.train(
        data=yaml_path,
        imgsz=1024,
        rect=False,  # Stretch to 1024x1024 square rather than letterboxing (Step 4)
        epochs=epochs,
        batch=batch_size,  # Safe for 4GB VRAM GTX 1650 at 1024x1024
        amp=False,  # Native FP32 avoids GTX 1650 TU117 FP16 NaN gradient bugs
        workers=2,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.40,  # Brightness & exposure jitter for outdoor lighting variation
        degrees=0.0,  # Minimal geometric distortion since camera angle is fixed
        mosaic=0.0,  # Disable mosaic so tiny 15px balls aren't shrunk further
        scale=0.15,
        fliplr=0.5,
        project="runs/stride_detect",
        name="yolo26s_1024_stretched",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    best_pt = os.path.join(
        "runs", "stride_detect", "yolo26s_1024_stretched", "weights", "best.pt"
    )
    target_pt = os.path.join("models", "stride_ball_racket_best.pt")
    if os.path.exists(best_pt):
        os.makedirs("models", exist_ok=True)
        shutil.copy2(best_pt, target_pt)
        print("\n" + "=" * 72)
        print(f"[Success!] Fine-tuned weights copied to: {target_pt}")
        print("Run the full pipeline with your custom model using:")
        print(
            f"  python run_pipeline.py --video data/raw_videos/sample.mp4 --det-weights {target_pt} --rerun-cache"
        )
        print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steps 2, 3 & 4: 2FPS Dataset Sampler, Reviewer & 1024x1024 Trainer"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["sample", "review", "train"],
        help="'sample'=Extract 2FPS & auto-label | 'review'=OpenCV box editor | 'train'=Fine-tune 1024x1024",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=["data/raw_videos/sample.mp4"],
        help="List of raw video clips to sample at 2 FPS",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/frames_2fps",
        help="Staging folder for 2 FPS images and labels",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/dataset",
        help="Final train/val YOLO dataset folder",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default="models/yolo26s.pt",
        help="Pretrained YOLO26 weights to initialize from",
    )
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Training batch size (2 for 4GB VRAM @ 1024x1024)",
    )
    args = parser.parse_args()

    if args.mode == "sample":
        sample_and_autolabel(
            args.videos, args.raw_dir, args.base_weights, sample_fps=2.0
        )
    elif args.mode == "review":
        interactive_review_ui(args.raw_dir)
    elif args.mode == "train":
        build_split_and_train(
            args.raw_dir,
            args.dataset_dir,
            args.base_weights,
            epochs=args.epochs,
            batch_size=args.batch,
        )
