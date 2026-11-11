import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def verify_system():
    print("=" * 60)
    print("1. HARDWARE & CUDA CHECK")
    print("=" * 60)
    cuda_ok = torch.cuda.is_available()
    print(f"PyTorch Version : {torch.__version__}")
    print(f"OpenCV Version  : {cv2.__version__}")
    print(f"CUDA Available  : {cuda_ok}")

    if not cuda_ok:
        raise RuntimeError(
            "CUDA is not detected! Make sure you installed the cu124 PyTorch build."
        )

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Device      : {device_name} ({total_vram:.2f} GB Total VRAM)")

    print("\n" + "=" * 60)
    print("2. LOADING PRETRAINED YOLO26 MODELS")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)

    # Load Detection (for ball & racket) and Pose (for player keypoints)
    det_model = YOLO("models/yolo26s.pt")
    pose_model = YOLO("models/yolo26s-pose.pt")

    # Create a dummy 1920x1080 HD frame to simulate raw camera footage
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    print("\n" + "=" * 60)
    print("3. RUNNING 1024x1024 STRETCHED BENCHMARK")
    print("=" * 60)
    # Warmup pass
    _ = det_model.predict(dummy_frame, imgsz=1024, rect=False, half=True, verbose=False)
    _ = pose_model.predict(dummy_frame, imgsz=640, half=True, verbose=False)
    torch.cuda.synchronize()

    # Timed benchmark over 10 frames
    t0 = time.perf_counter()
    for _ in range(10):
        # Classes 32 ('sports ball') and 38 ('tennis racket') exist natively in COCO
        _ = det_model.predict(
            dummy_frame,
            imgsz=1024,
            rect=False,
            half=True,
            classes=[32, 38],
            verbose=False,
        )
        _ = pose_model.predict(dummy_frame, imgsz=640, half=True, verbose=False)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 10.0

    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
    print(
        f"Combined Inference Time : {elapsed * 1000:.1f} ms/frame (~{1.0 / elapsed:.1f} FPS)"
    )
    print(f"Peak GPU Memory Used    : {peak_vram:.1f} MB / {total_vram * 1024:.0f} MB")
    print(
        "\nSetup verified! Ready for Stage 2 (Interactive Wall Calibration & Caching)."
    )


if __name__ == "__main__":
    verify_system()
