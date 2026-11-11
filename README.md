# STRIDE Tennis AI: Monocular Wall-Practice Telemetry & Biomechanics Engine

**STRIDE** (*Stroke Tracking, Rally Intelligence & Dynamic Evaluation*) is an offline, single-camera computer vision (**Monocular CV** *[single-camera 3D vision]*) pipeline that transforms standard smartphone video of tennis wall practice into 3D ball speed telemetry, 2D wall shot-placement maps, and biomechanical posture coaching—running 100% locally on consumer GPUs (tested on an NVIDIA GTX 1650 with 4GB VRAM).

---

## 1. Quick Overview (How This Works)

* **Input:** Video recorded from a single stationary camera ($1080\text{p}$ at $30\text{--}60\text{ FPS}$).
* **Local GPU Perception:** High-resolution $1024 \times 1024$ stretched **YOLO26s** *[real-time object detector]* for tiny, fast-moving tennis ball and racket detection + **YOLO26s-pose** *[human skeletal joint tracker]* for 17-keypoint body tracking.
* **Physics & Signal Processing:** A gravity-biased **Kalman Filter** *[predictive motion tracker]*, a 1D **Z-Score** *[standardized statistical deviation]* depth-proximity wave, **Autocorrelation** *[repeating wave pattern math]* for rally timing, **Parabolic Interpolation** *[3-point sub-frame curve fitting]*, and **Pinhole Camera Calibration** *[focal-length depth geometry]*.
* **Biomechanics Engine:** Evaluates a $220\text{ ms}$ pre-contact loading window to measure peak knee flexion angles ($^\circ$) and **Spine-Normalized Spacing** *[rotation-proof ball-to-body distance]* to generate live coaching cues.
* **AR Video & Dashboard:** Rendered via Python, OpenCV, and SciPy using **Planar Homography** *[2D perspective warp matrix]* and **EMA Smoothing** *[jitter-free moving average]*.

---

## 2. Tech Stack, Models & Core Libraries

| Category | Technology / Model | Role in Pipeline & Plain-English Note |
| :--- | :--- | :--- |
| **Language** | **Python 3.11** | Core programming language running all vision, physics, and rendering scripts. |
| **Environment** | **Miniconda** | Lightweight package & virtual environment manager *[isolated Python sandbox]*. |
| **Deep Learning** | **PyTorch (CUDA 12.4)** | GPU tensor library executing neural networks on NVIDIA hardware. |
| **Ball/Racket Model** | **YOLO26s** (`yolo26s.pt`) | Pretrained & custom fine-tuned object detector run at $1024 \times 1024$ stretched resolution. |
| **Skeleton Model** | **YOLO26s-pose** (`yolo26s-pose.pt`) | Extracts 17 **COCO Keypoints** *[standard human joint coordinates]* per frame. |
| **Computer Vision** | **OpenCV (`cv2`)** | Frame reading, homography warping, interactive GUI tools, and H.264 video encoding. |
| **Signal Processing** | **SciPy (`scipy.signal`)** | Runs **Savitzky-Golay Filtering** *[polynomial curve smoothing]*, peak finding, and autocorrelation. |
| **Numerical Math** | **NumPy** | Fast matrix algebra for Kalman state vectors and 3D pinhole projection. |
| **Audio Analysis** | **Librosa** | **Acoustic Onset Detection** *[sound spike timestamp finder]* for audio-visual impact fusion. |
| **Diagnostics** | **Matplotlib & tqdm** | Generates 3-panel waveform diagnostic charts and live terminal progress bars. |

---

## 3. Project Folder Structure

```text
stride-tennis/
│
├── run_pipeline.py                  # Master 1-command runner executing Pass 0 through Pass 3
├── README.md                        # project documentation
│
├── data/
│   ├── raw_videos/
│   │   └── sample.mp4               # Input raw tennis wall-practice video
│   ├── frames_2fps/                 # Sampled 2 FPS images & YOLO .txt labels (for fine-tuning)
│   │   ├── images/
│   │   └── labels/
│   └── dataset/                     # 80/20 Train/Val split + tennis_wall.yaml config
│
├── models/
│   ├── yolo26s.pt                   # Base pretrained COCO object detector weights
│   ├── yolo26s-pose.pt              # Pretrained 17-keypoint human pose estimation weights
│   └── stride_ball_racket_best.pt   # Custom fine-tuned 1024x1024 2-class detector weights
│
├── cache/
│   ├── sample_wall_quad.json        # Saved 4 wall corners & 3x3 homography matrices
│   ├── sample_detections.json       # Frame-indexed raw ball, racket, and pose detections
│   ├── sample_tracks_and_events.json# Kalman-smoothed trajectory & sub-frame stroke/wall events
│   └── sample_analysis.json         # Calibrated focal length, 3D speeds, and biomechanics
│
├── outputs/
│   ├── sample_diagnostics.png       # 3-panel trajectory, proximity wave & autocorrelation plot
│   ├── sample_metrics.csv           # Spreadsheet of shot speeds, wall UV coords & posture cues
│   └── sample_annotated.mp4         # Final 1080p @ 60 FPS AR video with live telemetry HUD
│
└── src/
    ├── 01_setup_and_cache.py        # Pass 0 (Wall Quad GUI) & Pass 1 (Batched GPU Inference)
    ├── 02_track_and_detect_events.py# Pass 2A (Static Filter, Kalman Tracker, Proximity & Audio)
    ├── 03_calibrate_and_analyze.py  # Pass 2B (Focal Length Solver, 3D Speed & Biomechanics)
    ├── 04_render_ar_video.py        # Pass 3 (1080p AR Overlays, Mini-Map & Speed Bar Chart)
    └── 05_sample_and_finetune.py    # 2 FPS Sampler, OpenCV Label Reviewer & 1024x1024 Trainer
```

---

## 4. Environment Setup & Hardware Verification

Follow these exact steps on Windows using **Anaconda PowerShell Prompt** to build a reproducible GPU environment.

### Step 4.1: Create & Activate the Conda Environment
```powershell
# Navigate to your project drive and create project folders
mkdir data\raw_videos, models, cache, outputs, src

# Create an isolated Python 3.11 environment named 'stride'
conda create -n stride python=3.11 -y
conda activate stride
```

### Step 4.2: Install PyTorch with CUDA 12.4 Support
Installing PyTorch with the explicit `--index-url` ensures you get GPU acceleration instead of the slow CPU-only version:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 4.3: Install Computer Vision, Signal & Audio Packages
```powershell
pip install ultralytics opencv-python scipy numpy matplotlib librosa tqdm soundfile
```

### Step 4.4: Verify GPU & CUDA Detection
Run this one-liner in your terminal to confirm PyTorch sees your NVIDIA GPU:
```powershell
python -c "import torch, cv2, scipy, librosa, ultralytics; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
* **Expected Output:** `CUDA Available: True | GPU: NVIDIA GeForce GTX 1650`

> **(?) Hardware Note — Adapting to Your GPU Specs (Tested on GTX 1650 4GB):**
> * **Why FP32 is Default (`half=False`, `amp=False`):** The Turing GTX 1650 (TU117) lacks specialized Tensor Cores and can produce `NaN` *[Not-a-Number math overflow]* gradients under 16-bit mixed precision (`FP16`). All scripts default to native 32-bit floating-point math (`batch=8` for inference, `batch=2` for training) so the pipeline runs reliably within a $4\text{ GB}$ VRAM ceiling on any GPU.
> * **Have an RTX 20/30/40-Series GPU or $\ge 8\text{ GB}$ VRAM?** You can safely increase `--batch 16` (or `32`) in `run_pipeline.py` and set `half=True` / `amp=True` inside `src/01_setup_and_cache.py` and `src/05_sample_and_finetune.py` for ~2x faster Tensor Core execution.

---

## 5. Core Architecture 

Why does this pipeline work from a single camera when 3D depth ($Z$) is normally invisible in 2D video? Below is every mathematical principle used across the 4 passes.

### Pass 0: 4-Corner Wall Target & Planar Homography (`01_setup_and_cache.py`)
* **The Problem:** The camera views the practice wall at an angle, making a rectangular target zone look like a skewed trapezoid in pixel space.
* **The Principle — Planar Homography** *[projective transformation matrix]*:
  When you click 4 corners on the wall in Frame 0 (Top-Left, Top-Right, Bottom-Right, Bottom-Left), `cv2.getPerspectiveTransform` solves a $3 \times 3$ transformation matrix $H_{\text{wall}}$ that maps any skewed pixel coordinate $(u, v)$ on the wall directly onto a flat $[0, 1] \times [0, 1]$ unit square:
  $$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H_{\text{wall}} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad u_{\text{norm}} = \frac{x'}{w'}, \quad v_{\text{norm}} = \frac{y'}{w'}$$
* **Why We Don't Train a Neural Network to Detect the Wall:** In a static-camera clip, clicking 4 corners once takes 3 seconds and gives zero-jitter sub-pixel accuracy, whereas a bounding-box detector cannot capture 3D perspective tilt.

---

### Pass 1: Stretched $1024 \times 1024$ Inference & JSON Caching (`01_setup_and_cache.py`)
* **The Problem:** In a $1920 \times 1080$ video frame, a tennis ball near the player is $\approx 24\text{ px}$ wide and shrinks to $\approx 10\text{ px}$ at the wall. Standard YOLO models resize images to $640 \times 640$ with **Letterboxing** *[black padding bars]*, shrinking the ball to $\approx 6\text{--}8\text{ px}$ where convolutional layers lose it completely.
* **The Principle — Aspect-Ratio Stretched High-Res Inference** (`imgsz=1024, rect=False`):
  Stretching the $1920 \times 1080$ frame into a full $1024 \times 1024$ square preserves vertical resolution ($\approx 95\%$ of $1080\text{p}$) without wasting a third of the tensor on black letterbox bars.
* **Cache-First Design:** GPU inference takes $\approx 90\text{ seconds}$ per clip, while tracking and math take $< 3\text{ seconds}$. Saving all raw bounding boxes and 17 pose keypoints to `cache/{clip}_detections.json` means you never have to wait for the GPU again while tuning downstream physics or overlays.

---

### Pass 2A: Static Suppression, Kalman Tracking & 1D Depth Signal (`02_track_and_detect_events.py`)

#### 1. Pre-Tracking Static Scenery Filter & Physical Size Gate
* **Static Bucket Suppression:** Pale marks on the wall or court background often look like tennis balls to a neural network. Before tracking starts, the script divides the frame into $8 \times 8\text{ px}$ spatial buckets. Any bucket holding detections for $\ge 0.8\text{ seconds}$ is flagged as static scenery and deleted.
* **Physical Size Gate:** In $1080\text{p}$ footage, a tennis ball cannot exceed $45\text{ px}$ in minor-axis diameter. Dropping boxes with $\min(w, h) > 45\text{ px}$ eliminates false positives on white tennis shoes or racket heads.

#### 2. 4-Rule Asymmetric Kalman Filter with Gravity Bias
A **Kalman Filter** *[linear quadratic state estimation]* maintains a 4D state vector $\mathbf{x}_k = [c_x, c_y, v_x, v_y]^T$ (position and velocity) and predicts where the ball should appear in the next frame, governed by four domain rules:
1. **Nearest-to-Prediction Association:** Associates candidate detections by shortest Euclidean distance to the predicted position rather than highest detector confidence.
2. **Asymmetric Confidence Hysteresis** *[dual-threshold state locking]*: Starting a new track requires a high-confidence detection ($\text{conf} \ge 0.50$). Continuing an already-locked track accepts faint, motion-blurred detections down to $\text{conf} \ge 0.12$.
3. **Gravity-Biased Coasting:** If the ball disappears in motion blur during racket contact, the filter coasts on its prediction for up to $8\text{ frames}$ while applying a downward gravitational acceleration prior ($+g_y$).
4. **Frame-Bounds Termination:** If a coasting prediction exits the $1920 \times 1080$ frame boundaries, the track resets immediately.

#### 3. Blur-Compensated Minor-Axis Ball Sizing
* **The Problem:** During fast flight, shutter **Motion Blur** *[exposure streak along motion]* stretches the ball into an elongated pill shape along its velocity vector, artificially inflating its bounding-box width or height.
* **The Principle:** Whenever ball speed exceeds $8\text{ px/frame}$, the script uses $\min(w, h)$ (the minor axis perpendicular to motion, which is unaffected by directional blur) as the true apparent pixel diameter $d$.

#### 4. Zero-Phase Savitzky-Golay Trajectory Smoothing
* Bridges tiny $1\text{--}4\text{ frame}$ coasting gaps via linear interpolation and runs a bidirectional **Savitzky-Golay Filter** *[local polynomial least-squares smoothing]* (`window_length=7, polyorder=2`) over $(c_x, c_y, d)$ to remove single-frame bounding-box jitter while preserving sharp turnaround peaks.

#### 5. 1D Depth-Proximity Wave & Autocorrelation Rally Period
* **Combining Two Monocular Depth Cues:** Because the camera sits behind and slightly beside the player:
  * **Near the Racket:** The ball is **low in the image** (large vertical pixel coordinate $c_y$) and **physically close to the lens** (large pixel diameter $d$).
  * **At the Wall:** The ball is **high in the image** (small $c_y$) and **far from the lens** (small $d$).
* **Z-Score Fusion:** Standardizing both signals into dimensionless **Z-Scores** *[zero-mean unit-variance normalization]* and adding them together yields a clean 1D Depth-Proximity wave $P(t)$:
  $$z_y(t) = \frac{c_y(t) - \mu_y}{\sigma_y}, \quad z_d(t) = \frac{d(t) - \mu_d}{\sigma_d}, \quad P(t) = z_y(t) + z_d(t)$$
  * **Local Maxima (Peaks) of $P(t)$** $=$ Racket Strokes.
  * **Local Minima (Troughs) of $P(t)$** $=$ Wall Impacts.
* **Dynamic Rally Period via Autocorrelation:** Rather than hardcoding how fast a player rallies, the script computes the normalized **Autocorrelation** *[signal cross-correlated with itself]* of $P(t)$. The first prominent peak between $0.6\text{ s}$ and $3.0\text{ s}$ gives the clip's natural rally period $T_{\text{rally}}$ (e.g., $1.58\text{ s}$).

#### 6. 3-Point Parabolic Sub-Frame Timing & Active Audio-Visual Fusion
* **Sub-Frame Parabolic Fit:** Even at $60\text{ FPS}$, the ball travels $\approx 25\text{ cm}$ between frames. Fitting a parabola through the 3 samples $(s_{-1}, s_0, s_{+1})$ around each peak locates the exact sub-frame turnaround timestamp $t^*$:
  $$t^* = k + \frac{s_{k-1} - s_{k+1}}{2(s_{k-1} - 2s_k + s_{k+1})}$$
* **Active Audio-Visual Fusion:** `librosa` scans the audio stream (`hop_length=256`, $\approx 11.6\text{ ms}$ resolution) for acoustic onset spikes (racket string hits and wall thuds). If an acoustic transient occurs within $\pm 2.5\text{ frames}$ of $t^*$, the timestamp snaps toward the audio spike ($55\%$ audio + $45\%$ visual); if the video has no audio, it falls back seamlessly to `[VIS-ONLY]`.

---

### Pass 2B: Camera Calibration, 3D Speed & Biomechanics (`03_calibrate_and_analyze.py`)

#### 1. Monocular Focal Length Solver ($f$)
* **Pinhole Camera Law** *[pinhole projection formula]*: An object of real-world diameter $D_{\text{ball}} = 0.067\text{ m}$ ($6.7\text{ cm}$ standard tennis ball) at depth $Z$ projects to pixel size $d = \frac{D_{\text{ball}} \cdot f}{Z}$, where $f$ is the camera's focal length in pixels.
* Given the measured player-to-wall distance $\Delta Z_{\text{wall}} = Z_{\text{wall}} - Z_{\text{racket}}$ (e.g., $6.5\text{ m}$) and the clip-wide **median** ball pixel diameters at racket contact ($d_{\text{racket}}$) and wall impact ($d_{\text{wall}}$), we solve directly for $f$:
  $$D_{\text{ball}} \cdot f \left(\frac{1}{d_{\text{wall}}} - \frac{1}{d_{\text{racket}}}\right) = \Delta Z_{\text{wall}} \implies f = \frac{\Delta Z_{\text{wall}}}{D_{\text{ball}} \left(\frac{1}{d_{\text{wall}}} - \frac{1}{d_{\text{racket}}}\right)}$$
* **Why Medians Matter:** Single-frame ball diameter is too noisy to differentiate frame-by-frame, but taking the median across all rally contacts cancels out bounding-box noise and pins down $f$, $Z_{\text{racket}}$, and $Z_{\text{wall}}$.

#### 2. Outbound 3D Shot Speed (Without Depth Differentiation)
* Instead of estimating noisy depth frame-by-frame, we know the ball traveled exactly $\Delta Z = \Delta Z_{\text{wall}}$ in depth between the stroke timestamp $t_{\text{stroke}}^*$ and wall impact timestamp $t_{\text{impact}}^*$.
* Back-projecting the image-plane displacements $(\Delta X, \Delta Y)$ at the calibrated depths gives the total 3D flight distance and outbound shot speed in $\text{km/h}$:
  $$\Delta X = \frac{(u_{\text{wall}} - c_x) Z_{\text{wall}} - (u_{\text{racket}} - c_x) Z_{\text{racket}}}{f}, \quad \Delta Y = \frac{(v_{\text{wall}} - c_y) Z_{\text{wall}} - (v_{\text{racket}} - c_y) Z_{\text{racket}}}{f}$$
  $$v_{\text{outbound}} = \frac{\sqrt{(\Delta X)^2 + (\Delta Y)^2 + (\Delta Z_{\text{wall}})^2}}{t_{\text{impact}}^* - t_{\text{stroke}}^*} \times 3.6\text{ (km/h)}$$

#### 3. 3D-Aware Torso-Spine & Hitting-Wrist Stroke Classifier
* Combines **Shoulders (`kp 5, 6`)**, **Hips (`kp 11, 12`)**, and the **Dominant Hitting Wrist (`kp 10` or `kp 9`)** with the ball's contact position (`ball_x`) relative to the torso centerline (`torso_center_x`).
* Uses the `tennis-racket` bounding box as a tie-breaker when a shot is struck within $14\text{ px}$ of the body midline.

#### 4. Pre-Contact Knee Flexion & Spine-Normalized Spacing (Coach Cues)
* **Peak Knee Load ($220\text{ ms}$ Window):** Players bend their knees deepest *before* racket contact and push upward as they hit. The script scans the $220\text{ ms}$ window prior to `stroke_frame`, computes the interior 2D vector angle across the **Hip–Knee–Ankle** joints using the dot-product law $\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|\|\vec{v}\|}\right)$, and records the 15th-percentile minimum angle (`knee_angle_deg`).
* **Rotation-Invariant Spine Normalization:** When a player rotates sideways into a **Unit Turn** *[coiling shoulders perpendicular to net]*, their 2D horizontal shoulder width shrinks toward zero. Normalizing horizontal ball spacing (`|ball_x - torso_center_x|`) by the **Vertical Torso Spine Length** ($\|\text{Mid}_{\text{shoulder}} - \text{Mid}_{\text{hip}}\|$) provides a rotation-proof spacing ratio (`rel_spacing_ratio`):
  * `Knee > 158 deg` $\rightarrow$ *"Bend knees more in prep (stiff legs)"*
  * `Spacing < 0.55x` $\rightarrow$ *"Jammed close to body (create spacing)"*
  * `Spacing > 1.55x` $\rightarrow$ *"Reaching outside base (move feet)"*
  * `Knee <= 138 deg` $\rightarrow$ *"Strong knee load & unit turn"*

---

### Pass 3: Smooth $1080\text{p}$ AR Rendering (`04_render_ar_video.py`)
* **EMA Skeleton Filter:** Applies an **Exponential Moving Average** *[weighted blend with previous frame]* ($\alpha = 0.45$) to the 17 pose keypoints so the green skeletal overlay glides smoothly without $60\text{ FPS}$ detector vibration.
* **Hardware H.264 Encoding (`CAP_MSMF` + `avc1`):** Uses Windows Media Foundation hardware encoding to write smooth $60\text{ FPS}$ H.264 `.mp4` streams at $\approx 50\text{ FPS}$.

---

## 6. How to Run the Pipeline (Command-Line Guide)

Make sure your Conda environment is active (`conda activate stride`) and you are in `D:\stride-tennis`.

### Option A: One-Command Master Pipeline (`run_pipeline.py`)

**1. First Run on a Video (Interactive Wall Click + Full GPU Cache + Render):**
```powershell
python run_pipeline.py --video data/raw_videos/sample.mp4 --wall-dist 6.5 --det-weights models/stride_ball_racket_best.pt
```
* **What happens on Pass 0:** A window pops up showing Frame 0. Click the **4 corners of your wall target zone** in clockwise order (**1: Top-Left**, **2: Top-Right**, **3: Bottom-Right**, **4: Bottom-Left**), then press **`Enter`** or **`Space`** (press **`r`** to reset if you misclick).

**2. Fast Re-Run Using Cached Detections (Skips 90s GPU Pass):**
Once `cache/sample_detections.json` exists, running the exact same command skips GPU inference automatically and finishes in $\approx 28\text{ seconds}$.

**3. Instant Math & Biomechanics Tuning (Skips Video Rendering — Takes $< 4\text{ seconds}$):**
```powershell
python run_pipeline.py --video data/raw_videos/sample.mp4 --wall-dist 6.5 --skip-render
```

**4. Force Re-Selecting the 4 Wall Target Corners:**
```powershell
python run_pipeline.py --video data/raw_videos/sample.mp4 --wall-dist 6.5 --reset-wall
```

**5. Force Re-Running GPU Inference (e.g., After Fine-Tuning New Weights):**
```powershell
python run_pipeline.py --video data/raw_videos/sample.mp4 --wall-dist 6.5 --det-weights models/stride_ball_racket_best.pt --rerun-cache
```

### Option B: Running Individual Stage Scripts Manually
If you want to inspect or debug one pass at a time:
```powershell
# Pass 0 & Pass 1: Wall corner selection + batched GPU caching
python src/01_setup_and_cache.py --video data/raw_videos/sample.mp4 --det-weights models/stride_ball_racket_best.pt

# Pass 2A: Static suppression, Kalman tracking, proximity signal & diagnostic plot
python src/02_track_and_detect_events.py --video data/raw_videos/sample.mp4

# Pass 2B: Camera calibration, 3D speed & biomechanical posture analysis
python src/03_calibrate_and_analyze.py --video data/raw_videos/sample.mp4 --wall-dist 6.5 --handedness right

# Pass 3: Render 1080p AR video & telemetry HUD
python src/04_render_ar_video.py --video data/raw_videos/sample.mp4
```

---

## 7. Custom Detector Fine-Tuning Guide (`05_sample_and_finetune.py`)

Pretrained COCO weights (`yolo26s.pt`) often miss motion-blurred balls near racket impact or confuse white shoes with tennis balls. Fine-tuning on just $40\text{--}150$ frames sampled at $2\text{ FPS}$ boosts ball recall to $100\%$ (`0.960 mAP50`) and eliminates false positives.

### Step 7.1: Sample Frames at 2 FPS & Auto-Label (`--mode sample`)
Consecutive $60\text{ FPS}$ frames are nearly identical; sampling at $2\text{ FPS}$ captures distinct ball positions while pre-labeling boxes and dropping static wall marks automatically:
```powershell
python src/05_sample_and_finetune.py --mode sample --videos data/raw_videos/sample.mp4
```

### Step 7.2: Interactive OpenCV Label Review (`--mode review`)
Opens a fast annotation GUI so you can draw boxes on motion-blurred balls missed by the pretrained model:
```powershell
python src/05_sample_and_finetune.py --mode review
```
* **Keyboard & Mouse Controls inside the Review Window:**
  * **`m` key:** Jump directly to the next frame missing a ball (`MISSING BALL` red banner).
  * **Left-Click & Drag:** Draw a green `tennis-ball` box (automatically replaces any old ball box on that frame).
  * **Right-Click & Drag:** Draw an orange `tennis-racket` box.
  * **`c` key:** Clear the ball box on the current frame.
  * **`d` key:** Undo/delete the last drawn box.
  * **`Space` / `Enter` / `n`:** Save labels and advance to the next frame.
  * **`p` key:** Save labels and go back to the previous frame.
  * **`q` / `ESC`:** Save all changes and exit.

### Step 7.3: Train at $1024 \times 1024$ Stretched Resolution (`--mode train`)
Splits your reviewed dataset $80\%$ train / $20\%$ val and fine-tunes `yolo26s.pt` at $1024 \times 1024$ stretched (`rect=False`, `mosaic=0.0`, `amp=False` for GTX 1650 stability):
```powershell
python src/05_sample_and_finetune.py --mode train --epochs 35 --batch 2
```
Once finished, copy the best checkpoint into `models/` (if Ultralytics nested the output path) and re-run the pipeline:
```powershell
Copy-Item "runs\detect\runs\stride_detect\yolo26s_1024_stretched\weights\best.pt" -Destination "models\stride_ball_racket_best.pt" -Force
python run_pipeline.py --video data/raw_videos/sample.mp4 --wall-dist 6.5 --det-weights models/stride_ball_racket_best.pt --rerun-cache
```

---

## 8. Understanding the Output Files

Every pipeline run produces four synchronized artifacts inside `outputs/` and `cache/`:

1. **`outputs/{clip}_annotated.mp4`:** Full $1080\text{p}$ AR video featuring:
   * **3D Perspective Wall Target Grid:** Flashes green on wall impact and pins numbered, color-coded landing markers (**Cyan = Forehand**, **Orange = Backhand**).
   * **Top-Left Telemetry Card:** Live stroke counter, Forehand/Backhand split, last shot speed ($\text{km/h}$), peak knee load angle ($^\circ$), contact spacing ratio, and live coaching cue.
   * **Top-Right 2D Wall Target Mini-Map:** Warps wall impacts via $H_{\text{wall}}$ onto a flat square showing shot clustering as the wall sees it.
   * **Bottom-Right Live Speed Bar Chart:** Plots each shot's outbound speed in $\text{km/h}$ as bars build across the clip.
2. **`outputs/{clip}_diagnostics.png`:** A 3-panel scientific verification chart:
   * *Panel 1:* Kalman-tracked vertical ball trajectory $c_y(t)$ (inverted so top $=$ near wall, bottom $=$ near racket).
   * *Panel 2:* Smoothed 1D Depth-Proximity wave $P(t) = z_y + z_d$ with red markers on validated strokes and blue markers on validated wall impacts.
   * *Panel 3:* Autocorrelation curve showing the dynamically detected rally period peak $T_{\text{rally}}$.
3. **`outputs/{clip}_metrics.csv`:** Spreadsheet containing per-shot telemetry (`shot_number`, `stroke_time_sec`, `impact_time_sec`, `flight_time_sec`, `audio_verified`, `stroke_type`, `speed_kmh`, `wall_target_uv`, `inside_target_quad`, `unit_turn_ratio`, `rel_spacing_ratio`, `knee_angle_deg`, `posture_note`).
4. **`cache/{clip}_analysis.json`:** Complete structured JSON output including camera focal length $f$, median ball sizes, racket/wall metric depths ($Z_{\text{racket}}, Z_{\text{wall}}$), and player crouch depth sanity-check metrics.

---

## 9. Troubleshooting & Common Questions

* **Why did `[libopenh264] Unable to create encoder` appear in the terminal?**
  * This is a harmless C++ message from OpenCV checking its bundled FFmpeg backend before switching to Windows Media Foundation (`CAP_MSMF`). As long as you see `[Pass 3] Initialized VideoWriter with codec: 'avc1'`, hardware H.264 encoding is active. To silence the probe message completely, ensure `cv2.CAP_MSMF` is listed first in `create_smooth_video_writer()` inside `src/04_render_ar_video.py`.
* **Why does `Audio-Visual Fused Contacts` show `0/11 [VIS-ONLY]`?**
  * If your input `.mp4` has no audio stream (muted clip), `extract_audio_onsets()` safely catches the empty stream and falls back $100\%$ to 3-point parabolic visual sub-frame interpolation (`[VIS-ONLY]`) without crashing.
* **How do I adjust the speed calibration if the player stood closer or farther from the wall?**
  * Outbound speed scales linearly with `--wall-dist`. Measure or estimate the distance in meters from the player's baseline to the wall (e.g., `--wall-dist 5.5` or `--wall-dist 7.5`) and run `python run_pipeline.py --video data/raw_videos/sample.mp4 --wall-dist 5.5`—it will recalculate all speeds in $< 1\text{ second}$ using your cached detections.