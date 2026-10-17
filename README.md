# Vehicle and Pedestrian Traffic Analysis System

## 1. Introduction

This project implements a real-time, zone-aware vehicle and pedestrian traffic analysis system in native C++, combining a VisDrone-trained YOLO26x detector with a from-scratch BoT-SORT multi-object tracker. The system processes intersection footage to count, classify, and track road users across ten manually defined zones (crosswalks, sidewalks, and road surfaces), producing a live analytics dashboard, an annotated output video, a per-second CSV time series, and a JSON summary report.

The implementation is fully self-contained and dependency-light: detection is performed through OpenCV's DNN module on an ONNX-exported model, and tracking is a hand-written reimplementation of the BoT-SORT algorithm, including the Kalman filter, Global Motion Compensation (GMC), the Hungarian assignment algorithm, and the cascaded high/low-confidence matching strategy that BoT-SORT uses. The result is a single executable with no external machine-learning framework runtime required.

On top of detection and tracking, the system adds traffic-specific analysis: per-zone occupancy and dwell-time statistics, motion-based moving/stationary classification, and a crosswalk safety mechanism that flags a crossing red the instant a motorized vehicle's box touches it, or blue when pedestrians clearly dominate the crossing.

**Core Features:**
- Real-time detection of 10 VisDrone-DET classes (2 person types, 8 vehicle types) via OpenCV DNN
- From-scratch BoT-SORT tracker: Kalman filtering, GMC camera-motion compensation, cascaded IoU matching, optional ReID hook
- 10-zone ROI system (3 crosswalks, 4 road surfaces, 3 sidewalks) with automatic scaling to non-reference resolutions
- Per-zone unique counts, peak instantaneous occupancy, and average dwell time (pedestrian and vehicle)
- Moving/stationary classification per tracked object via displacement-window analysis
- Crosswalk safety masking: red on vehicle intrusion, blue when moving pedestrians dominate
- Live multi-panel dashboard: per-category pie charts (crosswalk/sidewalk by sub-zone, road by vehicle type)
- Annotated video, per-second CSV time series, and JSON summary export
- Interactive ROI calibration mode for defining zones on a new video

## 2. Methodology / Approach

The system combines a single-pass object detector with a dedicated multi-object tracker, then layers spatial zone logic and motion analysis on top of the tracked trajectories.

### 2.1 System Architecture

```
[Video Input]
    ↓
[Letterbox Preprocess] → [YOLO26x Inference (OpenCV DNN)] → [Class-wise NMS]
    ↓
[BoT-SORT Tracking] → [Kalman Predict] → [GMC Warp] → [Cascaded IoU/ReID Matching]
    ↓
[Zone Assignment (Point-in-Polygon)] → [Motion Classification] → [Crosswalk Safety Logic]
    ↓
[Dashboard Panel Rendering] → [Video Compositing]
    ↓
[Annotated Video Output] + [CSV Time Series] + [JSON Summary]
```

### 2.2 Detection Pipeline

The detector (`Detector.hpp/.cpp`) runs `visdrone.onnx` (an Ultralytics YOLO26x model fine-tuned for 50 epochs on the VisDrone dataset) through OpenCV's `cv::dnn` module. Frames are letterboxed to the model's training resolution (1280×1280), passed through the network, and the raw `[1, 4+numClasses, numBoxes]` output (box coordinates + per-class sigmoid scores, no built-in NMS) is decoded and filtered with class-wise Non-Maximum Suppression before boxes are mapped back to original-frame coordinates.

### 2.3 Tracking Pipeline

`BotSort` (`BotSort.hpp/.cpp`) implements the BoT-SORT tracking algorithm, using the exact defaults of `botsort.yaml`:

1. **Kalman Prediction:** An 8-state constant-velocity filter (`KalmanFilterXYWH`) predicts each existing track's next `(x, y, w, h)`.
2. **Global Motion Compensation:** `GMC` estimates a 2×3 affine camera-motion warp via sparse optical flow and applies it to all track states before association, so a panning camera isn't mistaken for object motion.
3. **Cascaded Matching:** High-confidence detections are matched to tracks first via IoU (fused with detection confidence), then low-confidence detections are matched against the remaining tracks (ByteTrack-style recovery), and finally unmatched high-confidence detections spawn new tracks.
4. **Assignment:** `Matching::linearAssignment` solves each cost matrix with a from-scratch Hungarian algorithm (`Hungarian.cpp`), rejecting pairs above the configured distance threshold.
5. **ReID Hook:** An optional appearance-embedding hook (`setReidEncoder`) is fully wired but inactive by default (`withReid = false`), consistent with BoT-SORT's default configuration.

### 2.4 Zone & Motion Analysis

Each track's ground-contact point (bounding-box bottom-center) is tested against 10 ROI polygons using `cv::pointPolygonTest`; the first matching zone wins, with crosswalks listed ahead of the generic road/sidewalk zones so shared borders aren't double-counted. Separately, each track's exact box center is compared against its own recent history (`MOTION_WINDOW` frames) to classify it as moving or stationary based on average displacement.

### 2.5 Crosswalk Safety Logic

For each of the three crosswalk zones, the system evaluates, per frame:
- **Blue (pedestrian priority):** pedestrian count exceeds motorized-vehicle count by `PEDESTRIAN_OVERRIDE_MARGIN` or more, or `MIN_MOVING_PEDESTRIANS`+ pedestrians are actively moving inside the zone.
- **Red (vehicle intrusion):** any motorized vehicle's bounding box touches the crosswalk polygon's border or interior (tested via a full segment-intersection + containment check, not just the ground point).

Bicycles are excluded from the "motorized" check, since they don't constitute a vehicle-intrusion hazard on a crossing in the same way.

### 2.6 Live Dashboard

A dashboard strip is rendered above each output frame with three columns (CROSSWALK, SIDEWALK, ROAD), each showing one pie chart: the crosswalk/sidewalk columns split their live pedestrian count across sub-zones, while the road column pools every road zone's vehicles by type (roads carry negligible foot traffic in this scene, so pedestrian data is omitted there).

## 3. Mathematical Framework

### 3.1 Letterbox Preprocessing

Input frames are resized with aspect ratio preserved and padded to a square:

$$\text{gain} = \min\left(\frac{\text{imgSize}}{w}, \frac{\text{imgSize}}{h}\right), \quad \text{pad}_x = \frac{\text{imgSize} - w \cdot \text{gain}}{2}, \quad \text{pad}_y = \frac{\text{imgSize} - h \cdot \text{gain}}{2}$$

### 3.2 Kalman Filter (Constant-Velocity, xywh)

State vector $\mathbf{x} = [x, y, w, h, v_x, v_y, v_w, v_h]^T$ evolves under a linear motion model:

$$\mathbf{x}_{t} = \mathbf{F}\mathbf{x}_{t-1}, \quad \mathbf{F} = \begin{bmatrix} \mathbf{I}_4 & \mathbf{I}_4 \\ \mathbf{0}_4 & \mathbf{I}_4 \end{bmatrix}$$

$$\mathbf{P}_{t} = \mathbf{F}\mathbf{P}_{t-1}\mathbf{F}^T + \mathbf{Q}$$

Measurement update projects the state to the observation space $(x, y, w, h)$ and applies the standard Kalman gain correction:

$$\mathbf{K} = \mathbf{P}\mathbf{H}^T(\mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R})^{-1}, \quad \mathbf{x}' = \mathbf{x} + \mathbf{K}(\mathbf{z} - \mathbf{H}\mathbf{x})$$

### 3.3 IoU Distance & Score Fusion

$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}, \quad \text{cost}_{\text{IoU}} = 1 - \text{IoU}$$

Detection confidence is fused into the cost matrix before assignment:

$$\text{sim}_{\text{fuse}} = (1 - \text{cost}_{\text{IoU}}) \times \text{score}_{\text{det}}, \quad \text{cost}_{\text{fused}} = 1 - \text{sim}_{\text{fuse}}$$

### 3.4 Linear Assignment (Hungarian Algorithm)

Given cost matrix $\mathbf{C} \in \mathbb{R}^{m \times n}$, the Hungarian algorithm finds the assignment $\sigma$ minimizing total cost:

$$\sigma^* = \arg\min_{\sigma} \sum_{i=1}^{\min(m,n)} C_{i, \sigma(i)}$$

Pairs with $C_{i,\sigma(i)} > \tau_{\text{match}}$ are rejected and returned to the unmatched pools (equivalent to `lap.lapjv` with a cost limit).

### 3.5 Global Motion Compensation

A 2×3 affine warp $\mathbf{W}$ estimated from sparse optical flow between consecutive frames is applied to each track's mean and covariance before matching:

$$\mathbf{p}' = \mathbf{W} \begin{bmatrix} \mathbf{p} \\ 1 \end{bmatrix}, \quad \mathbf{W} \in \mathbb{R}^{2 \times 3}$$

### 3.6 Point-in-Polygon Zone Test

A track's ground-contact point $P = \left(\frac{x_1+x_2}{2}, y_2\right)$ is tested against each zone polygon via OpenCV's ray-casting `pointPolygonTest`, returning the first zone (in priority order) for which the point lies inside or on the boundary.

### 3.7 Box-Polygon Intersection Test

Crosswalk vehicle intrusion is detected as any of: a box edge crossing a polygon edge (orientation + on-segment test), a polygon vertex inside the box, or a box corner inside the polygon:

$$\text{Intersects}(B, \text{poly}) = \bigvee_{i,j} \text{SegmentsIntersect}(e_i^B, e_j^{\text{poly}}) \; \lor \; \exists\, v \in \text{poly}: v \in B \; \lor \; \exists\, c \in \text{corners}(B): c \in \text{poly}$$

### 3.8 Motion Classification

A track is classified as moving when the mean displacement between the first and second halves of its recent position history exceeds a threshold:

$$\bar{P}_{\text{old}} = \frac{1}{k}\sum_{i=1}^{k} P_i, \quad \bar{P}_{\text{new}} = \frac{1}{k}\sum_{i=n-k}^{n} P_i, \quad \text{Moving} = \lVert \bar{P}_{\text{new}} - \bar{P}_{\text{old}} \rVert \geq \tau_{\text{motion}}$$

where $k = \lceil w / 2 \rceil$ (`w` = `MOTION_WINDOW`) and $\tau_{\text{motion}} = 1.5$ px.

## 4. Dataset & Model

**Model:** `visdrone.onnx`: Ultralytics YOLO26x, fine-tuned for 50 epochs on the VisDrone-DET dataset, mAP50 61.8%, exported to ONNX for inference through OpenCV's DNN module (no PyTorch/ultralytics runtime dependency).

**Classes (VisDrone-DET, not COCO):**

| Index | Class | Group |
|-------|-------|-------|
| 0 | pedestrian | Person |
| 1 | people | Person |
| 2 | bicycle | Vehicle |
| 3 | car | Vehicle |
| 4 | van | Vehicle |
| 5 | truck | Vehicle |
| 6 | tricycle | Vehicle |
| 7 | awning-tricycle | Vehicle |
| 8 | bus | Vehicle |
| 9 | motor | Vehicle |

**ROI Zones (reference resolution 1920×1080, auto-scaled otherwise):**

| Zone | Type |
|------|------|
| CROSSWALK_S, CROSSWALK_NW, CROSSWALK_NE | Zebra-striped crosswalk |
| ROAD_MID, ROAD_SW, ROAD_E, ROAD_NW | Open road/intersection surface |
| SIDEWALK_E, SIDEWALK_W, SIDEWALK_N | Pedestrian-only sidewalk |

## 5. Requirements

**Build Tools:**
```txt
CMake >= 3.16
C++17-compliant compiler (Clang, GCC, or MSVC)
```

**Libraries:**
```txt
OpenCV (components: core, imgproc, imgcodecs, videoio, dnn, video, calib3d, highgui)
```

**Model Weights:**
```txt
visdrone.onnx  # ONNX export of the fine-tuned YOLO26x VisDrone model
```

## 6. Installation & Configuration

### 6.1 Environment Setup

```bash
# Clone the repository
git clone https://github.com/kemalkilicaslan/Vehicle-and-Pedestrian-Traffic-Analysis-System.git
cd Vehicle-and-Pedestrian-Traffic-Analysis-System

# Install OpenCV (macOS example via Homebrew)
brew install opencv cmake
```

### 6.2 Project Structure

```
Vehicle-and-Pedestrian-Traffic-Analysis-System/
├── src/
│   ├── main.cpp                # Pipeline orchestration, ROI/zone logic, dashboard, I/O
│   ├── Detector.cpp             # OpenCV DNN inference + letterbox + NMS
│   ├── BotSort.cpp              # BoT-SORT tracker (cascaded matching)
│   ├── KalmanFilterXYWH.cpp     # Constant-velocity Kalman filter
│   ├── STrack.cpp               # Per-track state (Kalman mean/covariance, ReID features)
│   ├── Matching.cpp             # IoU distance, score fusion, embedding distance
│   ├── GMC.cpp                  # Global Motion Compensation (sparse optical flow)
│   └── Hungarian.cpp            # Hungarian algorithm for linear assignment
├── include/                     # Corresponding headers
├── CMakeLists.txt
├── Dockerfile                   # container image build recipe (CMake + OpenCV)
├── .dockerignore                # build context exclusions
├── docker-compose.yml           # container run configuration
├── visdrone.onnx                # Model weights (place here)
├── .gitignore
├── README.md
└── LICENSE
```

### 6.3 Building

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
```

This produces the `vehicle-and-pedestrian-traffic-analysis-system` executable in `build/`.

## 7. Usage / How to Run

### 7.1 Basic Execution

Place the input video (`Vehicle-and-Pedestrian-Traffic.mp4`) and the model weights (`visdrone.onnx`) next to the executable, then run:

```bash
./build/vehicle-and-pedestrian-traffic-analysis-system
```

### 7.2 ROI Calibration Mode

Before analyzing a new camera angle, set `CHECK_ROI = true` in `main.cpp` and rebuild. The program draws the current ROI polygons on the video's first frame, saves a PNG preview, and opens an interactive window; left-clicking any point prints its pixel coordinates to the console for updating `makeRoi()`.

### 7.3 Configuration Parameters

```cpp
// Detection
static const float CONF_THRESHOLD = 0.0f;   // low value feeds BoT-SORT's low-score matching
static const float IOU_THRESHOLD  = 0.3f;    // NMS IoU threshold
static const int   IMG_SIZE       = 1280;    // must match training resolution
static const std::string DEVICE   = "auto";  // auto | cpu | cuda | opencl

// Zone / motion behavior
static const double MOTION_THRESH = 1.5;     // px displacement to be classed as "moving"
static const size_t MOTION_WINDOW = 5;       // frames used to judge motion
static const int MIN_MOVING_PEDESTRIANS = 3; // triggers blue crosswalk mask
static const int PEDESTRIAN_OVERRIDE_MARGIN = 5; // pedestrian lead that forces blue over red

// Processing
static const int STRIDE      = 1;    // process every Nth frame
static const int MAX_FRAMES  = 0;    // 0 = whole video
static const bool SHOW_PREVIEW = true;
static const bool CHECK_ROI    = false;
```

### 7.4 Controls

- Press `q` in the preview window to stop processing (outputs saved so far are kept).
- Press `q` or `Esc` in ROI calibration mode to close the preview.

## 8. Docker Deployment

The system is a C++ / OpenCV application. It can be compiled and run inside a
container so that the toolchain, OpenCV and the runtime environment stay
consistent across machines.

### Files

- `Dockerfile` - builds OpenCV toolchain image, compiles the CMake project and installs the binary.
- `.dockerignore` - keeps the build directory and large media/model files out of the build context.
- `docker-compose.yml` - runs the compiled binary and mounts the project folder for input/output.

### Build and run

```bash
docker compose up --build

# or with the Docker CLI directly
docker build -t vehicle-pedestrian-traffic-analysis:latest .
docker run --rm -v "$(pwd)":/data vehicle-pedestrian-traffic-analysis:latest
```

### Notes

- The input video (`Vehicle-and-Pedestrian-Traffic.mp4`) and the ONNX model
  (`visdrone.onnx`) are read from the mounted project folder. The binary writes
  its outputs (`*-Analysis.mp4`, `*-timeseries.csv`, `*-summary.json`) next to the
  input video, so results appear on the host after the run.
- The binary is installed to `/usr/local/bin` inside the image, while the working
  directory `/data` is the mounted project folder; this keeps the compiled binary
  from being hidden by the bind mount.
- The application uses OpenCV highgui (`imshow`), which needs a graphical display.
  On a headless server the output files are still written to disk; the live preview
  window is only available on Linux with X11 forwarding (see the commented lines in
  `docker-compose.yml`).

## 9. Application / Results

### 9.1 Input Video

[![Vehicle and Pedestrian Traffic](https://github.com/kemalkilicaslan/Vehicle-and-Pedestrian-Traffic-Analysis-System/blob/main/Vehicle-and-Pedestrian-Traffic.webp)](https://www.youtube.com/watch?v=Mt8o7vzJwfw)

### 9.2 Output Video

[![Vehicle and Pedestrian Traffic Analysis](https://github.com/kemalkilicaslan/Vehicle-and-Pedestrian-Traffic-Analysis-System/blob/main/Vehicle-and-Pedestrian-Traffic-Analysis-System.webp)](https://www.youtube.com/watch?v=x7B7fu8B_eQ)

### 9.3 Sample Run Statistics

From an actual analysis run (`Vehicle-and-Pedestrian-Traffic-summary.json`):

| Property | Value |
|----------|-------|
| Resolution | 1920×1080 |
| FPS | 59.94 |
| Processed Frames | 3,039 |
| Analysis Duration | 50.68 s |
| Total Unique Pedestrians | 2,740 |
| Total Unique Vehicles | 118 |

**Person Type Distribution:**

| Type | Unique Count |
|------|---------------|
| pedestrian | 2,723 |
| people | 953 |

**Vehicle Type Distribution:**

| Type | Unique Count |
|------|---------------|
| motor | 59 |
| bicycle | 44 |
| car | 33 |
| van | 25 |
| bus | 13 |
| truck | 5 |
| awning-tricycle | 5 |
| tricycle | 4 |

### 9.4 Per-Zone Highlights

| Zone | Unique Pedestrian | Peak Pedestrian | Avg Dwell (Pedestrian) | Unique Vehicle | Peak Vehicle | Avg Dwell (Vehicle) |
|------|--------------------|------------------|--------------------------|-----------------|----------------|------------------------|
| CROSSWALK_S | 896 | 133 | 3.44 s | 36 | 5 | 0.58 s |
| CROSSWALK_NW | 456 | 57 | 2.45 s | 6 | 1 | 0.65 s |
| CROSSWALK_NE | 199 | 25 | 2.20 s | 18 | 4 | 0.53 s |
| ROAD_MID | 587 | 129 | 4.59 s | 47 | 8 | 1.26 s |
| ROAD_E | 36 | 11 | 5.51 s | 24 | 10 | 17.10 s |
| SIDEWALK_E | 817 | 124 | 5.08 s | 10 | 2 | 5.29 s |

*Full per-zone breakdown for all 10 zones is available in the generated JSON summary.*

## 10. Output Files

Three artifacts are written next to the input video, named from its stem:

| File | Contents |
|------|----------|
| `<stem>-Analysis.mp4` | Annotated video: live dashboard panel stacked above the source frame, crosswalk safety masks, optional per-object boxes |
| `<stem>-timeseries.csv` | One row per second: per-zone pedestrian/vehicle instant counts + frame-level totals |
| `<stem>-summary.json` | Per-zone unique counts, peak occupancy, average dwell time, and frame-level type distributions |

## 11. Tech Stack

### 11.1 Core Technologies

- **Language:** C++17
- **Build System:** CMake 3.16+
- **Computer Vision / Inference:** OpenCV (`dnn`, `video`, `calib3d`, `videoio`, `highgui`, `imgproc`, `imgcodecs`, `core`)
- **Model Format:** ONNX (Ultralytics YOLO26x, VisDrone-trained)

### 11.2 Algorithm Components

| Component | Method | Purpose |
|-----------|--------|---------|
| Detection | YOLO26x (ONNX) via `cv::dnn` | Multi-class object detection |
| Motion Model | 8-state constant-velocity Kalman filter | Track state prediction |
| Camera Compensation | Sparse optical flow (GMC) | Neutralize camera pan/motion |
| Association | Hungarian algorithm + cascaded IoU/score fusion | Detection-to-track assignment |
| Zone Test | Ray-casting point-in-polygon | ROI membership |
| Intrusion Test | Segment intersection + containment | Crosswalk vehicle detection |
| Motion Classification | Windowed displacement averaging | Moving/stationary labeling |

### 11.3 Module Overview

| File | Lines | Responsibility |
|------|-------|-----------------|
| `main.cpp` | ~980 | Orchestration, ROI/zone/crosswalk logic, dashboard rendering, CSV/JSON export |
| `BotSort.cpp` | ~213 | Tracker update loop, cascaded matching stages |
| `Detector.cpp` | ~143 | DNN inference, letterbox, NMS, coordinate remapping |
| `STrack.cpp` | ~122 | Per-track lifecycle (activate/update/predict/lost/removed) |
| `Matching.cpp` | ~91 | IoU distance, score fusion, embedding distance |
| `KalmanFilterXYWH.cpp` | ~78 | Kalman predict/update/project |
| `Hungarian.cpp` | ~74 | Linear assignment solver |
| `GMC.cpp` | ~61 | Affine camera-motion estimation |

## 12. License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://github.com/kemalkilicaslan/Vehicle-and-Pedestrian-Traffic-Analysis-System/blob/main/LICENSE).

## 13. References

1. Ultralytics [YOLO26](https://docs.ultralytics.com/) Documentation.
2. Ultralytics [BoT-SORT Tracker](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py) Source Reference.
3. Zhang, Y., et al. (2022). [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864). *ECCV 2022*.
4. Aharon, N., et al. (2022). [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651).
5. Zhu, P., et al. (2021). [Detection and Tracking Meet Drones Challenge (VisDrone)](https://arxiv.org/abs/2001.06303). *IEEE TPAMI*.
6. OpenCV [Deep Neural Network (dnn) Module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html) Documentation.
7. Kuhn, H. W. (1955). [The Hungarian Method for the Assignment Problem](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109). *Naval Research Logistics Quarterly*.

---

### Acknowledgments

Thanks to the Ultralytics team for the YOLO26 architecture and the BoT-SORT/ByteTrack tracking algorithms this project's tracker implements. Thanks to the VisDrone dataset team (AISKYEYE, Tianjin University) for the drone-perspective traffic dataset used to fine-tune the detection model, and to the OpenCV community for the DNN and video processing infrastructure that makes a self-contained deployment possible.

---

**Note:** This system is designed for research, educational, and authorized traffic-monitoring purposes. ROI polygons are calibrated for a specific camera angle and reference resolution (1920×1080); recalibrate via `CHECK_ROI` mode before deploying on a different camera or intersection layout. Detection and tracking accuracy depend on camera height, angle, occlusion, and lighting; the VisDrone-trained model is optimized for elevated/aerial viewpoints. For production deployment in traffic management or enforcement contexts, validate results against ground truth and ensure compliance with local video-surveillance and data-retention regulations.
