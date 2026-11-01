# Trap Tracker Real-Time Camera App

Trap Tracker Real-Time Camera App is a Docker-based web application for real-time camera trap ingestion, AI-assisted detection and alerting. It is designed for conservationists, researchers and biodiversity practitioners who need to process incoming camera trap images and generate species-based alerts.

## Use and licensing

This tool is free to use for non-commercial conservation, ecological research, biodiversity monitoring, academic research and education.

Commercial use, paid consultancy, resale, hosted services, or integration into proprietary commercial platforms requires prior written permission. See [`LICENSE`](LICENSE), [`NON_COMMERCIAL_USE.md`](NON_COMMERCIAL_USE.md) and [`COMMERCIAL_USE.md`](COMMERCIAL_USE.md).

Model weights are not included in this repository. Official Trap Tracker AI model releases will be made available separately through Zenodo so they can be cited using DOIs.

---

# TrapTracker RT — Docker

Real-time, email-driven wildlife monitoring. Branded to match TrapTracker DT,
but instead of processing a desktop folder of images, it watches camera-trap
emails, classifies attached images with your ONNX model, and emails alerts to
stakeholders when watch-list species are detected.

> **New here? See [TUTORIAL.md](TUTORIAL.md)** for a full step-by-step guide,
> or open the **Tutorial** tab inside the app.

## ⚠️ Run on your own machine only

TrapTracker RT has **no login** and stores email **app passwords** in its local
database (plain text). This is acceptable for a personal, single-machine
deployment but means you must **not** expose it to the internet or an untrusted
network. The Docker setup publishes the app on **`127.0.0.1` (localhost) only**
by default — it's reachable from the host machine and nowhere else. Only change
this if you understand and accept the risk.

## How it works

Each **project** has:

- **Cameras** — each maps a subject token to a routing key. Cameras email
  images with a subject containing `(=CameraName=)`. Only emails whose token
  matches an enabled camera are downloaded.
- **Email source** — the mailbox (IMAP for ingest, SMTP for alerts) the cameras
  send to. Configured per project in the UI.
- **Watch-list** — the species to alert on, each with a minimum confidence.
- **Recipients** — who receives alert emails.

When a project is **started**, a background runner:

1. Polls the mailbox (default every 30s), pulling images from matching camera
   emails into `Inbound/<Camera>/`, then trashing the email.
2. Scans `Inbound/` (default every 10s), runs ONNX detection on each new image,
   moves the original into `Processed/Original Images/`, writes an annotated
   copy into `Processed/Classified Images/`, and records detections.
3. For every watch-list species detected at/above its confidence, emails all
   recipients the annotated image. De-duplicated per (rule, image).

## Setup

1. **Add your model(s).** Drop your plain (unencrypted) ONNX model(s) into
   `models/`. Each model must have a class list with the **same base name** and
   a `.txt` extension (one class per line). For example:

   ```
   models/traptracker_uk_mammals_yolo26x_v1.0.0.onnx
   models/traptracker_uk_mammals_yolo26x_v1.0.0.txt
   models/traptracker_subsaharan_africa_mammals_yolo26x_v1.0.0.onnx
   models/traptracker_subsaharan_africa_mammals_yolo26x_v1.0.0.txt
   ```

   The app auto-discovers every `.onnx` in `models/`, pairs it with its
   `.txt`, and lets each project pick which model to use. The selected model's
   class list drives that project's species watch-list dropdown.

   A project only ever runs the model it selected. If that model's files are
   missing, monitoring won't start (no other model is substituted), and
   changing a project's model while it is running switches the running
   monitor over within one inbound poll. A class list whose number of classes
   doesn't match the model's own embedded names is refused, so a model can't
   run with another model's species names.

   Single-stage models are YOLO ONNX exports (YOLOv8/v10/11/26). End-to-end
   outputs (`[x1, y1, x2, y2, score, class_id]` rows) and raw grid outputs
   (NMS is applied in the app) both work. They run at `INPUT_SIZE` unless the
   model has a fixed input shape.

   **Two-stage models (DeepFaune).** A detector such as MegaDetector v6 finds
   the animals, people and vehicles, then a DeepFaune species classifier labels
   each animal crop. Put the pair in `models/two-stage-detectors/`:

   ```
   models/two-stage-detectors/deepfaune_ssa_v1.2_stage1_detector.onnx
   models/two-stage-detectors/deepfaune_ssa_v1.2_stage2_classifier.onnx
   models/two-stage-detectors/deepfaune_ssa_v1.2_stage2_classifier.txt
   ```

   It shows up as one model, `deepfaune_ssa_v1.2 (two-stage)`. The detector
   runs at the size it was trained at (1280 px for MegaDetector v6, read from
   the model), not `INPUT_SIZE`. The classifier's crop size, normalisation,
   crop padding and class list come from its ONNX metadata (see
   [`models/README.md`](models/README.md)). Person and vehicle boxes keep the
   detector's label, so `person` and `vehicle` can also go on a watch-list.
   The reported confidence is the classifier's species probability.

2. **Run it.**

   **Default (CPU) — works on any machine** (Intel/AMD, Apple Silicon, Linux,
   macOS, Windows; no GPU needed):
   ```bash
   docker compose up --build
   ```

   **GPU (optional, NVIDIA only)** — requires an NVIDIA GPU, recent drivers, and
   `nvidia-container-toolkit` on the host:
   ```bash
   docker compose -f docker-compose.gpu.yml up --build
   ```

   You don't need to know in advance whether your GPU will work: the GPU build
   tries the GPU first, and if the driver/CUDA/hardware isn't usable it logs the
   reason and **falls back to CPU automatically**. If you don't have an NVIDIA
   GPU at all, just use the default CPU command above.

   GPU compatibility:
   - **Tested on:** NVIDIA RTX A6000, driver 573.73, CUDA 12.8
   - **Minimum:** NVIDIA driver ≥ 525.60.13 (CUDA 12.x capable), x86_64
   - **Image targets:** CUDA 12.x, cuDNN 9.x, onnxruntime-gpu 1.19.2

   Check your driver/CUDA with `nvidia-smi`; if CUDA Version is 12.x the GPU
   build will work.

3. Open **http://localhost:8502**, create a project (choosing its detection
   model), add cameras, configure the email source, build the watch-list and
   recipients, then **Start monitoring**.

## Camera email format

Subject must contain the camera token between `=` markers, e.g.:

```
Trail camera motion (=Camera01=)
```

`Camera01` must match a camera's **Name** or **Routing key** (case-insensitive).
Image attachments (`.jpg .jpeg .png .bmp .tif .tiff`) are extracted.

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_DIR` | `/app/models` | Folder scanned for `.onnx` models + `.txt` class lists |
| `INPUT_SIZE` | `640` | Single-stage model input size (two-stage detectors use their own training size) |
| `CONF_THRESHOLD` | `0.10` | Detector confidence floor |
| `ORT_PROVIDERS` | `CPUExecutionProvider` (default build) / `CUDAExecutionProvider,CPUExecutionProvider` (GPU build) | ONNX Runtime providers to try, in order. GPU is attempted first then falls back to CPU |
| `POLL_EMAIL_SECONDS` | `30` | Mailbox poll interval |
| `POLL_INBOUND_SECONDS` | `10` | Inbound scan interval |
| `HOST` | `127.0.0.1` | Bind address (localhost by default; the container sets `0.0.0.0` internally and the compose port mapping restricts host exposure to localhost) |
| `PORT` | `8501` | Web port |
| `SECRET_KEY` | *(auto-generated)* | Flask session key. If unset, a random key is generated and persisted to `data/.secret_key` |

Email credentials are **not** environment variables — they're entered per
project in the UI (Gmail app passwords recommended). There is no login.

## Persistent data (mounted volumes)

- `data/` — SQLite database and the generated secret key
- `projects/` — per-project image storage (`Inbound/`, `Processed/`)
- `models/` — your ONNX model(s) + class lists

## Notes

- **Localhost only by default.** No login and no credential encryption — keep it
  on your own machine. See the warning at the top of this file.
- **Auto-resume:** projects that were monitoring before a restart resume
  automatically when the app starts again.
- **Health endpoint:** `GET /healthz` lists detected models; the Docker
  healthcheck uses it.
- Gmail: enable IMAP and use a 16-character **app password**, not your account
  password.
- Each model's `.txt` class list (same base name as the `.onnx`) drives that
  project's species dropdown.
- If alert emails land in spam, mark them "Not spam" or add a filter — common
  for automated mail. The app sends attachments with correct MIME types and
  reports SMTP rejections in the monitoring log.
