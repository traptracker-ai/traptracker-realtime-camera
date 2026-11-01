# TrapTracker RT — User Guide & Tutorial

A step-by-step guide to setting up real-time camera-trap monitoring with
TrapTracker RT.

---

## What TrapTracker RT does

TrapTracker RT watches an email inbox that your trail cameras send photos to.
When a new image arrives, it:

1. Downloads the photo from the email
2. Runs it through an AI model that identifies the species
3. Checks the result against your **watch-list** (the species you care about)
4. Emails your stakeholders the labelled image whenever a watch-list species
   is detected

You organise everything into **projects**. Each project has its own cameras,
its own mailbox, its own detection model, and its own watch-list and recipients.

---

## ⚠️ Important: run it on your own machine only

TrapTracker RT has **no login screen**, and it stores your email **app password**
in its local database. That is fine when it runs on your own computer, but it
means you must **not** expose it to the internet or an untrusted network.

The default Docker setup publishes the app on `127.0.0.1` (localhost) only — it
is reachable from the machine it runs on and nowhere else. Keep it that way
unless you fully understand the consequences of changing it.

---

## Before you start — what you need

- **Docker** and **Docker Compose** installed
  ([Docker Desktop](https://www.docker.com/products/docker-desktop/) on Windows/Mac,
  or Docker Engine on Linux)
- One or more **ONNX detection models** (`.onnx`) with a matching class list
  (`.txt`). Place these in the `models/` folder.
- A **dedicated email account** for your cameras (Gmail works well). You will
  need an **App Password**, not your normal account password.
- Trail cameras that can **email photos** with the camera's name in the subject.

**You do _not_ need a GPU.** The default build runs on any computer (Windows,
Mac including Apple Silicon, or Linux) using the CPU. A GPU is optional and only
speeds things up — see Step 2.

---

## Step 1 — Add your model(s)

Put your model files in the `models/single-stage-detectors/` folder. Each model
is a `.onnx` file with a `.txt` class list of the **same name**:

```
models/single-stage-detectors/cai_uk_mammals_yolo26x_v1.0.0.onnx
models/single-stage-detectors/cai_uk_mammals_yolo26x_v1.0.0.txt
```

The `.txt` file lists one species per line. This list becomes the species
dropdown when you build a watch-list.

You can add several models (e.g. UK mammals and African mammals) — each project
picks which one to use.

**Two-stage DeepFaune models.** These come as a pair: a detector
(MegaDetector) that finds each animal, and a classifier that names the
species. Put both files, plus the class list, in `models/two-stage-detectors/`:

```
models/two-stage-detectors/deepfaune_ssa_v1.2_stage1_detector.onnx
models/two-stage-detectors/deepfaune_ssa_v1.2_stage2_classifier.onnx
models/two-stage-detectors/deepfaune_ssa_v1.2_stage2_classifier.txt
```

The pair appears in the model list as one model, marked *two-stage*. Its
watch-list dropdown has every species plus `person` and `vehicle`.

---

## Step 2 — Start the app

TrapTracker RT runs on almost any machine. There are two ways to run it: a
**CPU build** that works everywhere, and an optional **GPU build** that's faster
if you have an NVIDIA graphics card.

### Which one should I use?

- **Most people: use the CPU build.** It works on any computer — Windows, Mac
  (both Intel and Apple Silicon), and Linux — and needs no special hardware or
  drivers. For typical camera-trap volumes (images arriving every few minutes)
  CPU speed is perfectly adequate.
- **Use the GPU build only if** you have an NVIDIA GPU *and* have installed the
  NVIDIA Container Toolkit on the host (see below). It speeds up detection, which
  matters mainly if you're processing a high volume of images at once.

### Default (CPU) — recommended

```bash
docker compose up --build
```

This works on any machine, including Apple Silicon Macs and ARM servers. No GPU
required.

### GPU (optional, NVIDIA only)

```bash
docker compose -f docker-compose.gpu.yml up --build
```

Requirements for the GPU build:
- An NVIDIA GPU
- Up-to-date NVIDIA drivers on the host machine
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed on the host (this is what lets Docker access the GPU)

**GPU compatibility**

| | |
| --- | --- |
| **Tested on** | NVIDIA RTX A6000 · Driver 573.73 · CUDA 12.8 |
| **Minimum** | NVIDIA driver ≥ 525.60.13 (CUDA 12.x capable), x86_64 |
| **Image targets** | CUDA 12.x · cuDNN 9.x · onnxruntime-gpu 1.19.2 |

Check your own driver/CUDA version any time with `nvidia-smi` — the top banner
shows **Driver Version** and **CUDA Version**. As long as your CUDA Version reads
12.x (driver ≥ 525), the GPU build will work; the bundled CUDA libraries handle
the rest. Older drivers should use the CPU build.

### Automatic CPU fallback

You don't have to know in advance whether your GPU will work. Even with the GPU
build, the app **tries the GPU first, and if it can't be used** — missing driver,
wrong CUDA version, no GPU, or unsupported hardware — **it automatically falls
back to CPU** and writes a clear message explaining why in the monitoring log.
Detection continues either way; you never get a hard crash over GPU issues.

You can see what it's using in two places:
- The **System Status** box in the sidebar shows **Compute: GPU** or **CPU**.
- The endpoint **http://localhost:8502/healthz** reports the active provider and
  any fallback message.

### When it's ready

You'll see:
```
============================================================
TrapTracker RT listening on http://0.0.0.0:8501
Models found: 2
============================================================
```

Open **http://localhost:8502** in your browser.

---

## Step 3 — Set up your camera email account (Gmail example)

Your cameras need a mailbox to send photos to. Using Gmail:

1. Create or choose a Gmail account just for your cameras.
2. Turn on **2-Step Verification** (required for app passwords):
   Google Account → Security → 2-Step Verification.
3. Create an **App Password**:
   Google Account → Security → App Passwords → create one for "Mail".
   You'll get a 16-character password like `abcd efgh ijkl mnop`.
4. Make sure **IMAP is enabled**: Gmail → Settings → Forwarding and POP/IMAP →
   Enable IMAP.

Keep the app password handy for Step 6.

---

## Step 4 — Create a project

1. Click **New Project**.
2. Give it a name (e.g. "North Pennines Survey").
3. Choose the **detection model** for this project.
4. Click **Create project**.

---

## Step 5 — Add your cameras

In the project page, find the **Cameras** section.

Each camera is identified by a **token** in the email subject line. When a
camera sends a photo, the subject must contain its name between `=` signs:

```
Trail camera motion (=Camera01=)
```

Here, `Camera01` is the token. Add a camera with:

- **Name** — must match the token your camera puts in the subject (e.g. `Camera01`)
- **Routing key** — optional alternative the token can match (defaults to Name)
- The other fields (location, type, notes) are for your reference

> **Tip:** Configure your camera's email subject to include `(=YourCameraName=)`.
> How you do this depends on the camera brand — check its email/SMS settings.
> Only emails whose token matches a camera you've added will be processed.

---

## Step 6 — Configure the email source

In the **Email Source** section, enter:

| Field | Value (Gmail) |
| --- | --- |
| Email address | your camera Gmail address |
| App password | the 16-character app password from Step 3 |
| IMAP host | `imap.gmail.com` |
| IMAP port | `993` |
| Use SSL for IMAP | ✓ ticked |
| SMTP host | `smtp.gmail.com` |
| SMTP port | `587` |
| Use STARTTLS for SMTP | ✓ ticked |

Click **Save email settings**.

Then use **Send test email** — enter an address (ideally a *different* one from
the sender) and click **Send test**. If it arrives, your settings work.

> **If the test lands in spam:** that's common for automated mail. In the
> receiving inbox, mark it "Not spam" and/or add the sender to your contacts, or
> create a filter to never send it to spam. See the Troubleshooting section.

---

## Step 7 — Build your watch-list

In **Species Watch-list**:

1. Pick a species from the dropdown (these come from your model's class list).
2. Set a **minimum confidence** (0.50 is a sensible start — higher means fewer
   but more certain alerts).
3. Click **Add to watch-list**.

Add as many species as you want to be alerted about.

---

## Step 8 — Add alert recipients

In **Alert Recipients**, add the email addresses of everyone who should receive
alerts (rangers, colleagues, yourself). Each recipient gets the labelled image
attached whenever a watch-list species is detected.

---

## Step 9 — Start monitoring

Scroll to the **Monitoring** section and click **Start monitoring**.

The live log shows what's happening:
- Email polls and downloads
- Each image being detected and its best result
- Classified images being created
- Alerts being sent

The app now runs continuously. When a camera emails a photo of a watch-list
species, your recipients get an alert within seconds of the next poll.

To stop, click **Stop monitoring**. If the app or computer restarts, projects
that were running will **resume automatically**.

---

## Viewing results

Click **Gallery** on the project page to see processed images, each with its
best detection and confidence. Boxed (labelled) images are shown where a
detection was made.

---

## How the whole flow works (at a glance)

```
Camera takes photo
      │
      ▼
Camera emails it to your mailbox, subject: "... (=Camera01=)"
      │
      ▼
TrapTracker RT polls the mailbox (every ~30s)
      │  matches the (=Camera01=) token to a camera you added
      ▼
Downloads the image → runs the AI model
      │
      ▼
Species on your watch-list, above its confidence?
      │ yes
      ▼
Emails all recipients the labelled image
```

---

## Troubleshooting

### Install & startup

**Build error: `No matching distribution found for onnxruntime-gpu`**
You're building the **GPU** image on a machine that can't use it — most often an
Apple Silicon (ARM) Mac or ARM server, where the GPU package doesn't exist. Use
the default CPU build instead: `docker compose up --build`.

**Errors about missing GPU, NVIDIA driver, or CUDA version**
The GPU build needs an NVIDIA GPU, current drivers, and the NVIDIA Container
Toolkit on the host. If you don't have all three, use the default CPU build:
`docker compose up --build`. (The app also falls back to CPU on its own and logs
the reason, so a running GPU build won't crash over this — but if it won't even
build, switch to the CPU command.)

**GPU build runs but logs `libcublasLt.so.*: cannot open shared object file`**
This was a CUDA library mismatch in an older build. The current GPU image bundles
the matching CUDA 12 / cuDNN 9 runtime libraries via pip and registers them with
the dynamic linker, so a clean rebuild resolves it: `docker compose -f
docker-compose.gpu.yml build --no-cache` then `up`. Your NVIDIA driver must be
recent enough for CUDA 12.x (driver >= 525); check with `nvidia-smi`.

**It built but I'm not sure if it's using the GPU**
Check the **System Status** box in the sidebar (Compute: GPU / CPU) or open
**http://localhost:8502/healthz**. The monitoring log also states which it's
using when a project starts.

**Port 8502 already in use**
Another program (or a previous container) is using the port. Stop the other
process, or change the published port in the compose file (e.g.
`"127.0.0.1:8600:8501"`) and open `http://localhost:8600`.

### Email & detection

**"Alert sent" in the log but no email received**
Check the **Spam** and **All Mail** folders of the recipient. Automated mail
with attachments often gets filtered the first time. Mark it "Not spam" or add a
filter. If sender and recipient are the *same* Gmail account, Gmail may not show
a separate copy — test with a different recipient.

**Email poll error / IMAP login failed**
Double-check the email address and that you used the **App Password**, not the
account password. Confirm IMAP is enabled in Gmail. After fixing settings, click
**Stop**, then **Start** again.

**No species in the watch-list dropdown**
The project's model has no class list, or the `.txt` file isn't named to match
the `.onnx`. Check the `models/` folder.

**Images aren't being picked up**
Confirm the email subject contains `(=CameraName=)` and that the name matches a
camera you added (case doesn't matter). Emails without a matching token are
ignored by design.

**Models not showing**
Make sure each `.onnx` has a `.txt` of the same base name in `models/`, then
refresh. Visit **http://localhost:8502/healthz** to see what the app detects.

**Monitoring stopped after a restart**
Projects that were running resume automatically on startup. If one didn't, open
it and click **Start monitoring** again.

---

## Data & privacy

- All data stays on your machine: the SQLite database in `data/`, images in
  `projects/`, models in `models/`.
- Email app passwords are stored in the local database in plain text. This is
  acceptable for a localhost-only deployment; do not expose the app to a network
  or share your `data/` folder.
- Nothing is sent anywhere except the alert emails you configure.
