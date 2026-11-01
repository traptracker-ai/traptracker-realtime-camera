# Models folder

Place your compatible model files in this folder when running the application locally.

Expected layout:

```text
models/
  your-model.onnx
  your-model.txt
  two-stage-detectors/
    <name>_stage1_detector.onnx
    <name>_stage2_classifier.onnx
    <name>_stage2_classifier.txt
```

The `.txt` file should use the same base name as the `.onnx` file and contain one class name per line.

A two-stage model (e.g. DeepFaune: MegaDetector v6 + a species classifier) is a `_stage1_detector.onnx` / `_stage2_classifier.onnx` pair with the same `<name>`, listed in the app as one model called `<name>`. The classifier's `.txt` is optional when the class list is embedded in the ONNX.

Large model weights are intentionally not included in this GitHub repository. Official Trap Tracker AI model releases will be distributed separately through Zenodo or another release service, with model cards and licence terms.

## Two-stage classifier metadata

The stage 2 classifier ONNX can carry these `metadata_props` (JSON values). All are optional. The DeepFaune-UK and DeepFaune-SSA exports set them all.

| Key | Example | Purpose |
|---|---|---|
| `classes` | `["Car", "MelesMeles", ...]` | Label for each output, in order. Takes precedence over the `.txt` file. |
| `input_size` | `224` | Square crop size. |
| `mean`, `std` | `[0.485, 0.456, 0.406]` | Normalisation. |
| `class_groups` | `{"Car": "vehicle", "MelesMeles": "animal", "Person": "person"}` | The stage 1 detector class each label belongs under. |
| `crop_padding` | `0.2` | Context added around each box before the square crop, as a fraction of the box's longer side on every side. Defaults to `0`. Set it to whatever your training crops used. |

With `class_groups`, each box is classified only among the labels grouped under its detector class (compared case-insensitively). An `animal` box can then never be labelled `Car`, and a `vehicle` box is labelled with the best vehicle label instead of just "vehicle". Boxes from a detector class with no labels keep the detector's label. The reported confidence is still the label's probability over all classes, so a crop that fits none of its allowed labels stays low-confidence.

Crops are made the DeepFaune way: pad the box by `crop_padding`, expand it to a square about its centre, clip it to the image, then resize it to `input_size` with antialiased bilinear resampling (the same as torchvision's `Resize`).

Without `class_groups`, only `animal` boxes are reclassified (over every label) when the detector has an `animal` class; otherwise every box is.

The stage 1 detector runs at the `imgsz` recorded in its metadata (1280 for MegaDetector v6) unless it has a fixed input shape. Its class names come from the Ultralytics `names` metadata.

This is the same two-stage behaviour as the Trap Tracker Offline Processor, so both apps give the same labels for the same image.
