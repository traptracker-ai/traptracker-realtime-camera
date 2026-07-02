# Models folder

Place your compatible model files in this folder when running the application locally.

Expected layout:

```text
models/
  your-model.onnx
  your-model.txt
```

The `.txt` file should use the same base name as the `.onnx` file and contain one class name per line.

Large model weights are intentionally not included in this GitHub repository. Official Trap Tracker AI model releases will be distributed separately through Zenodo or another release service, with model cards and licence terms.
