# Frigate 0.18.2 on JetPack 5 (`v0.18.2-jp5`)

Upstream Frigate dropped JetPack 5 images in 0.16. This branch ports Frigate **v0.18.2** to JetPack 5 (L4T R35.x / Ubuntu 20.04 Focal / CUDA 11.4 / TensorRT 8.5.2), using the CUDA 11.4 build of `onnxruntime-gpu 1.19.2` from [`onnxruntime-jetpack5`](https://github.com/gonuit/onnxruntime-jetpack5).

## Key Changes from `v0.17.2-jp5` -> `v0.18.2-jp5`
1. **Rebased onto upstream Frigate `v0.18.2`** (`go2rtc 1.9.14`, `python-multipart 0.0.26`, `axengine 0.1.3`).
2. **Fixed deprecated Google Coral APT repository (`403 Forbidden`):** `docker/main/install_deps.sh` now installs `libedgetpu1-max_16.0tf2.17.1-1.ubuntu20.04_arm64.deb` directly from [`feranick/libedgetpu`](https://github.com/feranick/libedgetpu/releases/tag/16.0TF2.17.1-1).
3. **FP16 TensorRT Engine + 256 MB Workspace Cap for Xavier NX:**
   - `frigate/detectors/plugins/onnx.py`: Enables `requires_fp16=True` for the `onnx` detector.
   - `frigate/util/model.py`: Sets `trt_max_workspace_size=268435456` (256 MB) on `TensorrtExecutionProvider` to prevent 1 GB `NvMap` allocation failures (`error 12`) during first-run tactic profiling on 8 GB Jetson Xavier NX boards.
4. **Fast Multi-Stage Overlay Build (`jetpack5/Dockerfile.0.18.2-jp5`):**
   - Layers upstream `ghcr.io/blakeblackshear/frigate:0.18.2` onto `ghcr.io/gonuit/frigate-jetpack5:0.17.2-tensorrt-jp5` and applies the JetPack 5 TensorRT patches in ~2 minutes without recompiling OpenCV/Nginx from C++ source.

## Option A: Fast Overlay Build (~2 minutes on Jetson)

```sh
docker build -t frigate:0.18.2-tensorrt-jp5 -f jetpack5/Dockerfile.0.18.2-jp5 jetpack5/
```

## Option B: Full Source Build via Buildx

```sh
make local-trt-jp5
docker tag frigate:latest-tensorrt-jp5 frigate:0.18.2-tensorrt-jp5
```

## YOLOv9 ONNX Export (Opset 17 + `onnxsim`)

Export with `--opset 17` and simplify with `onnxsim` so TensorRT 8.5 compiles a single static FP16 `.engine` without `Shape`/`Gather` subgraph splitting:

```yaml
detectors:
  onnx:
    type: onnx
    device: Tensorrt

model:
  model_type: yolo-generic
  path: /config/model_cache/yolov9-s-320.onnx
  labelmap_path: /labelmap/coco-80.txt
  input_tensor: nchw
  input_dtype: float
  width: 320
  height: 320
```

Validated on Jetson Xavier NX (L4T R35.6.5 / JetPack 5.1.6): H.265 `2560x1440` + `640x360` hardware decode (`preset-jetson-h265` / `hevc_nvmpi`), single 17.4 MB FP16 `sm72` TensorRT engine, `~18.9 ms` inference in 10W mode (`~9 ms` in 20W mode).
