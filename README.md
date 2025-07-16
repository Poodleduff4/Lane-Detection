
# DeepLanes – Lane Detection with U-Net & TransUNet

A concise PyTorch implementation of semantic-segmentation models for lane detection.  The project compares two architectures:

1. **Model 1 – U-Net** (baseline)
2. **Model 2 – TransUNet** (Vision Transformer encoder + U-Net decoder)

Both models are trained and evaluated on the **[CULane dataset](https://xingangpan.github.io/projects/CULane.html)** (133 k frames captured from a dash-cam in varying traffic, lighting & weather).  See the accompanying conference-style report for full details.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `model_1_train.py` / `model_1_inference.py` | U-Net training & inference scripts |
| `model_2_train.py` / `model_2_inference.py` | TransUNet training & inference |
| `model_2.py` | Stand-alone definition of the `TransUNet` network |
| `video_inference.py` | Overlay predictions onto a driving video |
| `yolo_detection.py` | Legacy YOLOv5 baseline (kept for comparison) |
| `unet_requirements.txt` | Reproducible Python dependencies |
| `examples/` | Sample frames + masks (see below) |

---

## Quick Start 🚀

1.  Environment (Python 3.10)
    ```bash
    python -m venv venv && source venv/bin/activate  # or use Conda
    pip install -r requirements.txt
    ```
2.  Training (example for TransUNet – adjust paths for CULane)
    ```bash
    python model_2_train.py \
        --images_dir /path/to/CULane/images/train \
        --masks_dir  /path/to/CULane/laneseg_label_w16/train
    ```
3.  Single-frame inference
    ```bash
    python model_2_inference.py --weight best_unet_lane_detection.pth \
        --image tests/frame.jpg --out results/frame_mask.png
    ```
4.  Video overlay
    ```bash
    python video_inference.py --weights best_unet_lane_detection.pth --video input.mov
    ```

---

## Example results

The `examples/` folder contains three illustrative PNGs used in the report:

| Original frame | Ground-truth mask | Model prediction |
|----------------|------------------|------------------|
| ![orig](examples/original.png) | ![gt](examples/ground_truth.png) | ![pred](examples/predicted.png) |

> Replace these placeholders with your own exported frames for a polished portfolio presentation.

---

## Dataset citation

> Pan, Xingang, et al. *Spatial As Deep: High-Performance Lane Detection.* 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2018.
>
> Dataset: **CULane** – https://xingangpan.github.io/projects/CULane.html

Please download the dataset from the official source and respect its license.

---

## Results (10 epochs)

| Model | Loss ↓ | Accuracy ↑ | F1 ↑ | Jaccard ↑ |
|-------|--------|------------|------|-----------|
| **U-Net** | 0.28 | 0.9761 | 0.5075 | 0.3466 |
| **TransUNet** | 0.41 | 0.9758 | 0.4753 | 0.3183 |

*(see report for full experimental setup, hyper-parameters & discussion)*

---

---

## Authors

* Yohann Kuruvila  ·  Anais Zulfequar  ·  Nicholas Cramarossa  ·  Luke Guardino  ·  Matthew Bush



