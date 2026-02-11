# Analogue Gauge Detection

## Getting started

### Download data

Download zip from Roboflow gauge object detection data [[Download link](https://universe.roboflow.com/ds/8df5qM7D4g?key=KTJdjYlukK)].

### Setup

Setup environment with uv

```bash
uv sync
```

## Approach

2 stage approach:

1. Detect different objects (min, max, center, pointer, gauge)
2. Read gauge value

Comparing different approaches

- Object detection
  - CNN-based model (Yolov8, Faster R-CNN)
  - Transformer-based model (pure ViT, RT-DETR)
- Reaeding gauge value
  - using OCR and angle computation
  - VLM
