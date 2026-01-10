
# Enhancing Fine-Grained Livestock Behavior Detection with Lightweight Parallel Attention in YOLOv12


## Abstract

Fine-grained livestock behavior recognition remains a key challenge in precision farming, as subtle posture variations and complex barn conditions often hinder the performance of standard YOLO-based detectors. To address these issues, we propose PLCA-Head, a lightweight parallel local and channel attention module integrated into YOLOv12. By combining dynamic local spatial recalibration with global channel-wise modulation in parallel paths, PLCA-Head effectively captures nuanced spatial and semantic features critical for distinguishing behaviors such as rumination. Evaluated on the CBVD-5 benchmark, our approach achieves 97.1% mAP@50, 72.4% mAP@50–95, 93.3% precision, and 94.4% rumination AP, outperforming both YOLOv12s and SlowFastNet across all major metrics. Compared to SlowFastNet, it reduces parameters by ~70%, GFLOPs by ~68.8%, and delivers 10× faster inference, while surpassing YOLOv12s in accuracy with 9.7% fewer convergence epochs and maintaining real-time performance.

Source code and trained models: https://github.com/YifangGaoinPG/yolov12le

## Quick Start 🚀

### 1. Installation

```
pip install -q git+https://github.com/YifangGaoinPG/yolov12le.git roboflow supervision flash-attn
```

### 2. Dataset
Use the following code:
```python

import pillow_heif

# Mock register_avif_opener for compatibility
def mock_register_avif_opener(*args, **kwargs):
    pass

pillow_heif.register_avif_opener = mock_register_avif_opener

from roboflow import Roboflow

# Replace with your own Roboflow API key
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY_HERE"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("shyam-hdmec").project("cbvd")
version = project.version(1)
dataset = version.download("yolov12")

# Optional: clean up data.yaml (remove extra path lines)
import os
yaml_path = os.path.join(dataset.location, "data.yaml")
with open(yaml_path, "r") as f:
    lines = f.readlines()
with open(yaml_path, "w") as f:
    f.writelines(lines[:-4])   # usually remove last 4 lines
```

### Training
```python
from ultralytics import YOLO

model = YOLO("yolov12s.yaml")  # loads version with PLCA-Head

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=600,
    patience=50,
    batch=32,
    imgsz=640,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    cos_lr=True,
    amp=True,
    workers=8,
    name="yolov12le_train",
    pretrained=True
)
```
### Validation
```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/yolov12le_train/weights/best.pt")

# Validate
metrics = model.val(data=f"{dataset.location}/data.yaml")
```

### Citation
@article{gao2026enhancing,
  title={Enhancing Fine-Grained Livestock Behavior Detection with Lightweight Parallel Attention in YOLOv12},
  author={Gao, Yifang and Luo, Wei and Zhang, Shunshun and Ahmad, Nur Syazreen and Wang, Xiaojun and Goh, Patrick},
  journal={The Visual Computer},
  year={2026},
  publisher={Springer}
  % doi = {insert DOI when available}
}

### Related Projects 🔗

- Based on [Ultralytics](https://github.com/ultralytics/ultralytics)
- CBVD-5 Dataset paper: https://www.nature.com/articles/s41598-024-52266-1

Thank you for your interest in our work!  
Any questions → please open an issue on GitHub.
