# Gender Classification Repository

This repository contains a gender classification model with ONNX support.

## Setup Instructions

### 1. Create Python Environment
Create a Python 3.10 environment using your preferred environment manager (e.g., `venv`, `conda`).

#### Using `venv`:
```bash
python3.10 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

#### Using `conda`:
```bash
conda create -n gender_classification python=3.10
conda activate gender_classification
```

### 2. Install Dependencies
Install the required packages by running:
```bash
pip install -r setup.txt
```

## Face Detection Model
- Download the face detection model:
  - [Caffemodel](https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/blob/master/res10_300x300_ssd_iter_140000.caffemodel)
  - [Prototxt](https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/blob/master/deploy.prototxt.txt)
- Place the files in `models/face_detector/`.

## Gender Classification Models
- **Original Model**:
  - Source: [https://huggingface.co/rizvandwiki/gender-classification](https://huggingface.co/rizvandwiki/gender-classification)
  - Place files in `models/gender/rizvandwiki/`.
- **ONNX Model**:
  - Source: [https://huggingface.co/onnx-community/gender-classification-ONNX](https://huggingface.co/onnx-community/gender-classification-ONNX)
  - Place files in `models/gender/rizvandwiki_onnx/`.
  - Move the desired model from `models/gender/rizvandwiki_onnx/onnx/` to `models/gender/rizvandwiki_onnx/`.
  - Note: Avoid using the int8 model due to errors.

## Files structure
```
gender_classification/
├── models/
│   ├── face_detector/
│   │   ├── res10_300x300_ssd_iter_140000.caffemodel
│   │   └── deploy.prototxt.txt
│   └── gender/
│       ├── rizvandwiki/
│       │   └── [original model files from Hugging Face]
│       └── rizvandwiki_onnx/
│           ├── [ONNX model file(s), e.g., gender_classification.onnx]
│           └── onnx/
│               └── [additional ONNX model files, not int8]
├── webcam_gender.py
├── webcam_gender_onnx.py
└── setup.txt
```

## Requirements
- Python: 3.10.14
- Dependencies listed in `setup.txt`