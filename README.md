# Smoke and Fire Detection

This project focuses on detecting smoke and fire in videos using deep learning models. It includes scripts for data preprocessing, model training, and evaluation. The framework utilizes optical flow, CNN-based feature extraction, and spatial transformations to improve detection accuracy.

---

## **Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
---

## **Features**
- Converts videos into image sequences for easier processing.
- Generates optical flow to extract motion information.
- Includes CNN-based deep learning models for smoke and fire detection.
- Modular design for data preprocessing, model training, and evaluation.

---

## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Smoke_Fire_Detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Smoke_Fire_Detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
1. **Convert Videos to Images**:
   Use the `video_to_image.py` script to extract frames from videos.
   ```bash
   python video_to_image.py --video_path <path_to_video> --output_dir <output_directory>
   ```

2. **Generate Optical Flow**:
   Run the `optical_flow.py` script to compute optical flow from frames.
   ```bash
   python optical_flow.py --input_dir <frames_directory> --output_dir <optical_flow_directory>
   ```

3. **Prepare Dataset**:
   Use `dataset.py` for dataset loading and preprocessing.
   ```bash
   python dataset.py --data_dir <data_directory>
   ```

4. **Train Models**:
   Train the CNN models using `cnn_models.py` and `main.py`.
   ```bash
   python main.py --config <config_file>
   ```

5. **Generate Clips**:
   Use `generate_clips.py` to create video clips for better annotation and analysis.
   ```bash
   python generate_clips.py --video_path <path_to_video> --clip_length <length_in_seconds>
   ```

---

## **File Descriptions**
- **`cnn_models.py`**: Contains CNN architectures for smoke and fire detection.
- **`generate_clips.py`**: Script for creating video clips from long videos.
- **`optical_flow.py`**: Computes optical flow between video frames.
- **`dataset.py`**: Handles dataset loading and preprocessing.
- **`video_to_image.py`**: Converts videos into image sequences.
- **`main.py`**: The main script for training and evaluating models.
- **`spatial_transforms.py`**: Implements spatial transformations for data augmentation.

---

---
