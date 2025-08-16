# CV-Classroom-Analytics

# Student Attendance and Engagement Analysis using Computer Vision

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the final project for the **Fundamentals of Computer Vision** course at the Iran University of Science and Technology.

---

## 🎯 About The Project

This project is an intelligent system that analyzes student attendance and engagement levels in a classroom using computer vision tools. By processing recorded class videos, the system identifies students, analyzes their behavior, and provides actionable insights.

### ✨ Key Features

* **Intelligent Image Preprocessing:** Automatically enhances the quality of low-resolution face images using Super-Resolution models (EDSR, ESPCN).
* **Face Detection and Alignment:** Utilizes the powerful MTCNN model for accurate face detection and alignment to improve recognition accuracy.
* **Unsupervised Face Clustering:** Automatically identifies individuals in the class without initial labeling using the HDBSCAN algorithm.
* **Recognition Database Creation:** Builds a robust face database after manual review and correction of the clustered groups.
* **Real-time Recognition and Tracking:** Identifies and tracks students across video frames using modern algorithms.
* **(In Development) Emotion and Attention Analysis:** Analyzes students' emotional states and gaze direction to assess engagement levels.
* **(In Development) Analytical Dashboard:** Provides statistical and visual reports of the analysis results.

---

## 📂 Project Structure

```
project-root/
│
├── data/
│   ├── raw_videos/          # Raw class videos
│   └── processed_faces/     # Preprocessed faces (upscaled, aligned)
│
├── grouped_faces/           # Manually reviewed and corrected face clusters
├── models/                  # Downloaded models (e.g., EDSR_x4.pb)
├── src/                     # Main project source code
│   ├── 01_preprocess_faces.py    # Script for preprocessing and face extraction
│   ├── 02_cluster_faces.py       # Script for clustering
│   ├── 03_create_database.py     # Script for creating the recognition database
│   └── 04_main_analysis.py       # Main script for video analysis
│
├── requirements.txt         # List of required libraries
└── README.md                # This file
```

---

## 🛠️ Installation & Setup

To get a local copy up and running, follow these simple steps.

**1. Create and Activate a Virtual Environment:**
```bash
# Create the virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```


**2. Clone the Repository:**
```bash
git clone https://github.com/dark-noob830/CV-Classroom-Analytics.git
cd CV-Classroom-Analytics
```

**3. Install Dependencies:**
All required libraries are listed in the `requirements.txt` file. Install them with the following command:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

The project workflow is divided into several steps:


**Step 1: Preprocess and Extract Faces**
1. First, place your raw videos in the `videos` directory. 
2. Run the following script to extract and save faces to the `data_extract/temp_faces` directory.
- ```bash
  python face_extractor.py 
  ```
3. After done run this script for Upscaling extracted face image
- ```bash
  python upscale.py
  ```
--- 
