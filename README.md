# Non-Invasive Iron Deficiency Anaemia Prediction using Multimodal Fusion
A deep learning framework for **simultaneous anaemia classification and haemoglobin level estimation** from palm and nail images, using a novel three-stage attention-based multimodal fusion mechanism.

## What is this project?

Iron Deficiency Anaemia (IDA) affects over 1.6 billion people worldwide. Diagnosis typically requires invasive blood tests which are costly and inaccessible in low-resource settings where anaemia is most prevalent.

This project proposes a **non-invasive screening framework** that uses the following modalities to to simultaneously **classify anaemia** and **estimate haemoglobin (Hb) levels in g/dL**
- 📸 A palm image
- 💅 A nail image
- 👤 Age and gender


## What makes this different?

Most existing non-invasive anaemia detection models rely on a single image modality (just nail or just palm) and use simple feature concatenation. This framework introduces three key novelties:

<img width="639" height="607" alt="image" src="https://github.com/user-attachments/assets/ff051ed3-f9a2-4e0a-b480-32292174ca37" />

### 1. Dual Image Modality
Palm and nail images are processed independently through separate CNN streams, as different individuals show pallor more prominently in different regions, so combining both modalities captures this variation and improves generalisation.

### 2. Three-Stage Attention-Based Fusion
Rather than simply concatenating features, this framework uses a custom three-stage attention mechanism:

- **Self-Attention** — refines spatial features within each modality, helping the model focus on clinically relevant regions like the nail bed and palmar pallor zones
- **Cross-Attention** — allows demographic metadata (age and gender) to guide visual feature interpretation, personalising predictions for each individual
- **Fusion Attention** — dynamically weights the contribution of each modality rather than treating them equally

### 3. Joint Multi-Task Learning
Classification and haemoglobin regression are performed **simultaneously in a single forward pass**, allowing the two tasks to inform each other and improve overall generalisation.

## Key Results

| Metric | Value |
|--------|-------|
| Test AUC | 0.72 |
| Accuracy | 69% |
| Recall (Anaemic) | **0.78** |
| Haemoglobin MAE | **0.689 g/dL** |
| R² | 0.53 |

The model correctly identified **78% of true anaemic cases** , outperforming the baseline comparison model on the most clinically critical metric despite using less metadata.


## Interpretability

The framework generates spatial attention heatmaps for every prediction, showing exactly where the model focused on the palm and nail images and how strongly demographic metadata influenced each decision. This addresses a key gap in existing models, which function as black boxes without any explanation of how predictions were made.


## Project Structure

```
├── Data/                        # Dataset (not included — see below)
├── Backbones.py                 # CNN backbone definitions (CNN1–CNN5 + pretrained)
├── Attention.py                 # Self-attention, cross-attention, fusion attention
├── Build.py                     # Model builder — classification / regression / joint
├── Augmentation.py              # Data augmentation pipeline
├── Data Preparation.ipynb       # Preprocessing pipeline
├── Classification.ipynb         # Backbone comparison — classification mode
├── Regression.ipynb             # Backbone comparison — regression mode
├── JointTask.ipynb              # Final joint model + ablation + attention maps
├── requirements.txt             # Dependencies
└── README.md
```


## How to Reproduce Results

**1. Clone the repository**
```bash
git clone https://github.com/MadhunishaBala/Non-Invasive-Iron-Deficiency-Anaemia-Prediction-using-Multimodal-Fusion.git
cd Non-Invasive-Iron-Deficiency-Anaemia-Prediction-using-Multimodal-Fusion
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

This project uses the publicly available dataset introduced by Navarro et al. (2025), available under the CC BY 4.0 license.

📁 [Dataset Repository](https://drive.google.com/drive/folders/140wuhoE9kUREEtEHTuSzjSpZqaYJAb26?usp=sharing)

```
Data/
└── SubjectID/
    ├── info.json
    ├── Palm_ROI/
    │   └── frame_03.jpg
    └── nail_roi.jpg
```

**4. Run the notebooks in order**
```
1. Data Preparation.ipynb     — preprocessing and ROI extraction
2. Classification.ipynb       — backbone comparison (classification)
3. Regression.ipynb           — backbone comparison (regression)
4. JointTask.ipynb            — final model, ablation study, attention maps
```

## Requirements
```
tensorflow>=2.10
opencv-python
scikit-learn
numpy
pandas
matplotlib
seaborn
```
Install with:
```bash
pip install -r requirements.txt
```

## Live Demo

🚀 **Application deployed at:** https://web-production-e16a5.up.railway.app/

The application allows you to upload a palm image, nail image and enter age and gender to receive an instant anaemia classification and haemoglobin 
level prediction.

⚠️ 
This application is a research prototype and is currently under development. Predictions are based on a model trained on a limited dataset of 527 subjects aged 18–25 from Peru, and may not generalise to all individuals. It is intended as a preliminary screening tool only and should not be used as a substitute for clinical diagnosis.

**Note:** The deployment code for the web application is maintained in a separate repository. This repository contains the research and training pipeline only.
[https://github.com/MadhunishaBala/Anaemia-Detection-Application]
