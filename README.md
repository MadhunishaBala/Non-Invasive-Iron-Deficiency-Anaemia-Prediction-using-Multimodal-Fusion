# Non-Invasive-Iron-Deficiency-Anaemia-Detection-Using- Multimodal-Fusion

## About

Iron Deficiency Anaemia (IDA) remains one of the most under-diagnosed and common disorders globally, especially in low-resource settings, where there is limited access to laboratory testing. Non-invasive anaemia detection using images of conjunctiva, fingernails, and palm along with patient information have been explored, but current methods mostly rely on single modalities, and some of recent approaches have used fusion where all features are treated equally. These approaches fail to capture complex relationships between visual cues and patient data.

This research proposes a deep learning framework with attention-based multi-modal fusion to address these limitations. Convolutional Neural Networks and Multilayer Perceptrons are used to extract the image and textual data. Then a self-attention block is created to focus on the most useful parts of each type of data, and cross-attention block identifies the links between data types, which enhances feature representation and prediction accuracy. The goal is to make anaemia detection better, safer, and more accurate without needing invasive methods.

## Data
- **Palm images** (RGB, 224×224)
- **Nail images** (RGB, 224×224)
- **Metadata** (2 numerical features)
- **Labels**:
  - Binary classification (anemia vs. non‑anemia)
  - Continuous regression (hemoglobin value)


Application Deployed in : https://web-production-e16a5.up.railway.app/
