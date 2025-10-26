# Spam-Email-Detector-BERT

A lightweight BERT-based text classifier to detect spam versus ham (non-spam) messages, optimized for deployment and evaluation with TensorFlow and TensorFlow Lite.

---

## Description
This project trains a **BERT-style text classifier** to distinguish between spam and non-spam (ham) messages.  
It uses pre-trained BERT embeddings for sentence representation and a compact dense classifier head for efficient deployment.  
The workflow includes dataset balancing, CPU vs GPU benchmarking, performance evaluation (precision, recall, F1), confusion matrix visualization, and model export in both **SavedModel** and **TFLite** formats.

---

## Dataset
- **Source:** `spam.csv` (SMS Spam Collection dataset)  
- **Columns:** `Category`, `Message`  
- **Original Counts:**  
  - Ham: 4,825  
  - Spam: 747  
- **Balanced Training Set:**  
  - Downsampled ham to 4,000  
  - Upsampled spam to 4,000  
  - **Total Samples:** 8,000  

---

## Model Architecture
**Embedding:**  
- Text preprocessing and BERT embeddings from TensorFlow Hub (uncased, 768-dim pooled output)

**Classifier:**  
`Input(768) → Dropout(0.1) → Dense(1, Sigmoid)`  
- Total trainable parameters: ~769  

---

## Results and Performance
**Training Configuration:**  
- Batch Size: 64  
- Epochs: 10  
- Optimizer: Adam  
- Loss: Binary Cross-Entropy  
- Data Pipeline: `tf.data`  

**Evaluation Results:**  
- Final Test Accuracy: **0.9510**  
- Macro F1-Score: **≈ 0.95** (on 2,000-sample test set)  
- Confusion Matrix: correctly separates most spam and ham messages with minimal misclassifications.  

---

## Optimization for Deployment
The trained model was exported to TensorFlow’s **SavedModel** format and converted to **TFLite** for mobile and edge deployment.

- **SavedModel Export:** Successful  
- **TFLite Conversion:** Successful  
- **Quantized TFLite Model:** Similar base size (classifier head only, very lightweight)  

Example logs confirm both export and quantization completed without errors.

---

## Setup and Usage

### Clone
```bash
git clone https://github.com/Aaryan-Reddy/Spam-Email-Detector-BERT.git
cd Spam-Email-Detector-BERT
