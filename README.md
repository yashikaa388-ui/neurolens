---
title: NeuroLens
sdk: docker
app_port: 7860
---
# NeuroLens: Explainable Brain Tumor Detection

## 1. Setup (VS Code terminal)
```
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Use Python 3.10-3.11.

## 2. Get the dataset
Download "Brain Tumor MRI Dataset" (Masoud Nickparvar) from Kaggle and unzip so you have
`dataset/Training/...` and `dataset/Testing/...` (glioma, meningioma, notumor, pituitary).

## 3. Train (use Google Colab GPU if you have no GPU)
```
python train.py
```
Creates `model/brain_tumor_model.keras`, `model/class_names.json`, `model/test_report.txt`, `model/confusion_matrix.png`.

## 4. Run locally
```
python app.py
```
Open http://127.0.0.1:5000

## 5. Deploy free on Hugging Face Spaces
1. Create a Space at huggingface.co/new-space, SDK = Docker.
2. `git lfs install && git lfs track "*.keras"`
3. Push this folder (including `model/`, but not `dataset/`) to the Space repo.
4. The Space builds from the Dockerfile and gives you a public URL.
