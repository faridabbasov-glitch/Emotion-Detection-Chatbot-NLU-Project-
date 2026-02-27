# 🧠 Emotion Detection Chatbot - NLU Project

> Fine-tuned **DistilBERT** model for real-time emotion classification with an interactive chatbot interface.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FF9A00?style=flat&logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-92.70%25-brightgreen?style=flat)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.9268-brightgreen?style=flat)

---

## 📌 Overview

This project fine-tunes a **DistilBERT** transformer model on the [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion) to classify text into 6 emotional categories. It also features a real-time **command-line chatbot** that detects the user's emotion and responds with human-like replies.

---

## 🎯 Emotions Detected

| Label | Emotion |
|-------|---------|
| 😢 | Sadness |
| 😄 | Joy |
| ❤️ | Love |
| 😡 | Anger |
| 😨 | Fear |
| 😲 | Surprise |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.9270 |
| **Weighted F1** | 0.9268 |

> Evaluated on the held-out test split of the Emotion Dataset.

---

## 🏗️ Model Architecture

```
DistilBERT (pretrained, fine-tuned)
└── Classification Head
    └── Dense(6, softmax) → Emotion Label
```

- **Base model:** `distilbert-base-uncased`
- **Training framework:** Hugging Face `Trainer` API
- **Optimizer:** AdamW with linear warmup scheduler

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/faridabbasov-glitch/Emotion-Detection-Chatbot-NLU-Project.git
cd Emotion-Detection-Chatbot-NLU-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook NLU_Intent_classification.ipynb
```

---

## 💬 Chatbot Demo

```
You: I just got accepted to my dream university!
Bot: 😄 That sounds wonderful! I'm glad you're feeling happy!

You: I'm really nervous about tomorrow's interview.
Bot: 🤗 That sounds concerning. I hope everything works out for you.

You: exit
Bot: Goodbye! 👋
```

---

## 📁 Project Structure

```
├── NLU_Intent_classification.ipynb   # Training, evaluation & chatbot
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Model | DistilBERT (Hugging Face) |
| Framework | PyTorch |
| Training | Hugging Face Trainer API |
| Dataset | dair-ai/emotion |
| Interface | Jupyter Notebook |

---

## 📦 Requirements

```
transformers>=4.30
torch>=2.0
datasets>=2.12
scikit-learn>=1.2
```
