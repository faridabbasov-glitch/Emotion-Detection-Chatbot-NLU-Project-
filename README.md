# 🎭 Emotion Detection Chatbot

A fine-tuned **DistilBERT** model that detects emotions in text across 6 categories, wrapped in an interactive chatbot interface.

Built during a Artificial Intelligence Internship at **Intern Intelligence** (Oct–Nov 2025).

---

## What it does

The model reads a sentence and detects which emotion it expresses:

| Emotion | Example |
|---------|---------|
| 😔 Sadness | *"I feel so lost and alone."* |
| 😊 Joy | *"Today was the best day of my life!"* |
| ❤️ Love | *"I love spending time with you."* |
| 😤 Anger | *"This is so frustrating!"* |
| 🤗 Fear | *"I am terrified of what might happen."* |
| 😲 Surprise | *"Wow, I did not expect that at all!"* |

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 92.50% |
| F1 Score | 92.46% |

### Per-class results:

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Sadness | 0.97 | 0.96 | 0.96 |
| Joy | 0.93 | 0.97 | 0.95 |
| Love | 0.91 | 0.74 | 0.81 |
| Anger | 0.91 | 0.94 | 0.92 |
| Fear | 0.91 | 0.85 | 0.88 |
| Surprise | 0.68 | 0.85 | 0.76 |

> **Note:** Love and Surprise scored lower because the dataset has significantly fewer examples for these classes (159 and 66 samples vs 500+ for others).

---

## How it works

```
User types a sentence
        ↓
Tokenizer converts text to numbers
        ↓
DistilBERT processes the input
        ↓
Softmax converts output to probabilities
        ↓
Highest probability → predicted emotion + confidence %
```

---

## Project Structure

```
emotion-detection-chatbot/
├── config.py                  # All hyperparameters and settings
├── train.py                   # Fine-tuning script
├── NLU_Emotion_Chatbot.ipynb  # Full notebook (train + chatbot)
└── README.md
```

---

## Tech Stack

- **Model:** DistilBERT (distilbert-base-uncased)
- **Dataset:** [emotion](https://huggingface.co/datasets/emotion) — 16,000 training samples, 6 classes
- **Libraries:** HuggingFace Transformers, PyTorch, Datasets, scikit-learn
- **Platform:** Google Colab (T4 GPU)

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 8 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Max Sequence Length | 128 |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/faridabbasov-glitch/emotion-detection-chatbot
```

### 2. Open the notebook
Open `NLU_Emotion_Chatbot.ipynb` in Google Colab.

### 3. Train the model
Run all cells from top to bottom. The model will be saved to your Google Drive automatically.

### 4. Chat
Run the last cell — a chatbot interface with **Send** and **Exit** buttons will appear.

---

## Sample Output

```
You     : I am so happy today!
Emotion : JOY (97.3%)
Bot     : That sounds wonderful! 😊

You     : I feel so sad and depressed.
Emotion : SADNESS (98.7%)
Bot     : I'm sorry you're feeling down. 😔
```
